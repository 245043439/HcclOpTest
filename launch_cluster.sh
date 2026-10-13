#!/usr/bin/env bash
set -Eeuo pipefail

MASTER_ADDR="${MASTER_ADDR:-141.61.94.99}"
MASTER_PORT="${MASTER_PORT:-29600}"
CONTAINER_NAME="${CONTAINER_NAME:-hccl_bench}"
IMAGE_NAME="${IMAGE_NAME:?请设置 IMAGE_NAME 环境变量}"
DOCKER_SCRIPT="${DOCKER_SCRIPT:-/mnt/share/h00889099/docker-gen.sh}"
BENCH_SCRIPT="${BENCH_SCRIPT:-/mnt/share/h00889099/hccl_database_code/run_bench.py}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SIZE_MB="${SIZE_MB:-64}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
SSH_USER="${SSH_USER:-root}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"

if (( $# < 1 )); then
    echo "Usage: IMAGE_NAME=image bash $0 <node1_ip> [node2_ip ...]" >&2
    exit 2
fi
[[ "$TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] || { echo 'TIMEOUT_SEC must be positive integer' >&2; exit 2; }

NODES=("$MASTER_ADDR" "$@")
NNODES=${#NODES[@]}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
GROUP_SIZE="${GROUP_SIZE:-$WORLD_SIZE}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
LOG_DIR="${LOG_DIR:-./hccl_logs/$RUN_ID}"
mkdir -p "$LOG_DIR"
WORKER_SCRIPT="$(mktemp)"
PIDS=()
ACTIVE=()
PREPARED=0
CLEANING=0
WATCHDOG_PID=''

cat > "$WORKER_SCRIPT" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail
MODE="$1"; shift
NODE_RANK="$1"; NNODES="$2"; MASTER_ADDR="$3"; MASTER_PORT="$4"
CONTAINER="$5"; IMAGE="$6"; DOCKER_SCRIPT="$7"; BENCH_SCRIPT="$8"
NPROC="$9"; SIZE_MB="${10}"; GROUP_SIZE="${11}"; WARMUP="${12}"; REPEAT="${13}"; RUN_ID="${14}"

# Python is already required by torchrun. Select only actual python benchmark processes,
# not grep, the preflight shell, or unrelated Python tasks.
check_stale() {
    docker exec -i "$CONTAINER" python3 - "$BENCH_SCRIPT" <<'PY'
import os, sys
bench = sys.argv[1]
found = []
for item in os.listdir('/proc'):
    if not item.isdigit() or int(item) == os.getpid():
        continue
    try:
        args = open(f'/proc/{item}/cmdline', 'rb').read().split(b'\0')
        if not args:
            continue
        parts = [a.decode(errors='replace') for a in args if a]
        is_torchrun = any(os.path.basename(a) in ('torchrun', 'torchrun.exe') for a in parts[1:]) or os.path.basename(parts[0]) == 'torchrun'
        is_bench = any(a == bench or a.endswith('/run_bench.py') for a in parts[1:])
        if is_torchrun or is_bench:
            found.append((item, ' '.join(parts)[:240]))
    except (OSError, ProcessLookupError):
        pass
if found:
    for pid, cmd in found:
        print(f'Existing benchmark process PID={pid}: {cmd}', file=sys.stderr)
    sys.exit(1)
PY
}

cleanup_run() {
    # The identifier is inherited by torchrun and its subprocesses.
    # No broad pkill and no stopping/removing the container.
    if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then return 0; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]]; then return 0; fi
    docker exec -i "$CONTAINER" python3 - "$RUN_ID" <<'PY'
import os, signal, sys, time
marker = b'HCCL_BENCH_RUN_ID=' + sys.argv[1].encode()
me = os.getpid()
def targets():
    result = []
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) == me:
            continue
        try:
            env = open(f'/proc/{name}/environ', 'rb').read().split(b'\0')
            if marker in env:
                result.append(int(name))
        except (OSError, ProcessLookupError):
            pass
    return result
for pid in targets():
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
for _ in range(30):
    if not targets(): break
    time.sleep(.1)
for pid in targets():
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
PY
}

case "$MODE" in
    prepare)
        docker info >/dev/null
        if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
            echo "[Node $NODE_RANK] Creating container"
            bash "$DOCKER_SCRIPT" "$CONTAINER" "$IMAGE"
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]]; then
            docker start "$CONTAINER" >/dev/null
        fi
        NETWORK="$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$CONTAINER")"
        [[ "$NETWORK" == host ]] || { echo "[Node $NODE_RANK] expected host network, got $NETWORK" >&2; exit 1; }
        docker exec "$CONTAINER" test -f "$BENCH_SCRIPT"
        docker exec "$CONTAINER" bash -lc 'command -v torchrun >/dev/null && command -v python3 >/dev/null'
        check_stale
        echo "[Node $NODE_RANK] READY"
        ;;
    run)
        echo "[Node $NODE_RANK] Starting benchmark RUN_ID=$RUN_ID"
        docker exec -e "HCCL_BENCH_RUN_ID=$RUN_ID" "$CONTAINER" bash -lc '
            export HCCL_NPU_SOCKET_PORT_RANGE=auto
            exec torchrun \
                --nnodes="$1" --node_rank="$2" --nproc_per_node="$3" \
                --master_addr="$4" --master_port="$5" \
                "$6" --size-mb "$7" --group-size "$8" \
                --warmup "$9" --repeat "${10}"
        ' bash "$NNODES" "$NODE_RANK" "$NPROC" "$MASTER_ADDR" "$MASTER_PORT" \
          "$BENCH_SCRIPT" "$SIZE_MB" "$GROUP_SIZE" "$WARMUP" "$REPEAT"
        echo "[Node $NODE_RANK] Finished"
        ;;
    cleanup)
        cleanup_run
        echo "[Node $NODE_RANK] Cleanup complete"
        ;;
    *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
REMOTE_SCRIPT

invoke() {
    local idx="$1" mode="$2" ip="${NODES[$1]}" remote_args
    local args=("$mode" "$idx" "$NNODES" "$MASTER_ADDR" "$MASTER_PORT"
        "$CONTAINER_NAME" "$IMAGE_NAME" "$DOCKER_SCRIPT" "$BENCH_SCRIPT"
        "$NPROC_PER_NODE" "$SIZE_MB" "$GROUP_SIZE" "$WARMUP" "$REPEAT" "$RUN_ID")
    if (( idx == 0 )); then
        bash "$WORKER_SCRIPT" "${args[@]}"
    else
        printf -v remote_args '%q ' "${args[@]}"
        ssh -T -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
            -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
            "$SSH_USER@$ip" "bash -s -- $remote_args" < "$WORKER_SCRIPT"
    fi
}

cleanup() {
    local idx pid
    (( CLEANING == 0 )) || return 0
    CLEANING=1
    trap - EXIT INT TERM HUP
    [[ -z "$WATCHDOG_PID" ]] || { kill "$WATCHDOG_PID" 2>/dev/null || true; wait "$WATCHDOG_PID" 2>/dev/null || true; }
    # Kill local SSH/docker-exec clients; remote container tasks need separate cleanup.
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    if (( PREPARED )); then
        echo '========== Global cleanup =========='
        local cpids=()
        for idx in "${!NODES[@]}"; do
            ( invoke "$idx" cleanup > "$LOG_DIR/cleanup_${idx}.log" 2>&1 ) &
            cpids+=("$!")
        done
        for pid in "${cpids[@]}"; do wait "$pid" || echo "WARNING: cleanup failed for PID=$pid; inspect cleanup logs" >&2; done
    fi
    rm -f "$WORKER_SCRIPT"
}
trap cleanup EXIT
trap 'echo "Interrupted" >&2; exit 130' INT
trap 'echo "Terminated" >&2; exit 143' TERM
trap 'echo "Hangup" >&2; exit 129' HUP

printf 'RUN_ID=%s NNODES=%s WORLD_SIZE=%s GROUP_SIZE=%s TIMEOUT=%ss\n' \
    "$RUN_ID" "$NNODES" "$WORLD_SIZE" "$GROUP_SIZE" "$TIMEOUT_SEC"
printf 'Container=%s Image=%s Logs=%s\n' "$CONTAINER_NAME" "$IMAGE_NAME" "$LOG_DIR"

# Preflight is fully sequential: no torchrun starts until all nodes are ready.
echo '========== Preflight =========='
PREPARED=1
for i in "${!NODES[@]}"; do
    echo "Checking Node $i (${NODES[$i]})"
    if ! invoke "$i" prepare > "$LOG_DIR/prepare_${i}.log" 2>&1; then
        echo "Preflight FAILED on Node $i:" >&2
        tail -n 30 "$LOG_DIR/prepare_${i}.log" >&2
        exit 1
    fi
    echo "[Node $i] READY"
done

echo '========== Launch =========='
for i in "${!NODES[@]}"; do
    ( invoke "$i" run > "$LOG_DIR/node_${i}.log" 2>&1 ) &
    PIDS+=("$!")
    ACTIVE+=("$!")
    echo "Started Node $i (${NODES[$i]}), local PID=$!"
done

# A watchdog is a separate waitable child, so wait -n reports success/failure
# as soon as any worker finishes, regardless of node index.
( sleep "$TIMEOUT_SEC"; exit 124 ) &
WATCHDOG_PID=$!
ACTIVE+=("$WATCHDOG_PID")

# Bash 5.1+ required for wait -n -p.
while (( ${#ACTIVE[@]} > 1 )); do
    completed=''
    if wait -n -p completed "${ACTIVE[@]}"; then
        status=0
    else
        status=$?
    fi
    if [[ -z "$completed" ]]; then
        echo "Interrupted while waiting (status=$status)" >&2
        exit 1
    fi
    if [[ "$completed" == "$WATCHDOG_PID" ]]; then
        echo "Global timeout after ${TIMEOUT_SEC}s" >&2
        exit 124
    fi
    for i in "${!PIDS[@]}"; do
        if [[ "${PIDS[$i]}" == "$completed" ]]; then
            node="$i"
            break
        fi
    done
    if (( status != 0 )); then
        echo "[Node $node] FAILED (exit=$status)" >&2
        tail -n 30 "$LOG_DIR/node_${node}.log" >&2 || true
        exit 1
    fi
    echo "[Node $node] SUCCESS"
    next=()
    for pid in "${ACTIVE[@]}"; do
        [[ "$pid" == "$completed" ]] || next+=("$pid")
    done
    ACTIVE=("${next[@]}")
done

echo "All nodes completed successfully. Logs: $LOG_DIR"
