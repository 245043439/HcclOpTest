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
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-$TIMEOUT_SEC}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
MATRIX_OPERATOR="${MATRIX_OPERATOR:-}"
MATRIX_EXECUTION_MODE="${MATRIX_EXECUTION_MODE:-eager}"
MATRIX_EXPANSION_MODE="${MATRIX_EXPANSION_MODE:-DEFAULT}"
MATRIX_SIZE_BYTES="${MATRIX_SIZE_BYTES:-}"
MATRIX_DTYPE="${MATRIX_DTYPE:-float32}"
CASE_RESULT_FILE="${CASE_RESULT_FILE:-}"
MATRIX_CASES_JSON="${MATRIX_CASES_JSON:-}"

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
REMOTE_RESULT_FILE="/tmp/hccl_result_${RUN_ID}_node.json"
LOG_DIR="${LOG_DIR:-./hccl_logs/$RUN_ID}"
mkdir -p "$LOG_DIR"
WORKER_SCRIPT="$(mktemp)"
PIDS=()
ACTIVE=()
PREPARED=0
CLEANING=0
CLEANUP_STATUS=0
WATCHDOG_PID=''

cat > "$WORKER_SCRIPT" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail
MODE="$1"; shift
NODE_RANK="$1"; NNODES="$2"; MASTER_ADDR="$3"; MASTER_PORT="$4"
CONTAINER="$5"; IMAGE="$6"; DOCKER_SCRIPT="$7"; BENCH_SCRIPT="$8"
NPROC="$9"; SIZE_MB="${10}"; GROUP_SIZE="${11}"; WARMUP="${12}"; REPEAT="${13}"; RUN_ID="${14}"
OPERATOR="${15}"; EXECUTION_MODE="${16}"; EXPANSION_MODE="${17}"; SIZE_BYTES="${18}"; REMOTE_RESULT="${19}"; DTYPE="${20}"; CASES_JSON="${21:-}"; TIMEOUT_SEC="${22:-900}"
DIAG_DIR="/tmp/hccl_diag_${RUN_ID}"
worker_error() {
    local status="$1" line="$2" command="$3" frame=0
    printf '[Node %s] mode=%s failed exit=%s line=%s command=%s\n' "$NODE_RANK" "$MODE" "$status" "$line" "$command" >&2
    while caller "$frame"; do frame=$((frame + 1)); done >&2
}
trap 'worker_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
# Expansion errors (including nounset) bypass ERR; retain an exit diagnostic too.
trap 'status=$?; if (( status != 0 )); then printf "[Node %s] mode=%s exited %s; inspect worker.sh and preceding error\n" "$NODE_RANK" "$MODE" "$status" >&2; fi' EXIT

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
        set +e
        docker exec -e "HCCL_BENCH_RUN_ID=$RUN_ID" -e "CASE_TIMEOUT_SEC=$TIMEOUT_SEC" \
            -e "HCCL_BENCH_DIAG_DIR=$DIAG_DIR" -e PYTHONUNBUFFERED=1 \
            -e PYTHONFAULTHANDLER=1 "$CONTAINER" bash -lc '
            set -Eeuo pipefail
            export HCCL_NPU_SOCKET_PORT_RANGE=auto
            if [[ "$5" == DEFAULT ]]; then unset HCCL_OP_EXPANSION_MODE; else export HCCL_OP_EXPANSION_MODE="$5"; fi
            export MATRIX_EXPANSION_MODE="$5"
            size_args=()
            case_args=()
            if [[ -n "${17}" ]]; then case_args=(--cases-json "${17}"); else
                size_args=(--size-mb "$9")
                [[ -z "${10}" ]] || size_args=(--size-bytes "${10}")
                [[ -z "$4" ]] || case_args=(--operator "$4")
            fi
            exec torchrun \
                --nnodes="$1" --node_rank="$2" --nproc_per_node="$3" \
                --master_addr="$6" --master_port="$7" \
                "$8" "${size_args[@]}" --group-size "${11}" \
                --warmup "${12}" --repeat "${13}" \
                --execution-mode "${14}" "${case_args[@]}" --result-json "${15}" --dtype "${16}"
        ' bash "$NNODES" "$NODE_RANK" "$NPROC" "$OPERATOR" "$EXPANSION_MODE" \
          "$MASTER_ADDR" "$MASTER_PORT" "$BENCH_SCRIPT" "$SIZE_MB" "$SIZE_BYTES" \
          "$GROUP_SIZE" "$WARMUP" "$REPEAT" "$EXECUTION_MODE" "$REMOTE_RESULT" "$DTYPE" "$CASES_JSON"
        run_status=$?
        set -e
        if docker exec "$CONTAINER" bash -lc "compgen -G '$DIAG_DIR/timeout_rank_*' >/dev/null"; then
            run_status=124
        fi
        if (( NODE_RANK == 0 )) && [[ -n "${CASE_RESULT_FILE:-}" ]]; then
            docker cp "$CONTAINER:$REMOTE_RESULT" "$CASE_RESULT_FILE" || true
        fi
        (( run_status == 0 )) || exit "$run_status"
        echo "[Node $NODE_RANK] Finished"
        ;;
    collect)
        if docker exec "$CONTAINER" test -d "$DIAG_DIR"; then
            docker exec "$CONTAINER" tar -C "$DIAG_DIR" -cf - .
        else
            tar -cf - --files-from /dev/null
        fi
        ;;
    cleanup)
        cleanup_run
        echo "[Node $NODE_RANK] Cleanup complete"
        ;;
    *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
REMOTE_SCRIPT
cp "$WORKER_SCRIPT" "$LOG_DIR/worker.sh"

invoke() {
    local idx="$1" mode="$2" ip="${NODES[$1]}" remote_args
    local args=("$mode" "$idx" "$NNODES" "$MASTER_ADDR" "$MASTER_PORT"
        "$CONTAINER_NAME" "$IMAGE_NAME" "$DOCKER_SCRIPT" "$BENCH_SCRIPT"
        "$NPROC_PER_NODE" "$SIZE_MB" "$GROUP_SIZE" "$WARMUP" "$REPEAT" "$RUN_ID"
        "$MATRIX_OPERATOR" "$MATRIX_EXECUTION_MODE" "$MATRIX_EXPANSION_MODE" "$MATRIX_SIZE_BYTES"
        "$REMOTE_RESULT_FILE" "$MATRIX_DTYPE" "$MATRIX_CASES_JSON" "$CASE_TIMEOUT_SEC")
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
        for pid in "${cpids[@]}"; do
            if ! wait "$pid"; then
                echo "WARNING: cleanup failed for PID=$pid; inspect cleanup logs" >&2
                CLEANUP_STATUS=1
            fi
        done
        for idx in "${!NODES[@]}"; do
            if ! invoke "$idx" collect > "$LOG_DIR/node_${idx}.diagnostics.tar" 2> "$LOG_DIR/collect_${idx}.log"; then
                echo "WARNING: node $idx diagnostic collection failed; see collect_${idx}.log" >&2
            fi
        done
    fi
    rm -f "$WORKER_SCRIPT"
}
on_exit() {
    local status=$?
    cleanup
    if (( CLEANUP_STATUS != 0 )); then exit 70; fi
    exit "$status"
}
trap on_exit EXIT
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
        exit 69
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
        exit "$status"
    fi
    echo "[Node $node] SUCCESS"
    next=()
    for pid in "${ACTIVE[@]}"; do
        [[ "$pid" == "$completed" ]] || next+=("$pid")
    done
    ACTIVE=("${next[@]}")
done

echo "All nodes completed successfully. Logs: $LOG_DIR"
