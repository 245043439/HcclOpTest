"""Single-case HCCL benchmark. All ranks must receive identical arguments."""
import argparse
import json
import os
import re
import statistics
import sys
import socket
import traceback
import faulthandler
import time

# Install diagnostics before importing torch so import/device initialization failures
# are captured too. These files are collected from each container by the launcher.
STAGE = "import torch/torch_npu"
TRACE_FILE = None
trace_dir = os.environ.get("HCCL_BENCH_DIAG_DIR")
if trace_dir:
    os.makedirs(trace_dir, exist_ok=True)
    TRACE_FILE = open(os.path.join(trace_dir, f"rank_{os.environ.get('RANK', 'unknown')}.traceback.log"),
                      "a", buffering=1, encoding="utf-8")


def report_exception(exc_type, exc, tb):
    context = (f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
               f"host={socket.gethostname()} pid={os.getpid()} "
               f"run={os.environ.get('HCCL_BENCH_RUN_ID', 'unknown')} "
               f"rank={os.environ.get('RANK', 'unknown')} "
               f"local_rank={os.environ.get('LOCAL_RANK', 'unknown')} stage={STAGE}\n"
               f"arguments={sys.argv[1:]!r}\n")
    message = context + "".join(traceback.format_exception(exc_type, exc, tb))
    sys.stderr.write(message)
    sys.stderr.flush()
    if TRACE_FILE:
        TRACE_FILE.write(message)
        TRACE_FILE.flush()


sys.excepthook = report_exception
faulthandler.enable(file=TRACE_FILE or sys.stderr, all_threads=True)
stack_interval = int(os.environ.get("HCCL_BENCH_STACK_INTERVAL", "60"))
if stack_interval > 0:
    faulthandler.dump_traceback_later(stack_interval, repeat=True,
                                    file=TRACE_FILE or sys.stderr)

import torch
import torch_npu  # noqa: F401 - registers the NPU backend
import torch.distributed as dist


OPS = ("allreduce", "alltoall", "reducescatter", "allgather",
       "batch_send_recv", "broadcast", "send_recv")


def create_groups(group_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size % group_size:
        raise ValueError("world size must be divisible by group size")
    selected, selected_start = None, None
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            selected, selected_start = group, start
    return selected, selected_start


def max_latency(local_ms, group, participant_ranks=None):
    rank = dist.get_rank()
    selected = participant_ranks is None or rank in participant_ranks
    value = torch.tensor([local_ms if selected else 0.0], device="npu", dtype=torch.float64)
    # For point-to-point, mask ranks that do not transfer data, then reduce across
    # the existing group; this avoids creating a new communicator per iteration.
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
    return value.item()


def benchmark(fn, group, warmup, repeat, participants=None, graph_runner=None,
              progress_label="", is_group_leader=False):
    global STAGE
    invoke = graph_runner or fn
    if is_group_leader:
        print(f"[{progress_label}] warmup started ({warmup} iterations)", flush=True)
    for iteration in range(warmup):
        STAGE = f"{progress_label} warmup {iteration + 1}/{warmup}"
        invoke()
    torch.npu.synchronize()
    dist.barrier(group=group)
    if is_group_leader:
        print(f"[{progress_label}] measurement started ({repeat} iterations)", flush=True)
    samples = []
    progress_step = max(1, repeat // 5)
    for iteration in range(repeat):
        STAGE = f"{progress_label} iteration {iteration + 1}/{repeat} barrier"
        dist.barrier(group=group)
        torch.npu.synchronize()
        STAGE = f"{progress_label} iteration {iteration + 1}/{repeat} communication/synchronize"
        start = time.perf_counter()
        invoke()
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        STAGE = f"{progress_label} iteration {iteration + 1}/{repeat} latency reduction"
        samples.append(max_latency(elapsed, group, participants))
        if is_group_leader and (len(samples) % progress_step == 0 or len(samples) == repeat):
            print(f"[{progress_label}] progress {len(samples)}/{repeat}; "
                  f"latest_max_ms={samples[-1]:.3f}", flush=True)
    return {
        "mean_ms": statistics.fmean(samples), "min_ms": min(samples),
        "max_ms": max(samples), "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, int(0.95 * len(samples)) - 1)],
    }


def graph_callable(fn, mode):
    if mode == "eager":
        return fn
    if mode == "ascend_ir":
        try:
            import torchair
        except ImportError as exc:
            raise RuntimeError("TorchAir is unavailable") from exc
        patch = getattr(torchair, "patch_for_hcom", None)
        if patch is None:
            raise RuntimeError("TorchAir patch_for_hcom API is unavailable")
        patch()
        backend = torchair.get_npu_backend()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        # Trigger graph compilation before measurement and propagate graph breaks.
        compiled()
        torch.npu.synchronize()
        return compiled
    if mode == "aclgraph":
        graph_type = getattr(torch.npu, "NPUGraph", None)
        graph_context = getattr(torch.npu, "graph", None)
        if graph_type is None or graph_context is None:
            raise RuntimeError("torch_npu aclgraph capture API is unavailable")
        # Initialize HCCL communicators and graph-safe workspaces before capture.
        fn()
        torch.npu.synchronize()
        graph = graph_type()
        with graph_context(graph):
            fn()
        torch.npu.synchronize()
        return graph.replay
    raise ValueError(f"unknown execution mode: {mode}")


def run_operator(op, x, group, p, rank, group_start, group_rank, numel, device):
    if op == "allreduce":
        return lambda: dist.all_reduce(x, group=group), numel, numel, None
    if op == "broadcast":
        return lambda: dist.broadcast(x, src=group_start, group=group), numel, numel, None
    if op == "alltoall":
        out = torch.empty_like(x)
        return lambda: dist.all_to_all_single(out, x, group=group), numel, numel, None
    if op == "reducescatter":
        out = torch.empty(numel // p, device=device, dtype=x.dtype)
        return lambda: dist.reduce_scatter_tensor(out, x, group=group), numel, numel // p, None
    if op == "allgather":
        inp = torch.ones(numel // p, device=device, dtype=x.dtype)
        out = torch.empty_like(x)
        return lambda: dist.all_gather_into_tensor(out, inp, group=group), numel // p, numel, None
    if op == "batch_send_recv":
        recv = torch.empty_like(x)
        dst = group_start + (group_rank + 1) % p
        src = group_start + (group_rank - 1 + p) % p
        def ring():
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, x, dst, group),
                dist.P2POp(dist.irecv, recv, src, group),
            ])
            for req in reqs:
                req.wait()
        return ring, numel, numel, None
    if op == "send_recv":
        recv = torch.empty_like(x)
        src, dst = group_start, group_start + p - 1
        def pair():
            if rank == src:
                dist.send(x, dst=dst, group=group)
            elif rank == dst:
                dist.recv(recv, src=src, group=group)
            # All intermediate ranks remain in the common per-iteration barrier.
        return pair, numel, numel, [src, dst]
    raise ValueError(f"unknown operator: {op}")


def main():
    global STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=float)
    parser.add_argument("--size-bytes", type=int)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--operator", choices=OPS, action="append")
    parser.add_argument("--execution-mode", choices=("eager", "ascend_ir", "aclgraph"), default="eager")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--result-json")
    args = parser.parse_args()
    if (args.size_bytes is None) == (args.size_mb is None):
        parser.error("specify exactly one of --size-bytes or --size-mb")
    local_rank = int(os.environ["LOCAL_RANK"])
    STAGE = "set NPU device"
    torch.npu.set_device(local_rank)
    STAGE = "init_process_group(hccl)"
    print(f"[rank {os.environ.get('RANK')}] {STAGE}", flush=True)
    dist.init_process_group(backend="hccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    print(f"[rank {rank}/{world_size}] HCCL process group initialized", flush=True)
    p = args.group_size
    if p < 2 or p > world_size or world_size % p:
        raise ValueError("invalid group size for world size")
    if args.repeat <= 0 or args.warmup < 0:
        raise ValueError("repeat must be positive and warmup nonnegative")
    STAGE = f"create subgroups size={p}"
    group, group_start = create_groups(p)
    group_rank = dist.get_rank(group=group)
    print(f"[rank {rank}] subgroup creation complete; group_size={p}", flush=True)
    device, dtype = torch.device(f"npu:{local_rank}"), torch.float32
    size_bytes = args.size_bytes if args.size_bytes is not None else round(args.size_mb * 1024**2)
    if size_bytes <= 0 or size_bytes % torch.empty((), dtype=dtype).element_size():
        raise ValueError("size must be positive and divisible by float32 element size")
    numel = size_bytes // torch.empty((), dtype=dtype).element_size()
    op_list = args.operator or list(OPS[:5])
    requested_expansion = os.environ.get("MATRIX_EXPANSION_MODE", "DEFAULT")
    # Each rank writes its subgroup result into a fixed numeric row. Gathering
    # tensors avoids backend/version-dependent Python object collectives.
    result_width = 8
    result_tensor = torch.zeros((len(op_list), result_width), device=device,
                                dtype=torch.float64)
    for op_index, op in enumerate(op_list):
        if group_rank == 0:
            print(f"[case] op={op} size_bytes={size_bytes} group_size={p} "
                  f"execution_mode={args.execution_mode} "
                  f"expansion_mode={requested_expansion} preparing", flush=True)
        # AllGather input follows the legacy total-output-size convention.
        if op == "allgather":
            if numel % p:
                raise ValueError("allgather total output elements must divide group size")
        elif op in ("alltoall", "reducescatter") and numel % p:
            raise ValueError(f"{op} elements must divide group size")
        x = torch.ones(numel, device=device, dtype=dtype)
        fn, in_elems, out_elems, participants = run_operator(
            op, x, group, p, rank, group_start, group_rank, numel, device)
        STAGE = f"op={op} mode={args.execution_mode} graph preparation"
        run_fn = graph_callable(fn, args.execution_mode)
        metrics = benchmark(run_fn, group, args.warmup, args.repeat, participants,
                            progress_label=f"op={op} group={group_start}:{group_start+p-1}",
                            is_group_leader=(group_rank == 0))
        status, reason = "success", ""
        if group_rank == 0:
            metric_values = [metrics.get(key, 0.0) for key in
                             ("mean_ms", "min_ms", "max_ms", "p50_ms", "p95_ms")]
            result_tensor[op_index] = torch.tensor(
                [in_elems * 4, out_elems * 4, *metric_values,
                 1.0 if status == "success" else 2.0],
                device=device, dtype=torch.float64)
    print(f"[rank {rank}] benchmark operations complete; gathering subgroup results",
          flush=True)
    gathered_tensors = [torch.empty_like(result_tensor) for _ in range(world_size)]
    STAGE = "gather subgroup results"
    dist.all_gather(gathered_tensors, result_tensor)
    print(f"[rank {rank}] subgroup result gathering complete", flush=True)
    if rank == 0 and args.result_json:
        all_results = []
        for group_leader in range(0, world_size, p):
            values = gathered_tensors[group_leader].cpu().tolist()
            for op_index, op in enumerate(op_list):
                row = values[op_index]
                success = row[7] == 1.0
                all_results.append({
                    "operator": op, "group_start": group_leader,
                    "group_end": group_leader + p - 1,
                    "input_bytes": int(row[0]), "output_bytes": int(row[1]),
                    "metrics": ({"mean_ms": row[2], "min_ms": row[3],
                                 "max_ms": row[4], "p50_ms": row[5],
                                 "p95_ms": row[6]} if success else {}),
                    "status": "success" if success else "failed",
                    "reason": "" if success else "benchmark failed; see rank logs",
                    "requested_expansion_mode": requested_expansion,
                    "actual_expansion_mode": "unknown",
                })
        props = torch.npu.get_device_properties(local_rank)
        metadata = {
            "hardware": getattr(props, "name", "unknown"),
            "cann_version": os.environ.get("CANN_VERSION", "unknown"),
            "torch_version": torch.__version__,
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            "torchair_version": "unknown",
        }
        try:
            import torchair
            metadata["torchair_version"] = getattr(torchair, "__version__", "unknown")
        except ImportError:
            pass
        if metadata["cann_version"] == "unknown":
            for version_file in ("/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
                                 "/usr/local/Ascend/latest/version.cfg"):
                try:
                    content = open(version_file, encoding="utf-8").read()
                    match = re.search(r"(?m)^Version=(.+)$", content)
                    if match:
                        metadata["cann_version"] = match.group(1).strip()
                        break
                except OSError:
                    continue
        payload = {"world_size": world_size, "group_size": p,
                   "metadata": metadata, "results": all_results}
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        print(f"[rank 0] structured results written to {args.result_json}", flush=True)
    STAGE = "destroy process group"
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
