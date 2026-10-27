#!/usr/bin/env python3
"""Minimal AllReduce benchmark with eager and graph execution modes."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time


DTYPE_BYTES = {"float16": 2, "float32": 4, "bfloat16": 2}


def sample_summary(samples):
    ordered = sorted(samples)
    quartiles = [ordered[(len(ordered) * index + 3) // 4 - 1] for index in range(1, 5)]
    return statistics.fmean(samples), quartiles, statistics.pvariance(samples)


def graph_callable(fn, mode, torch):
    """Prepare graphs before warmup, following run_bench.py's execution modes."""
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
        compiled = torch.compile(fn, backend=torchair.get_npu_backend(), fullgraph=True)
        compiled()
        torch.npu.synchronize()
        return compiled
    if mode == "aclgraph":
        graph_type = getattr(torch.npu, "NPUGraph", None)
        graph_context = getattr(torch.npu, "graph", None)
        if graph_type is None or graph_context is None:
            raise RuntimeError("torch_npu aclgraph capture API is unavailable")
        # Initialize HCCL communicators and workspaces before capture.
        fn()
        torch.npu.synchronize()
        graph = graph_type()
        with graph_context(graph):
            fn()
        torch.npu.synchronize()
        return graph.replay
    raise ValueError(f"unknown execution mode: {mode}")


def run(args):
    if args.size_bytes <= 0 or args.size_bytes % DTYPE_BYTES[args.dtype]:
        raise ValueError("size-bytes must be positive and divisible by dtype element size")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be nonnegative and repeat must be positive")
    if args.msprof and args.python_stack:
        raise ValueError("--msprof and --python-stack cannot be used together")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise ValueError("launch with torchrun and at least two ranks")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"rank_{rank}.json"
    if result_path.exists():
        raise ValueError(f"refusing to overwrite {result_path}; use a fresh output directory")
    if args.msprof:
        profile_dir = args.output.resolve() / "profiler" / f"rank_{rank}"
        profile_dir.mkdir(parents=True, exist_ok=False)
        command = ["msprof", f"--output={profile_dir}", "--task-time=on", "--hccl=on",
                   "--ai-core=off", "--type=text", sys.executable, str(Path(__file__).resolve()),
                   "run", "--profiled", "--output", str(args.output.resolve()),
                   "--size-bytes", str(args.size_bytes), "--dtype", args.dtype,
                   "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                   "--execution-mode", args.execution_mode]
        return subprocess.run(command, check=False).returncode

    # CLI help deliberately does not import any NPU dependencies.
    import torch
    import torch_npu
    import torch.distributed as dist

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    try:
        x = torch.zeros(args.size_bytes // DTYPE_BYTES[args.dtype],
                        dtype=getattr(torch, args.dtype), device=f"npu:{local_rank}")
        run_fn = graph_callable(lambda: dist.all_reduce(x), args.execution_mode, torch)
        for _ in range(args.warmup):
            run_fn()
        torch.npu.synchronize()
        dist.barrier()
        samples = []
        profiler = nullcontext()
        if args.python_stack:
            stack_dir = args.output / "python_profiler" / f"rank_{rank}"
            stack_dir.mkdir(parents=True, exist_ok=False)
            profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=args.repeat, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(stack_dir)),
                with_stack=True,
            )
        with profiler:
            for _ in range(args.repeat):
                dist.barrier()
                torch.npu.synchronize()
                start = time.perf_counter()
                run_fn()
                torch.npu.synchronize()
                samples.append((time.perf_counter() - start) * 1_000_000)
                if args.python_stack:
                    profiler.step()
        payload = {
            "schema_version": 1, "rank": rank, "local_rank": local_rank,
            "world_size": world_size, "size_bytes": args.size_bytes, "dtype": args.dtype,
            "warmup": args.warmup, "repeat": args.repeat, "execution_mode": args.execution_mode,
            "expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE", "DEFAULT"),
            "profiled": args.profiled or args.python_stack,
            "python_stack": args.python_stack, "samples_us": samples,
            "metadata": {"hostname": socket.gethostname(),
                         "hardware": getattr(torch.npu.get_device_properties(local_rank), "name", "unknown"),
                         "torch_version": torch.__version__,
                         "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
                         "cann_version": os.environ.get("CANN_VERSION", "unknown")},
        }
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        print(f"rank {rank}: saved {len(samples)} samples to {result_path}", flush=True)
        mean, quartiles, variance = sample_summary(samples)
        print(f"rank {rank}: mean={mean:.3f} us; "
              f"Q1={quartiles[0]:.3f}, Q2={quartiles[1]:.3f}, "
              f"Q3={quartiles[2]:.3f}, Q4={quartiles[3]:.3f} us; "
              f"variance={variance:.3f} us^2", flush=True)
    finally:
        dist.destroy_process_group()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="launch under torchrun, optionally wrap each rank with msprof")
    runner.add_argument("--size-bytes", type=int, default=1024 ** 2)
    runner.add_argument("--dtype", choices=DTYPE_BYTES, default="float32")
    runner.add_argument("--warmup", type=int, default=10)
    runner.add_argument("--repeat", type=int, default=50)
    runner.add_argument("--execution-mode", choices=("eager", "ascend_ir", "aclgraph"), default="eager")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--msprof", action="store_true")
    runner.add_argument("--python-stack", action="store_true",
                        help="collect Python call stacks with torch_npu.profiler")
    runner.add_argument("--profiled", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
