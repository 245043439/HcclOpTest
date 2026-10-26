#!/usr/bin/env python3
"""Minimal eager AllReduce benchmark."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


DTYPE_BYTES = {"float16": 2, "float32": 4, "bfloat16": 2}


def run(args):
    if args.size_bytes <= 0 or args.size_bytes % DTYPE_BYTES[args.dtype]:
        raise ValueError("size-bytes must be positive and divisible by dtype element size")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be nonnegative and repeat must be positive")
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
                   "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
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
        for _ in range(args.warmup):
            dist.all_reduce(x)
        torch.npu.synchronize()
        dist.barrier()
        samples = []
        for _ in range(args.repeat):
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            dist.all_reduce(x)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1_000_000)
        payload = {
            "schema_version": 1, "rank": rank, "local_rank": local_rank,
            "world_size": world_size, "size_bytes": args.size_bytes, "dtype": args.dtype,
            "warmup": args.warmup, "repeat": args.repeat, "execution_mode": "eager",
            "expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE", "DEFAULT"),
            "profiled": args.profiled, "samples_us": samples,
            "metadata": {"hostname": socket.gethostname(),
                         "hardware": getattr(torch.npu.get_device_properties(local_rank), "name", "unknown"),
                         "torch_version": torch.__version__,
                         "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
                         "cann_version": os.environ.get("CANN_VERSION", "unknown")},
        }
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        print(f"rank {rank}: saved {len(samples)} samples to {result_path}", flush=True)
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
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--msprof", action="store_true")
    runner.add_argument("--profiled", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
