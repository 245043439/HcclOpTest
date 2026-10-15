#!/usr/bin/env python3
"""Standalone BF16 DeepEP benchmark; launch identical arguments with torchrun."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

from run_matrix import FIELDS


STAGE = "arguments"
OPERATORS = ("dispatch", "combine", "dispatch_combine")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name, default in (("num-tokens", 256), ("hidden", 7168), ("num-topk", 8),
                          ("num-experts", 256), ("warmup", 50), ("repeat", 50),
                          ("seed", 1), ("rank-stride", 8)):
        result.add_argument(f"--{name}", type=int, default=default)
    result.add_argument("--low-latency-strategy", choices=("default", "ops"), default="default")
    result.add_argument("--output-csv", type=Path, default=Path("dispatch_combine_results.csv"))
    result.add_argument("--check", action="store_true")
    return result


def validate(args, env):
    for key in ("num_tokens", "hidden", "num_topk", "num_experts", "repeat", "rank_stride"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.warmup < 0 or args.seed < 0:
        raise ValueError("warmup and seed must be nonnegative")
    if args.num_topk > args.num_experts:
        raise ValueError("num_topk must not exceed num_experts")
    try:
        rank, world, local, local_world = [int(env[k]) for k in
                                          ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")]
    except (KeyError, ValueError) as exc:
        raise ValueError("launch with torchrun (RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE required)") from exc
    if (world < 2 or local_world <= 0 or world % local_world or
            not 0 <= rank < world or not 0 <= local < local_world or
            rank % local_world != local):
        raise ValueError("require at least two ranks and uniform contiguous torchrun placement")
    if args.rank_stride < local_world:
        raise ValueError("rank_stride must be >= LOCAL_WORLD_SIZE")
    if args.num_experts % world:
        raise ValueError("num_experts must be divisible by WORLD_SIZE")
    for name in ("DEEP_NORMAL_MODE_USE_INT8_QUANT", "MOE_SHARED_EXPERT_RANK_NUM",
                 "MOE_ENABLE_TOPK_NEG_ONE"):
        if env.get(name, "0") != "0":
            raise ValueError(f"{name} conflicts with the BF16 standard-EP benchmark")
    mode = env.get("DEEP_USE_MODE", "").lower()
    if mode and mode != args.low_latency_strategy:
        raise ValueError("DEEP_USE_MODE conflicts with --low-latency-strategy")
    return rank, world, local, local_world


def statistics_ms(samples):
    return {"mean_ms": statistics.fmean(samples), "min_ms": min(samples),
            "max_ms": max(samples), "p50_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]}


def measure(operator, dispatch, combine, synchronize, barrier, reduce_max, warmup, repeat):
    """Always pair calls; preparation/cleanup and reductions are outside timing."""
    global STAGE
    samples = []
    for iteration in range(warmup + repeat):
        STAGE = f"{operator} {'warmup' if iteration < warmup else 'measurement'} {iteration + 1}"
        received = dispatch() if operator == "combine" else None
        barrier()
        synchronize()
        start = time.perf_counter()
        if operator != "combine":
            received = dispatch()
        if operator != "dispatch":
            combine(received)
        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if operator == "dispatch":
            combine(received)
            synchronize()
        if iteration >= warmup:
            samples.append(reduce_max(elapsed_ms))
    return statistics_ms(samples)


def check_csv(path):
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream), None) != FIELDS:
                raise ValueError(f"{path}: CSV header does not match current matrix schema")


def append_row(path, row):
    check_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def result_row(args, operator, world, local_world, recv_bytes, metadata, metrics):
    nodes = world // local_world
    size = args.num_tokens * args.hidden * 2
    label = (f"{operator}_g{world}_n{nodes}_p{local_world}_group{world}"
             f"_t{args.num_tokens}_h{args.hidden}_k{args.num_topk}_e{args.num_experts}"
             f"_bf16_eager_{args.low_latency_strategy}_seed{args.seed}"
             f"_check{int(args.check)}_stride{args.rank_stride}")
    row = dict.fromkeys(FIELDS, "")
    row.update(case_id=label, operator=operator, size_bytes=size, dtype="bfloat16",
               group_size=world,
               rank_list=json.dumps([n * args.rank_stride + p for n in range(nodes)
                                     for p in range(local_world)]),
               comm_layer_list=f"(1,{nodes},{local_world})", execution_mode="eager",
               requested_expansion_mode=os.environ.get("HCCL_OP_EXPANSION_MODE", "DEFAULT"),
               actual_expansion_mode="unknown", warmup=args.warmup, repeat=args.repeat,
               input_bytes=recv_bytes if operator == "combine" else size,
               output_bytes=recv_bytes if operator == "dispatch" else size,
               status="success")
    row.update(metadata)
    row.update(metrics)
    return row


def run(args):
    global STAGE
    rank, world, local, local_world = validate(args, os.environ)
    STAGE = "import torch/torch_npu/deep_ep"
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep import Buffer

    STAGE = "initialize HCCL"
    torch.npu.set_device(local)
    dist.init_process_group(backend="hccl")
    buffer = None
    try:
        # Check uniform launch settings before any shape-dependent collectives.
        STAGE = "validate distributed configuration"
        config = {**vars(args), "output_csv": str(args.output_csv), "local_world": local_world}
        configs = [None] * world
        dist.all_gather_object(configs, config)
        if any(item != config for item in configs):
            raise ValueError("all ranks must use identical arguments and LOCAL_WORLD_SIZE")

        def root_action(action):
            error = [None]
            if rank == 0:
                try:
                    action()
                except Exception as exc:
                    error[0] = str(exc)
            dist.broadcast_object_list(error, src=0)
            if error[0]:
                raise RuntimeError(error[0])

        root_action(lambda: check_csv(args.output_csv))
        device = f"npu:{local}"

        def maximum(value, dtype=None):
            tensor = torch.tensor([value], device=device, dtype=dtype or torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            return tensor.item()

        STAGE = "allocate inputs and DeepEP buffer"
        torch.manual_seed(args.seed + rank)
        x = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device=device)
        scores = torch.randn((args.num_tokens, args.num_experts), dtype=torch.float32,
                             device=device).abs() + 1
        indices = scores.topk(args.num_topk, dim=-1, sorted=True).indices
        weights = torch.randn((args.num_tokens, args.num_topk), dtype=torch.float32,
                              device=device).abs()
        del scores
        recv_stats = torch.zeros(args.num_experts // world, dtype=torch.int32, device=device)
        buffer = Buffer(dist.group.WORLD,
                        num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(
                            args.num_tokens, args.hidden, world, args.num_experts),
                        low_latency_mode=True, num_qps_per_rank=args.num_experts // world,
                        low_latency_strategy=args.low_latency_strategy)

        def dispatch():
            received, counts, handle, event, hook = buffer.low_latency_dispatch(
                x, indices, args.num_tokens, args.num_experts,
                cumulative_local_expert_recv_stats=recv_stats, async_finish=False,
                return_recv_hook=False, topk_weights=weights, use_fp8=False)
            return received, handle

        def combine(received):
            payload, handle = received
            output, event, hook = buffer.low_latency_combine(
                payload, indices, weights, handle, async_finish=False,
                zero_copy=False, return_recv_hook=False, use_mxfp8=False)
            return output

        STAGE = "initial dispatch/combine and optional correctness check"
        received = dispatch()
        if not isinstance(received[0], torch.Tensor) or received[0].dtype != torch.bfloat16:
            raise RuntimeError("dispatch must return a BF16 tensor; check DeepEP quantization settings")
        output = combine(received)
        torch.npu.synchronize()
        recv_bytes = maximum(received[0].numel() * received[0].element_size(), torch.int64)
        if args.check:
            golden = x.float() * weights.sum(dim=1, keepdim=True)
            valid = (torch.isfinite(output).all().item() and
                     torch.allclose(output.float(), golden, rtol=0.02, atol=0.02))
            if maximum(int(not valid), torch.int32):
                raise RuntimeError("BF16 combine correctness check failed on one or more ranks")
        del received, output
        metadata = {
            "hardware": getattr(torch.npu.get_device_properties(local), "name", "unknown"),
            "cann_version": os.environ.get("CANN_VERSION", "unknown"),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            "torchair_version": "unknown",
        }
        for operator in OPERATORS:
            metrics = measure(operator, dispatch, combine, torch.npu.synchronize,
                              dist.barrier, maximum, args.warmup, args.repeat)
            STAGE = f"write {operator} CSV"
            row = result_row(args, operator, world, local_world, recv_bytes, metadata, metrics)
            root_action(lambda: append_row(args.output_csv, row))
            if rank == 0:
                print(f"{operator}: " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
                      flush=True)
        torch.npu.synchronize()
    finally:
        # Keep the process group alive until the communication buffer is released.
        buffer = None
        dist.destroy_process_group()


def main():
    args = parser().parse_args()
    try:
        run(args)
    except Exception:
        print(f"rank={os.environ.get('RANK', 'unknown')} stage={STAGE}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
