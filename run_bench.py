
import os
import time
import argparse

import torch
import torch_npu
import torch.distributed as dist


def create_group(group_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size % group_size == 0

    group = None
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        new_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            group = new_group

    return group


def benchmark(fn, group, warmup=10, repeat=50):
    # Warmup
    for _ in range(warmup):
        fn()

    torch.npu.synchronize()
    dist.barrier(group=group)

    latencies = []

    for _ in range(repeat):
        # 每轮开始前同步通信域
        dist.barrier(group=group)

        torch.npu.synchronize()
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        latency = (time.perf_counter() - start) * 1000

        # 取通信域内所有 Rank 的最大时延
        t = torch.tensor([latency], dtype=torch.float32,
                         device="npu")
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
        latencies.append(t.item())

    return sum(latencies) / len(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=float, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(backend="hccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    p = args.group_size

    assert 2 <= p <= world_size
    assert world_size % p == 0
    assert args.size_mb > 0
    assert args.repeat > 0
    assert args.warmup >= 0

    group = create_group(p)
    group_rank = dist.get_rank(group=group)

    dtype = torch.float32
    elem_size = torch.empty((), dtype=dtype).element_size()

    numel = int(args.size_mb * 1024 * 1024 / elem_size)
    numel = (numel // p) * p
    assert numel > 0

    x = torch.ones(numel, device=device, dtype=dtype)

    results = {}

    # AllReduce
    def allreduce():
        dist.all_reduce(x, group=group)

    results["allreduce"] = benchmark(
        allreduce, group, args.warmup, args.repeat
    )

    # AllToAll
    a2a_out = torch.empty_like(x)

    def alltoall():
        dist.all_to_all_single(a2a_out, x, group=group)

    results["alltoall"] = benchmark(
        alltoall, group, args.warmup, args.repeat
    )

    # ReduceScatter
    rs_out = torch.empty(
        numel // p, device=device, dtype=dtype
    )

    def reducescatter():
        dist.reduce_scatter_tensor(rs_out, x, group=group)

    results["reducescatter"] = benchmark(
        reducescatter, group, args.warmup, args.repeat
    )

    # AllGather
    ag_in = torch.ones(
        numel // p, device=device, dtype=dtype
    )
    ag_out = torch.empty_like(x)

    def allgather():
        dist.all_gather_into_tensor(ag_out, ag_in, group=group)

    results["allgather"] = benchmark(
        allgather, group, args.warmup, args.repeat
    )

    # BatchSendRecv (Ring)
    group_start = (rank // p) * p
    next_rank = group_start + (group_rank + 1) % p
    prev_rank = group_start + (group_rank - 1 + p) % p

    recv_buf = torch.empty_like(x)

    def batch_send_recv():
        ops = [
            dist.P2POp(dist.isend, x, next_rank, group),
            dist.P2POp(dist.irecv, recv_buf, prev_rank, group),
        ]
        requests = dist.batch_isend_irecv(ops)
        for req in requests:
            req.wait()

    results["batchSendRecv"] = benchmark(
        batch_send_recv, group, args.warmup, args.repeat
    )

    if group_rank == 0:
        print(
            f"\nRanks: [{group_start}, {group_start + p - 1}]"
            f" | Group size: {p}"
            f" | Input size: {numel * elem_size / 1024**2:.3f} MB"
        )
        for name, latency in results.items():
            print(f"{name:20s}: {latency:.3f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
