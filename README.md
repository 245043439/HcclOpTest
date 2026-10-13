# HCCL communication latency matrix

`run_matrix.py` runs the Cartesian product in a JSON file, launching a fresh
distributed job for each supported combination. The CSV is appended after every
case, so completed measurements survive later failures. All sizes are bytes and
the benchmark dtype is float32.

## Run

Edit `matrix_config.example.json`, then run from a machine with the cluster SSH
and Docker access configured for `launch_cluster.sh`:

```bash
python3 run_matrix.py --config matrix_config.example.json --dry-run --nodes 10.0.0.2,10.0.0.3
IMAGE_NAME=my-hccl-image python3 run_matrix.py --config matrix_config.example.json --nodes 10.0.0.2,10.0.0.3
```

`--nodes` lists the nodes after the master. The master address comes from
`MASTER_ADDR` (default `141.61.94.99`). `NPROC_PER_NODE` defaults to 8.
`run_matrix.py` passes this checkout's `run_bench.py` path to the launcher by
default. The checkout must be visible at the same path on all nodes and inside
the container; set `BENCH_SCRIPT` to the container-visible path if it differs.

The supported operations are allreduce, alltoall, reducescatter, allgather,
broadcast, ring batch_send_recv, and unidirectional send_recv from the first to
last rank of each group. Group sizes must be powers of two and divide world
size. Invalid combinations are recorded as skipped. On Atlas 350, AIV and
CCU_MS are recorded as skipped according to the current HCCL documentation;
actual expansion selection for other settings is reported as `unknown` because
HCCL does not expose a portable runtime query for it.

`eager` uses direct PyTorch distributed calls. `ascend_ir` requires TorchAir's
HCCL patch and full graph compilation. `aclgraph` requires the NPU graph APIs
exposed by the installed torch_npu. Unsupported or failed graph runs are
recorded; they are never silently measured as eager. `run_matrix.py` records
one CSV row per operation and subgroup, with latency statistics in milliseconds
and a status/reason for every skipped or failed case.

The benchmark does not check numerical results. Run an initial small matrix on
the target A5/CANN stack before expanding sizes and group counts; this local
environment does not include NPU dependencies, so real measurements must be
collected on the target cluster.
