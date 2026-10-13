# HCCL communication latency matrix

`run_matrix.py` runs the Cartesian product in a JSON file. Cases with the same
group size, execution mode, expansion mode, and dtype share one distributed
launch and HCCL communicator; every configured operator and byte size runs in
order. The CSV is appended after every completed case, so measurements survive
later failures. All sizes are bytes.
Case IDs and batch log directories include topology labels such as `g8_n2_p4`
(group size 8, two cluster nodes, four processes per node).
Set the required tensor dtype with the JSON `dtype` field (the example uses
`float16`); supported names are `float16`,
`float32`, `bfloat16`, `int8`, `uint8`, `int16`, `int32`, and `int64`. HCCL
support for a dtype can vary by operation, product and CANN version, so runtime
rejections appear as failed cases in the CSV.

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
one CSV row per case and subgroup, with latency statistics in milliseconds
and a status/reason for every skipped or failed case.

The benchmark does not check numerical results. Run an initial small matrix on
the target A5/CANN stack before expanding sizes and group counts; this local
environment does not include NPU dependencies, so real measurements must be
collected on the target cluster.

## 异常与卡住诊断

运行器按批次和恢复次数创建日志目录，实时写入 `launcher.log`，每 30 秒提示等待时间。
节点的完整标准输出和错误输出保存在 `node_0.log`、`node_1.log` 等文件。
Python 异常会记录完整调用栈、异常链、rank、主机、进程、参数和当前阶段，
随后退出当前分布式任务；异常后不再执行 barrier。每个用例完成后 rank 0
原子保存检查点；批次中断时先写入已完成结果，再将检查点标记的失败用例
记为失败并重建通信域继续剩余用例。无法读取可信检查点时，本批剩余用例
记录为失败，不会无限重试。

每个 rank 还会单独保存 `rank_<全局rank>.traceback.log`，默认每 60 秒记录
一次所有 Python 线程的当前栈，帮助定位没有抛异常的阻塞。这种定时栈快照
本身不代表执行失败。任务结束或清理后，启动器从各容器收集这些文件到
`node_<节点编号>.diagnostics.tar`，无需共享输出目录。例如：

```bash
mkdir -p diagnostics
tar -xf hccl_matrix_logs/<case_id>/node_0.diagnostics.tar -C diagnostics
cat diagnostics/rank_0.traceback.log
```

Shell 命令失败会记录节点、阶段、退出码、命令和调用位置；展开未定义变量
产生的错误保留原始错误信息和退出记录。用例目录的 `worker.sh` 保存实际
执行的 worker 脚本，可按错误行号定位。诊断文件收集失败时查看
`collect_<节点编号>.log`。运行器自身的用例异常保存在 `matrix.traceback.log`。
