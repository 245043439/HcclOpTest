# HCCL communication latency matrix

`scripts/python/run_matrix.py` runs the Cartesian product in a JSON file. Cases with the same
group size, node count, ranks per node, execution mode, expansion mode, and dtype share one distributed
launch and HCCL communicator; every configured operator and byte size runs in
order. Each completed case is checkpointed; after the launcher exits, its results
are appended to the CSV, preserving measurements from later batch failures. All sizes are bytes.
Case IDs and batch log directories use labels such as `g8_n2_p4_group8`
(8 total ranks, two cluster nodes, four processes per node, communication group
size 8). Here `g` is total ranks and `group` is communication group size.
Set the required tensor dtype with the JSON `dtype` field (the example uses
`float16`); supported names are `float16`,
`float32`, `bfloat16`, `int8`, `uint8`, `int16`, `int32`, and `int64`. HCCL
support for a dtype can vary by operation, product and CANN version, so runtime
rejections appear as failed cases in the CSV.

## Run

Edit `config/matrix_config.example.json`, then run from a machine with the cluster SSH
and Docker access configured for `scripts/shell/launch_cluster.sh`:

```bash
python3 scripts/python/run_matrix.py --config config/matrix_config.example.json --dry-run --nodes 10.0.0.2,10.0.0.3
DOCKER_SCRIPT=/shared/docker-gen.sh BENCH_SCRIPT=/workspace/run_bench.py IMAGE_NAME=my-hccl-image python3 scripts/python/run_matrix.py --config config/matrix_config.example.json --nodes 10.0.0.2,10.0.0.3
```

### 失败后恢复

重新执行相同命令即可默认恢复：运行器读取配置中的 `output_csv`，按 case ID
（含算子、大小、dtype、执行模式、展开模式和拓扑）、warmup 和 repeat 匹配历史结果。
只有最新匹配记录为 `success` 且包含完整 rank 范围时才跳过；失败、超时、
未执行及其他未成功项会重新处理。历史 CSV 保留并追加新的结果，日志使用新的
attempt 目录。`--dry-run` 也会应用恢复筛选，显示剩余工作量。

需要全量重跑时添加 `--no-resume`。更换机器、容器镜像或软件版本时，请使用新的
`output_csv` 或添加该选项，因为这些环境变化不参与恢复匹配。
按一次 Ctrl+C 后，运行器会通知 launcher 清理集群任务，从主节点容器取回
checkpoint，并将已完成 case 的结果写入 CSV，以退出码 130 退出。
等待清理完成后重新执行原命令，已完成项会跳过，正在执行但未完成的 case 会重跑。
启动时还会从同一 `--log-dir` 自动导入尚未处理的 checkpoint；`--dry-run`
只预览恢复结果，不写文件。每次启动保存参数清单，避免导入其他输出文件或
warmup/repeat 不匹配的结果。此功能适用于新版生成的带 `attempt.json` 的日志。
恢复时请保留原 CSV 和日志目录。连续按 Ctrl+C 或清理超时会强制结束清理；
`kill -9` 不执行清理，尚未取回的容器 checkpoint 不保证能自动恢复。

`--nodes` lists the only hosts that run benchmark workers. The command machine
coordinates the run over SSH and does not need to be a worker. Each listed host
is addressed as `SSH_USER@IP` (default user `root`); the first listed host is the
distributed master. `NPROC_PER_NODE` is the per-host capacity (default 8).
Uniform placements satisfying `node_count * ranks_per_node == group_size` are
enumerated using only the listed hosts, in input order. For group 8 with two
listed hosts and capacity 8, launches use `(2,4)` and `(1,8)`. Omitting
`--nodes` keeps the legacy single-host mode on `MASTER_ADDR` (default
`141.61.94.99`). The standalone launcher still treats `NPROC_PER_NODE` as the
actual ranks per node.
Set `DOCKER_SCRIPT` to the `scripts/shell/docker-gen.sh` path visible on each worker host, and
`BENCH_SCRIPT` to the `scripts/python/run_bench.py` path visible inside the container. Both are
required and have no defaults. The benchmark script must also be visible at the
same path on every worker host if the container uses a host mount.

The supported operations are allreduce, alltoall, reducescatter, allgather,
broadcast, ring batch_send_recv, and unidirectional send_recv from the first to
last rank of each group. Group sizes must be powers of two and equal the world size of each launch.

所有算子统一使用 CSV 的 `rank_list` 记录通信域中的 global_rank，保存为 JSON 数组。
编号规则为 `节点在 --nodes 中的序号 × rank_stride + 本地卡号`，两者都从 0 开始。
JSON 配置的 `rank_stride` 默认是 8，应设置为每节点实际卡数，不能小于每节点启动的进程数。
例如 `n2_p1` 为 `[0, 8]`，`n1_p2` 为 `[0, 1]`，`n2_p2` 为 `[0, 1, 8, 9]`。
`--dry-run` 也会显示该列表；实际通信仍使用 torchrun 的连续 rank 编号。
`send_recv` 从列表首项发送到末项，`batch_send_recv` 按列表顺序组成环。
旧的 rank_start/rank_end、source_rank/target_rank 列不再输出。
旧 CSV 仍可恢复；追加结果时会原子转换为新表头，依据历史 case ID 中的拓扑和
rank 范围生成 rank_list。历史 rank_stride 缺失时按 8 处理，无法识别的拓扑记录为 `[]`。
节点顺序改变时，global_rank 对应的主机也会改变。

Groups without a placement within the available capacity are skipped. Invalid combinations are recorded as skipped. On Atlas 350, AIV and
CCU_MS are recorded as skipped according to the current HCCL documentation;
actual expansion selection for other settings is reported as `unknown` because
HCCL does not expose a portable runtime query for it.

检查指定节点上的 NPU 使用和进程：

```bash
python3 scripts/python/query_npu_usage.py --nodes 141.61.94.111,141.61.94.112
```

脚本通过 SSH 在每个节点执行 `npu-smi info`，输出设备利用率、显存和占用进程表；
SSH 用户默认为 `SSH_USER` 环境变量或 `root`，可用 `--ssh-user` 覆盖。需要控制机
可以免密 SSH 登录这些节点。
默认只显示每个节点是否有 NPU 进程占用；需要查看完整 `npu-smi info` 输出和进程明细时，添加 `--verbose`。

`eager` uses direct PyTorch distributed calls. `ascend_ir` requires TorchAir's
HCCL patch and full graph compilation. `aclgraph` requires the NPU graph APIs
exposed by the installed torch_npu. Unsupported or failed graph runs are
recorded; they are never silently measured as eager. `scripts/python/run_matrix.py` records
one CSV row per case and topology, with latency statistics in milliseconds
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
tar -xf hccl_matrix_logs/<batch_id>/attempt_1/node_0.diagnostics.tar -C diagnostics
cat diagnostics/rank_0.traceback.log
```

Shell 命令失败会记录节点、阶段、退出码、命令和调用位置；展开未定义变量
产生的错误保留原始错误信息和退出记录。用例目录的 `worker.sh` 保存实际
执行的 worker 脚本，可按错误行号定位。诊断文件收集失败时查看
`collect_<节点编号>.log`。运行器自身的用例异常保存在 `matrix.traceback.log`。
