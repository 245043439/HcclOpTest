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
The optional `size_scale_factors` field expands every `sizes_bytes` entry by
each configured power-of-two factor: divisible sizes are also divided by the
factor, and every size is multiplied by it. The original size is retained;
non-integer byte divisions are skipped, and duplicate sizes are emitted once
in first-seen order. Sizes above 8 GiB are omitted. The 8 GiB limit is also
checked per case against `size_bytes`, which is the per-rank communication
payload convention used by the benchmark and result database. For `allgather`,
`size_bytes` is the total output size; its local input is `size_bytes / group_size`.
Leave the field out or set it to `[]` to disable expansion.

## Run

Edit `config/matrix_config.example.json`, then run from a machine with the cluster SSH
and Docker access configured for `scripts/shell/launch_cluster.sh`:

```bash
python3 scripts/python/run_matrix.py --config config/matrix_config.example.json --dry-run --nodes 10.0.0.2,10.0.0.3
DOCKER_SCRIPT=/shared/docker-gen.sh BENCH_SCRIPT=/workspace/run_bench.py IMAGE_NAME=my-hccl-image python3 scripts/python/run_matrix.py --config config/matrix_config.example.json --nodes 10.0.0.2,10.0.0.3
```

### 失败后恢复

预检失败（退出码 69）只将当前批次的未完成用例标记为 `not_run`，继续执行后续批次，
包括同一 group 的其他拓扑和后续 group；每个批次仍独立预检。
清理失败（退出码 70）仍停止整个矩阵，并将剩余用例标记为 `not_run`，避免残留任务影响后续测试。

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

### `sizes_bytes` 与算子输入输出

配置中的 `sizes_bytes`（运行时对应 `size_bytes`，即常说的 `data_size`）统一按
**每个 rank 的张量大小**理解，单位为字节；元素数为 `size_bytes / dtype` 的单元素字节数。
以下 input/output 均指每个 rank 的本地张量，不是通信组的总量。`p` 表示通信组大小。

| 算子 | 每个 rank 的输入 | 每个 rank 的输出 | 通信含义 |
| --- | --- | --- | --- |
| `allreduce` | `size_bytes` | `size_bytes` | 每个 rank 对同尺寸输入做规约，结果仍为同尺寸。 |
| `alltoall` | `size_bytes` | `size_bytes` | 输入均分为 `p` 份，每份 `size_bytes / p`，分别发往组内各 rank；输出按来源 rank 收齐 `p` 份。 |
| `reducescatter` | `size_bytes` | `size_bytes / p` | 输入按 rank 分块规约，每个 rank 得到规约结果的一块。 |
| `allgather` | `size_bytes / p` | `size_bytes` | `size_bytes` 表示每个 rank 的总输出大小；每个 rank 输入一块，收集所有 rank 的块。 |
| `broadcast` | `size_bytes` | `size_bytes` | 组首 rank 广播同尺寸张量，其他 rank 接收并覆盖本地张量。 |
| `batch_send_recv` | `size_bytes` | `size_bytes` | 每个 rank 向环中的下一个 rank 发送一份，并从上一个 rank 接收一份。 |
| `send_recv` | `size_bytes` | `size_bytes` | 组首 rank 向组末 rank 发送一份；只有这两个端点实际发送/接收，中间 rank 只参加同步。 |

例如 `dtype=float16`、`size_bytes=1024`、`p=4` 时，`alltoall` 每个 rank
输入和输出各为 1024 字节，每对 rank 之间传输 256 字节；`allgather` 每个
rank 输入 256 字节、输出 1024 字节。`alltoall`、`reducescatter` 和
`allgather` 的切分要求可被 `p` 整除，不满足时该用例会报错。输入张量以全 1
初始化，因此此基准测量通信性能，不校验 rank 来源或计算结果。

所有算子统一使用 CSV 的 `rank_list` 记录参与该算子数据通信的 global_rank，保存为 JSON 数组。
编号规则为 `节点在 --nodes 中的序号 × rank_stride + 本地卡号`，两者都从 0 开始。
JSON 配置的 `rank_stride` 默认是 8，应设置为每节点实际卡数，不能小于每节点启动的进程数。
例如 `n2_p1` 为 `[0, 8]`，`n1_p2` 为 `[0, 1]`，`n2_p2` 的完整通信组为 `[0, 1, 8, 9]`。
`--dry-run` 也会显示该列表；实际通信仍使用 torchrun 的连续 rank 编号。
`send_recv` 的 `rank_list` 只列发送端和接收端（`n2_p2` 为 `[0, 9]`）；其他 rank 仍参与分布式组和同步。`batch_send_recv` 按完整列表组成环。
旧的 rank_start/rank_end、source_rank/target_rank 列不再输出。
旧 CSV 仍可恢复；追加结果时会原子转换为新表头，依据历史 case ID 中的拓扑和
rank 范围生成 rank_list。历史 rank_stride 缺失时按 8 处理，无法识别的拓扑记录为 `[]`。
节点顺序改变时，global_rank 对应的主机也会改变。

Groups without a placement within the available capacity are skipped. Invalid combinations are recorded as skipped.
Expansion modes, including AIV and CCU_MS, are passed to HCCL as requested; actual expansion selection is reported as `unknown` because HCCL does not expose a portable runtime query for it.

检查指定节点上的 NPU 使用和进程：

```bash
python3 scripts/python/query_npu_usage.py --nodes 141.61.94.111,141.61.94.112
```

脚本通过 SSH 在每个节点执行 `npu-smi info`，输出设备利用率、显存和占用进程表；
SSH 用户默认为 `SSH_USER` 环境变量或 `root`，可用 `--ssh-user` 覆盖。需要控制机
可以免密 SSH 登录这些节点。
默认按“有进程”“无进程”和“无法判断”分组显示节点；需要查看完整 `npu-smi info` 输出和进程明细时，添加 `--verbose`。SSH 登录横幅会自动过滤。

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

## AllReduce 最小用例

`scripts/python/compare_allreduce.py` 是独立的 AllReduce 最小用例，默认使用 eager 模式。
在目标服务器准备好 PyTorch、torch_npu、CANN 和 `msprof` 后，从仓库根目录运行：

```bash
# 基线：单机双卡，默认 float32、1 MiB、10 次预热、50 次测量。
torchrun --nnodes=1 --nproc-per-node=2 --master_addr=127.0.0.1 \
  scripts/python/compare_allreduce.py run --output ./allreduce_baseline

# 每个 rank 由普通 msprof 包装，保留相同的端到端计时。
torchrun --nnodes=1 --nproc-per-node=2 --master_addr=127.0.0.1 \
  scripts/python/compare_allreduce.py run --msprof --output ./allreduce_profiled

# ACL Graph 模式；可添加 --msprof 采集，也可改为 --execution-mode ascend_ir。
torchrun --nnodes=1 --nproc-per-node=2 --master_addr=127.0.0.1 \
  scripts/python/compare_allreduce.py run --execution-mode aclgraph --output ./allreduce_aclgraph

# 采集 Python 调用栈（该采集方式使用 torch_npu.profiler，不要同时添加 --msprof）。
torchrun --nnodes=1 --nproc-per-node=2 --master_addr=127.0.0.1 \
  scripts/python/compare_allreduce.py run --python-stack --output ./allreduce_python_stack
```

基线和 profiling 是两次独立采集；本脚本只负责运行并保存各 rank 的原始结果，不提供两次结果的离线对比。
支持 `--size-bytes`、`--dtype {float16,float32,bfloat16}`、`--warmup` 和 `--repeat`。
`--execution-mode {eager,ascend_ir,aclgraph}` 与矩阵基准含义一致：`ascend_ir` 使用 TorchAir
HCCL patch 和全图编译，`aclgraph` 使用 NPU 图捕获与重放。编译、捕获及其初始化调用在预热和
计时之前完成；msprof 产物可能包含这些额外调用。图模式不可用或执行失败时直接报错。
所选模式会透传给 msprof 子进程，并写入各 rank JSON 的 `execution_mode` 字段。
`--python-stack` 使用 `torch_npu.profiler` 同时采集 CPU/NPU 活动并启用 `with_stack=True`，
结果写入 `python_profiler/rank_<rank>/`。该选项不能与 `--msprof` 同时使用；调用栈采集会带来额外开销，
对应的耗时用于检查调用路径，不代表未开启采集时的基准性能。
可分别测试 1024、1048576、67108864 字节；每个大小的基线和 profiling 参数须一致，
并使用独立输出目录。所有输出目录应为新路径，重跑时不要混入历史文件。
脚本记录版本元数据；若需记录 CANN 版本，请在两次运行前设置准确的 `CANN_VERSION`。
`HCCL_OP_EXPANSION_MODE` 原样继承并记录，不自动修改。

每轮顺序为 barrier → NPU 同步 → `perf_counter()` → AllReduce → NPU 同步 → 停止计时。
输入为零，避免重复原地 SUM 溢出。计时范围与 `run_bench.py` 相同，但不执行它的逐轮
耗时归约 AllReduce，而是将各 rank 原始耗时保存到 `rank_<rank>.json` 后离线取最大值。
因此计时区间外的执行节奏与原脚本略有不同。此用例不做数值正确性验证。
每个 rank 测量结束后会打印样本平均值、Q1/Q2/Q3/Q4 和总体方差；分位数按最近秩计算，
Q4 表示最大值，方差单位为 us²。

`--msprof` 使用 `--task-time=on --hccl=on --ai-core=off --type=text`，
产物分别保存在 `profiler/rank_<rank>/`。使用完整 HCCL 通信算子时间，
不使用 `msprof op` 的单 kernel 重放时间。目标 CANN/msprof 版本需支持这些选项。
若未自动生成 `op_summary*.csv`，对每个 rank 的 `PROF_*` 目录执行
`msprof --export=on --output=<PROF目录> --type=text`，确保目录内只保留一套导出结果。

每个 rank 的端到端样本和版本、硬件元数据保存在 `rank_<rank>.json`；msprof 原始产物位于
`profiler/rank_<rank>/`。需要查看通信耗时或对照不同采集时，请直接使用这些原始文件和 msprof
导出结果自行分析。脚本不负责解析 `op_summary*.csv`，也不比较基线与 profiling 两次运行。

CPU 回归测试：`python3 -m unittest discover -s tests -p 'test_compare_allreduce.py' -v`。
双卡实测在目标服务器完成；CPU 测试不代表已验证目标环境的 profiler 兼容性。

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
