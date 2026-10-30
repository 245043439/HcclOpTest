# 单 node 启动分布与同步时延外推

`node_sync_model/node_sync_model.py` 实现通信建模文档第 4、5 节，仅依赖 Python 标准库。
输入为第 3 节的样本 CSV，时间单位统一为微秒，每个 `(iteration, node_id)` 一行：

```csv
iteration,node_id,A_us,B_us
1,0,1.0,3.0
1,1,2.0,5.0
2,0,1.5,4.0
2,1,2.5,6.0
```

`A_us`、`B_us` 分别为该 node 本轮相对本地基准的最早、最晚启动延迟。
允许附加列、负的相对时间和任意非空轮次/node 标识；拒绝空样本、重复样本、
非有限数值及 `A_us > B_us`。每份文件应仅包含固定算子、数据量、执行模式及每 node rank 数量的数据。

以下命令从仓库根目录执行：

```bash
python3 node_sync_model/node_sync_model.py \
  --input samples_2nodes.csv --nodes 1 2 4 8 16 \
  --output-dir sync_model_results

# 可选：以独立采集的较大规模数据验证；每个文件固定一个规模，规模由 node_id 自动推断。
python3 node_sync_model/node_sync_model.py \
  --input samples_2nodes.csv --nodes 4 8 \
  --validation samples_4nodes.csv --validation samples_8nodes.csv \
  --output-dir sync_model_validated
```

输出目录必须是新路径。`cdf.csv` 保存所有不同样本取值处的右连续经验 CDF
`t_us,F_A,F_B`；`predictions.csv` 保存 `node_count,expected_sync_us`。
计算按相邻样本取值划分区间，累加
`区间宽度 × [1 - F_B(t)^N - (1 - F_A(t))^N]`，无需设置积分步长或假设正态分布。
每个 node 每轮的样本对等权参与估计，不要求恰好采集 50 轮。

可使用 `--node-id 0` 仅以指定节点的样本估计 CDF：

```bash
python3 node_sync_model/node_sync_model.py \
  --input samples_2nodes.csv --node-id 0 --nodes 1 2 4 8 16 \
  --output-dir sync_model_node0
```

默认使用全部节点。`--node-id` 按字符串精确匹配 CSV 中去除首尾空白的 `node_id`
（例如 `01` 与 `1` 不同），没有匹配样本时报错。输入 CSV 仍会完整校验。
CDF、预测、节点摘要及报告中的训练样本统计均基于过滤后的数据；
`report.json` 的 `node_id_filter` 记录过滤值，未指定时为 null。
`--nodes` 仍表示目标预测规模；`--validation` 文件使用全部节点计算观测跨度。

`node_summary.csv` 保存各 node 的样本数、A/B 均值和边界，供检查固定节点差异。
提供 `--validation` 时，`validation.csv` 比较每轮 `max(B) - min(A)` 的观测均值与预测均值，
要求同一验证文件每轮 node 集合完整且一致。`error_us` 为预测减观测，
`relative_error` 为该差值除以观测均值（无量纲，观测为零时 CSV 留空、JSON 为 null）。
`report.json` 汇总输入规模、预测、验证、模型假设与局限。

模型假设各 node 具有共同发起参考时刻、node 样本对独立同分布，且单 node 分布不随规模改变。
验证使用相对本地基准的启动跨度，与文档定义一致。节点摘要和均值误差不能证明这些假设，
轮次相关性也需要另行检查。经验分布的预测不会超过 `max(B_us) - min(A_us)`；
外推大规模前需检查尾部样本覆盖。本工具输出均值，不估计置信区间、P95 或 P99。

CPU 回归测试：`python3 -m unittest discover -s tests -p 'test_node_sync_model.py' -v`。

