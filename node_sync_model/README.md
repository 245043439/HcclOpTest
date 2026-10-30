# 单 node 启动分布与同步时延外推

`node_sync_model/node_sync_model.py` 实现通信建模文档第 4、5 节。数值计算仅依赖 Python 标准库，
默认绘图需要 Matplotlib，可在仓库根目录执行 `pip install -r node_sync_model/requirements.txt` 安装。
使用 `--no-plots` 可跳过绘图及该依赖。
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
`t_us,F_A,F_B`；`predictions.csv` 保存预测值和误差，终端也输出相同结果。
默认还生成 `cdf.png` 与 `pdf.png`，每张图同时显示 A、B 两个分布：

- `cdf.png`：右连续经验 CDF 阶梯图，横轴为启动延迟（us），纵轴为累积概率。
- `pdf.png`：密度归一化直方图，纵轴为概率密度（1/us），各分布的总面积为 1。
  A、B 共用等宽分箱，分箱数为 `ceil(log2(M)) + 1`，M 为拟合样本对数。
  全部样本取值相同时，使用该值两侧的有限宽度区间绘图。

两张图均使用 `--node-id` 过滤后的拟合样本。经验分布是离散分布，没有连续 PDF；
这里的 PDF 图为直方图密度估计，其形状受分箱影响，不参与预测积分。

计算按相邻样本取值划分区间，累加
`区间宽度 × [1 - F_B(t)^N - (1 - F_A(t))^N]`，无需设置积分步长或假设正态分布。
每个 node 每轮的样本对等权参与估计，不要求恰好采集 50 轮。

对 `--nodes` 中的每个规模 N，自动从**未过滤的输入 CSV** 计算真实均值：
每个 step（`iteration`）选取 `node_id` 为字符串 `0` 到 `N-1` 的节点，计算
`y_true = max(B_i) - min(A_i)`，再对所有 step 的 y_true 等权求均值。
先逐 step 求跨度，再求均值；额外节点不参与该规模的真实值计算。

`predictions.csv` 和 `report.json` 中的预测结果包含：

- `expected_sync_us`：CDF 预测的平均同步时延。
- `y_true_mean_us`：所有 step 的真实跨度均值。
- `error_us`：预测均值减真实均值。
- `absolute_error_us`：上述差值的绝对值。
- `relative_error`：上述差值除以真实均值，无量纲；真实均值为零时留空。
- `iteration_count`：原始输入的 step 数量。
- `error_status`：成功为 `ok`；任意 step 缺少所需节点时为 `missing_required_nodes`，
  该规模的真实均值与误差留空（JSON 为 null），预测值仍输出，不丢弃缺失 step 后求均值。

例如输入包含节点 0、1，指定 `--nodes 2 4` 时，计算 2 node 的误差，4 node 仅输出预测。
此误差使用输入数据评估；独立数据验证仍通过 `--validation` 提供。

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

