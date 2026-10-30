# 基于单 node 启动分布的同步时延外推流程

## 1. 目标与流程

**建模背景**：
当前通信算子时延构成中，由于各rank因kernel启动时间不一致导致的同步时延占比很大，实测16卡alltoall的同步时延占比在50%以上。定义同步时延为：

$$T_{sync}=所有rank \space kernel启动时间的最大值-所有rank\space kernel启动时间的最小值$$
从已有通信 profiling 中提取单个 node 的最早、最晚 kernel 启动延迟，估计其分布，再预测不同 node 数量下的启动跨度期望。

$$
\boxed{
\text{采集各 node 的 }(A_i,B_i)
\;\longrightarrow\;
\text{估计 }F_A,F_B
\;\longrightarrow\;
\text{代入目标 node 数量 }N
}
$$

本文研究的场景是：**第 $k$ 轮，各 node 在同一参考时刻 $C_k$ 下发目标通信算子，随后由于各 node、各 rank 的启动延迟不同而产生同步时延。** 同时下发是模型的既定假设；kernel 实际开始执行的时刻仍然可以不同。本文直接将这一场景中的启动跨度记为 $T_{\mathrm{sync}}$。

采集时按每个 node 的本地参考时刻提取启动延迟样本，用于估计上述场景中的单 node 分布。参考时刻的估计方法见附录 A。

## 2. 符号与假设

| 符号 | 含义 |
|---|---|
| $k$ | 通信算子的重复轮次，当前采集 $K=50$ 轮 |
| $i$ | node 编号 |
| $r$ | node 内的 rank 编号 |
| $n_0$ | 当前用于采集的 node 数量 |
| $m$ | 每个 node 参与通信的 rank 数量 |
| $N$ | 需要预测的目标 node 数量 |
| $x_{k,i,r}$ | 相对于本 node 本轮基准的目标 kernel 启动延迟 |
| $A_{k,i},B_{k,i}$ | 本 node 本轮最早、最晚的目标 kernel 启动延迟 |

模型采用以下假设：

0. 目标场景中，各 node 的发起参考时刻相同，即 $C_{k,i}=C_k$。不同 node 的发起偏移不属于本模型要估计的量。

1. 每个 node 的 rank 数量、设备配置和运行条件一致。
2. 对同一轮，不同 node 的随机向量独立同分布：

   $$
   (A_{k,1},B_{k,1}),\ldots,(A_{k,n_0},B_{k,n_0})
   \overset{\mathrm{iid}}{\sim}H,
   \qquad A_{k,i}\le B_{k,i}.
   $$

3. 同一个 node 内的 $A$ 与 $B$ 允许相关，不要求二者独立或同分布。
4. 重复轮次的分布保持稳定；将各轮当作独立样本或估计置信区间时，需要另行检查轮次间的相关性。
5. 外推时，单 node 的分布不随 node 数量变化。仅有“每个规模内 iid”不足以保证这一点。

固定算子、数据量、执行模式及每 node 的 rank 数量后进行估计。若这些条件会改变启动分布，应分组建模。

## 3. 采集每个 node 的样本

对每轮通信，定位各 rank 上对应的同一次目标 kernel，并得到相对启动延迟 $x_{k,i,r}$。相对时间的具体计算见附录 A。

按 node 分组，提取：

$$
\boxed{
A_{k,i}=\min_{r\in\text{node }i}x_{k,i,r},
\qquad
B_{k,i}=\max_{r\in\text{node }i}x_{k,i,r}
}
$$

每轮每个 node 生成一对样本，而非把每个 rank 当作一个独立 node 样本。

建议保留以下样本表，所有时间统一为微秒：

| iteration | node_id | A_us | B_us |
|---|---|---|---|
| 1 | 0 | 第 1 轮 node 0 的最早延迟 | 第 1 轮 node 0 的最晚延迟 |
| 1 | 1 | 第 1 轮 node 1 的最早延迟 | 第 1 轮 node 1 的最晚延迟 |
| … | … | … | … |
| 50 | $n_0-1$ | 第 50 轮对应值 | 第 50 轮对应值 |

如果完整采集 50 轮、$n_0$ 个 node，就有 $M=50n_0$ 对样本。合并前检查各 node 的分布是否接近，避免把固定的快慢节点差异误当作同分布随机抖动。

## 4. 估计单 node 的两个分布

定义边缘累积分布函数：

$$
F_A(t)=P(A\le t),\qquad F_B(t)=P(B\le t).
$$

它们分别表示：单个 node 的最早、最晚启动延迟不超过 $t$ 的概率。

将全部样本展平为 $(a_j,b_j)$，$j=1,\ldots,M$，可以直接使用经验分布：

$$
\boxed{
\widehat F_A(t)=\frac1M\sum_{j=1}^{M}\mathbf1_{\{a_j\le t\}},
\qquad
\widehat F_B(t)=\frac1M\sum_{j=1}^{M}\mathbf1_{\{b_j\le t\}}
}
$$

不必额外假设正态分布、固定的 node 内跨度，或对 node 数量另拟合 $\log N$ 等函数。

经验分布是直接可用的起点。但它不产生超出已观测范围的尾部样本；外推到更大 $N$ 时，应检查样本量及尾部覆盖是否足够。参数分布或尾部拟合属于可选的分布估计方法，不是外推公式的前提。

## 5. 代入目标 node 数量

对目标规模，假设抽取 $N$ 对 iid 的 node 样本：

$$
(A_1,B_1),\ldots,(A_N,B_N)\overset{\mathrm{iid}}{\sim}H.
$$

在共同发起时刻的假设下，定义相对该时刻的最早、最晚 kernel 启动时间：

$$
U_N=\min_i A_i,\qquad V_N=\max_i B_i,
\qquad T_{\mathrm{sync}}(N)=V_N-U_N.
$$

由于不同 node 独立：

$$
P(V_N\le t)=F_B(t)^N,
\qquad
P(U_N>t)=[1-F_A(t)]^N.
$$

在时刻 $t$ 处于启动跨度内的概率为：

$$
P(U_N\le t<V_N)
=1-F_B(t)^N-[1-F_A(t)]^N.
$$

沿时间轴积分，得到期望，假设其有限：

$$
\boxed{
\mathbb E[T_{\mathrm{sync}}(N)]
=\int_{-\infty}^{+\infty}
\left[1-F_B(t)^N-[1-F_A(t)]^N\right]dt
}
$$

以上概率公式及期望积分的详细推导见附录 B。

计算平均值只需要两个边缘分布，不要求 $U_N,V_N$ 独立。若需要 P95、P99 或方差，则还需要保留 $(A,B)$ 的联合分布信息，例如对样本对进行成对采样。

### 5.1 使用经验分布直接计算

将 $\widehat F_A,\widehat F_B$ 代入上式即可。

经验 CDF 是阶梯函数，可避免设置任意积分步长：将所有 $a_j,b_j$ 的不同取值从小到大排序为

$$
z_1<z_2<\cdots<z_L.
$$

在每个区间 $[z_\ell,z_{\ell+1})$ 内，两个经验 CDF 都保持不变，因此插件估计值可以精确计算为：

$$
\boxed{
\widehat\mu_N
=\sum_{\ell=1}^{L-1}(z_{\ell+1}-z_\ell)
\left[
1-\widehat F_B(z_\ell)^N
-[1-\widehat F_A(z_\ell)]^N
\right]
}
$$

这里的“精确”指对已估计的经验分布精确积分，不表示没有分布估计误差。

### 5.2 外推与验证

先用较小规模数据估计分布，再预测已采集的较大规模；在验证通过后外推目标规模。例如，用 2 node 的数据预测 4 node，再用已有数据预测 8 node。

验证时，将预测值与相同定义下的观测均值比较：

$$
\overline T_{N}
=\frac1K\sum_{k=1}^{K}
\left[\max_i B_{k,i}-\min_i A_{k,i}\right].
$$

若误差随规模明显增大，应检查跨 node 依赖、单 node 分布随规模变化、尾部估计不足等因素。

## 附录 A：参考时刻的估计与含义

### A.1 采用的采集流程

每轮先执行 barrier，再执行目标通信算子。以已定位的 barrier 内部 allreduce 结束事件作为参考；需要区分该 Device 事件与 Host 侧 barrier API 的返回事件。

记：

- $e_{k,i,r}$：barrier 内部 allreduce 在该 rank 上的结束时间；
- $s_{k,i,r}$：目标通信 kernel 在该 rank 上的开始时间。

同一 rank 的两类事件需要处于可比较的时间轴上。定位事件时还应按轮次或通信序号匹配，避免把 barrier 自身的 allreduce 与目标算子混淆。

### A.2 每个 node 独立计算基准

$$
\boxed{
\widehat C_{k,i}=\min_{r\in\text{node }i}e_{k,i,r}
}
$$

这里在**本 node 内**取最小值，不跨 node。$\widehat C_{k,i}$ 表示采集时估计的本地参考时刻，其绝对数值可以因 node 和轮次而不同。模型中的 $C_k$ 则表示目标场景下各 node 共同的发起参考时刻，两者用途不同。

目标 kernel 的相对启动延迟为：

$$
x_{k,i,r}=s_{k,i,r}-\widehat C_{k,i}.
$$

所以主文中的样本可以直接计算为：

$$
A_{k,i}=\min_r s_{k,i,r}-\widehat C_{k,i},
\qquad
B_{k,i}=\max_r s_{k,i,r}-\widehat C_{k,i}.
$$

### A.3 本地基准用于提取单 node 启动延迟

每个 node 都计算相同含义的相对延迟：从本 node 的 barrier allreduce 最早结束，到本 node 目标 kernel 的最早、最晚启动。

若不同 node 的原始完整数据向量独立同分布，各 node 分别做相同的本地变换，所得 $(A,B)$ 仍然独立同分布。本地参考时刻的绝对数值不同，不妨碍估计这种相对延迟分布。

本流程将这些相对延迟样本用于估计目标“各 node 同时下发”场景中的 $F_A,F_B$。这是采集样本与目标场景之间采用的建模假设；不要求将采集期间各 node 的绝对发起偏移另行加入模型。

### A.4 同时下发假设下的同步时延

目标场景中，约定所有 node 的发起参考时刻相同：

$$
\boxed{C_{k,1}=C_{k,2}=\cdots=C_{k,N}=C_k.}
$$

设 $s^{\mathrm{model}}_{k,i,r}$ 为该场景中实际的 kernel 启动时间，则：

$$
s^{\mathrm{model}}_{k,i,r}=C_k+x_{k,i,r}.
$$

“同时下发”约束的是共同参考时刻 $C_k$，并不要求所有 rank 的 kernel 同时开始执行；其不同步由 $x_{k,i,r}$ 的差异描述。

因此，本轮同步时延为：

$$
\begin{aligned}
T_{\mathrm{sync},k}
&=\max_{i,r}s^{\mathrm{model}}_{k,i,r}
-\min_{i,r}s^{\mathrm{model}}_{k,i,r}\\
&=\max_i(C_k+B_{k,i})-\min_i(C_k+A_{k,i})\\
&=\max_i B_{k,i}-\min_i A_{k,i}.
\end{aligned}
$$

共同参考时刻 $C_k$ 完全抵消。因而在这一既定假设下，估计单 node 的 $F_A,F_B$ 后，即可直接使用正文的 node 数量外推公式；无需另外建模不同 node 的发起时刻偏移。

建议保留原始事件表 `iteration, node_id, rank_id, barrier_allreduce_end_us, target_kernel_start_us`，以及派生样本表 `iteration, node_id, baseline_us, A_us, B_us`，便于复核样本提取。

## 附录 B：同步时延期望公式的详细推导

### B.1 定义随机变量与所需假设

在各 node 同时下发的既定假设下，公共参考时刻已经抵消。省略轮次下标，记第 $i$ 个 node 最早、最晚的 kernel 启动延迟为 $A_i,B_i$，且 $A_i\le B_i$。

不同 node 的随机向量独立同分布：

$$
(A_1,B_1),\ldots,(A_N,B_N)\overset{\mathrm{iid}}{\sim}H.
$$

单 node 的两个边缘分布为：

$$
F_A(t)=P(A_i\le t),\qquad F_B(t)=P(B_i\le t).
$$

全局最早、最晚的启动延迟分别为：

$$
U_N=\min_{1\le i\le N}A_i,\qquad
V_N=\max_{1\le i\le N}B_i.
$$

因此：

$$
U_N\le V_N,\qquad T_{\mathrm{sync}}(N)=V_N-U_N.
$$

以下所有概率中的 $t$ 都是一个固定的观察时刻。随机的是各 node 的启动延迟，而不是 $t$。

### B.2 为什么最晚启动时间的 CDF 是 $F_B(t)^N$

事件 $V_N\le t$ 表示：全局最晚的启动延迟都没有超过 $t$。

这等价于每个 node 的最晚启动延迟都不超过 $t$：

$$
\{V_N\le t\}
=
\{B_1\le t,\ B_2\le t,\ldots,\ B_N\le t\}.
$$

因此：

$$
\begin{aligned}
P(V_N\le t)
&=P(B_1\le t,\ldots,B_N\le t)\\
&=\prod_{i=1}^{N}P(B_i\le t)\\
&=\prod_{i=1}^{N}F_B(t)\\
&=\boxed{F_B(t)^N}.
\end{aligned}
$$

第二步利用不同 node 的独立性，将同时发生的概率写成乘积；第三步利用同分布，每一项都是相同的 $F_B(t)$。

例如，如果一个 node 的最晚启动延迟不超过 $t$ 的概率是 $0.8$，那么两个独立 node 都满足这个条件的概率就是 $0.8^2=0.64$。

### B.3 为什么最早启动时间超过 $t$ 的概率是 $[1-F_A(t)]^N$

事件 $U_N>t$ 表示：连全局最早的启动延迟都超过了 $t$，即到 $t$ 为止还没有任何 rank 启动。

这等价于每个 node 的最早启动延迟都超过 $t$：

$$
\{U_N>t\}
=
\{A_1>t,\ A_2>t,\ldots,\ A_N>t\}.
$$

对于单个 node：

$$
P(A_i>t)=1-P(A_i\le t)=1-F_A(t).
$$

所以：

$$
\begin{aligned}
P(U_N>t)
&=P(A_1>t,\ldots,A_N>t)\\
&=\prod_{i=1}^{N}P(A_i>t)\\
&=\prod_{i=1}^{N}[1-F_A(t)]\\
&=\boxed{[1-F_A(t)]^N}.
\end{aligned}
$$

这里同样先使用跨 node 的独立性，再使用同分布。

### B.4 为什么区间覆盖概率等于两个概率的补集

由于 $U_N\le V_N$，对于任意固定时刻 $t$，只有以下三种互斥且覆盖全部可能的情况：

| 情况 | 事件 | 含义 |
|---|---|---|
| 尚无人启动 | $U_N>t$ | 所有 rank 的启动时间都在 $t$ 之后 |
| 处于启动跨度内 | $U_N\le t<V_N$ | 已有 rank 启动，同时仍有 rank 未启动 |
| 所有人均已启动 | $V_N\le t$ | 所有 rank 的启动时间都不超过 $t$ |

“已有 rank 启动”仅描述其开始事件，不表示该 rank 的算子仍在执行。

三种事件的概率之和为 1：

$$
P(U_N>t)+P(U_N\le t<V_N)+P(V_N\le t)=1.
$$

移项得到：

$$
P(U_N\le t<V_N)
=1-P(U_N>t)-P(V_N\le t).
$$

代入 B.2 和 B.3 的结果：

$$
\boxed{
P(U_N\le t<V_N)
=
1-F_B(t)^N-[1-F_A(t)]^N
}.
$$

这一步使用的是事件互斥，不需要假设 $U_N,V_N$ 独立。两者由同一批 node 数据计算得到，一般可能相关。

### B.5 为什么对覆盖概率积分，就是平均同步时延

#### B.5.1 一次观测只需计算 $V_N-U_N$

例如，一次观测得到 $U_N=1,V_N=7$，其同步时延直接为 $6$。

将时间轴上的区间 $[1,7)$ 画成高度为 1 的矩形，矩形面积也等于 $6$。把它写成积分，只是在用另一种方式表达同一个区间长度。

引入指示函数：

$$
I(t)=\mathbf{1}_{\{U_N\le t<V_N\}}
=
\begin{cases}
1,&U_N\le t<V_N,\\
0,&\text{其他情况}.
\end{cases}
$$

对于每一次观测，都有：

$$
\boxed{
V_N-U_N=\int_{-\infty}^{+\infty}I(t)\,dt
}.
$$

积分范围写成整个时间轴，是因为同步区间之外的指示值都为 0。

#### B.5.2 求平均时，可以按“可能结果”求和，也可以按“时间段”求和

下面的两种可能结果仅用于解释两种计算方式为何等价。所有时间单位为微秒：

| 可能结果 | $U$ | $V$ | 时延 $V-U$ | 发生概率 |
|---|---:|---:|---:|---:|
| 情况一 | 1 | 7 | 6 | $1/2$ |
| 情况二 | 3 | 5 | 2 | $1/2$ |

按结果求平均：

$$
\mathbb E[V-U]
=\frac12\times6+\frac12\times2
=4\ \mu s.
$$

再把时间轴切成三个长度为 $2\ \mu s$ 的时间段：

| 情况 | $[1,3)$ | $[3,5)$ | $[5,7)$ |
|---|---:|---:|---:|
| 情况一是否覆盖，概率 $1/2$ | 1 | 1 | 1 |
| 情况二是否覆盖，概率 $1/2$ | 0 | 1 | 0 |
| 对应时间段的覆盖概率 | $1/2$ | 1 | $1/2$ |

例如，$[1,3)$ 只在情况一发生时被计入同步时延，因此它平均贡献 $(1/2)\times2=1\ \mu s$；$[3,5)$ 在两种情况下都被计入，因此平均贡献 $1\times2=2\ \mu s$。

把每段的平均贡献加起来：

$$
\frac12\times2+1\times2+\frac12\times2
=4\ \mu s.
$$

两种计算完全相同。展开以后，区别只是加法顺序：

$$
\begin{aligned}
&\frac12(2+2+2)+\frac12(0+2+0)\\
&=\left(\frac12\times2+\frac12\times0\right)
+\left(\frac12\times2+\frac12\times2\right)
+\left(\frac12\times2+\frac12\times0\right).
\end{aligned}
$$

第一行先按情况统计完整时长，再按概率加权；第二行先统计各时间段的平均贡献，再沿时间轴相加。

#### B.5.3 从时间段求和到积分

将时间轴划分成长度为 $\Delta t$ 的小段，以 $t_j$ 表示第 $j$ 段的起点。

对一次观测，区间长度近似为：

$$
V_N-U_N
\approx\sum_j I(t_j)\,\Delta t.
$$

取期望，并利用期望对有限求和的线性性质：

$$
\begin{aligned}
\mathbb E[V_N-U_N]
&\approx\sum_j\mathbb E[I(t_j)]\,\Delta t\\
&=\sum_j P(U_N\le t_j<V_N)\,\Delta t.
\end{aligned}
$$

最后一步来自指示变量的基本性质：若某事件发生的概率是 $p$，其指示变量的期望就是 $1\times p+0\times(1-p)=p$。

把时间段切得越来越细，并覆盖整个时间轴，上述求和就成为：

$$
\boxed{
\mathbb E[V_N-U_N]
=
\int_{-\infty}^{+\infty}P(U_N\le t<V_N)\,dt
}.
$$

严格地说，由于 $I(t)$ 始终非负，可以根据非负函数的积分换序性质（Tonelli 定理）交换期望和时间积分：

$$
\begin{aligned}
\mathbb E[V_N-U_N]
&=\mathbb E\left[\int_{-\infty}^{+\infty}I(t)\,dt\right]\\
&=\int_{-\infty}^{+\infty}\mathbb E[I(t)]\,dt\\
&=\int_{-\infty}^{+\infty}P(U_N\le t<V_N)\,dt.
\end{aligned}
$$

这个等式也允许两边同时为正无穷；用于有限平均时延预测时，假设该积分有限。推导不要求不同时刻的指示变量彼此独立。

### B.6 代入覆盖概率，得到最终公式

把 B.4 的覆盖概率代入 B.5 的积分：

$$
\begin{aligned}
\mathbb E[T_{\mathrm{sync}}(N)]
&=\mathbb E[V_N-U_N]\\
&=\int_{-\infty}^{+\infty}P(U_N\le t<V_N)\,dt\\
&=\boxed{
\int_{-\infty}^{+\infty}
\left[
1-F_B(t)^N-[1-F_A(t)]^N
\right]dt
}.
\end{aligned}
$$

被积函数是一个概率，取值在 $[0,1]$；$dt$ 的单位是时间，所以积分的单位也是时间。

当 $N=1$ 时：

$$
\mathbb E[T_{\mathrm{sync}}(1)]
=\int_{-\infty}^{+\infty}[F_A(t)-F_B(t)]\,dt
=\mathbb E[B_1-A_1],
$$

恰好回到单个 node 内最晚、最早 kernel 启动时间差的平均值。该推导对离散分布、连续分布以及经验分布都适用，不需要引入概率密度。

