# MATR 早期设计与研究记录

历史批次资料。当前研究见[研究入口](../../simple_cptransformer_cp/RESEARCH_START_HERE.md)。

## BatteryLife-LCP 第0–4步实验设计

版本：2026-08-05

### 1. 总体问题与控制变量

第一阶段研究问题：在固定的MambaLithium式点预测器下，重新设计电芯级calibration和局部距离，能否在达到目标边际覆盖率的同时缩短RUL预测区间？

第一阶段只比较区间方法，不更换点预测器。所有CP/LCP方法必须共享同一个模型checkpoint、相同点预测、相同电芯划分和相同landmark样本。不能把点预测器的变化计入LCP收益。

### 2. 固定数据划分

保留BatteryLife官方test集合33块电芯，直到最终评价前不使用其标签进行模型选择、距离选择或带宽选择。

| 集合 | 来源 | 数量 | 用途 |
|---|---|---:|---|
| Train | 官方train的固定子集 | 90 | 模型训练、输入标准化拟合 |
| Model-validation | 官方train的固定子集 | 12 | 早停和模型checkpoint选择 |
| Calibration | 官方validation全集 | 34 | 每个landmark的CP/LCP校准 |
| Test | 官方test全集 | 33 | 最终评价 |

官方train内部采用seed 42固定划分，并按batch和寿命分位数组合分层，使12块model-validation电芯尽量覆盖b1–b4及不同寿命范围。划分完成后保存为JSON，所有方法复用，禁止按结果重新抽取。

目标覆盖率主结果设为90%（`alpha=0.1`）。`n_cal=34`时Split CP顺序统计量位置为：

\[
k=\lceil(34+1)(1-0.1)\rceil=32,
\]

不再使用最大校准残差。80%、85%、95%只在主流程稳定后作为coverage-width补充实验。

### 3. Landmark与样本定义

主landmark预先固定为：

\[
\mathcal T=\{50,100,150,200\}.
\]

真实标签为：

\[
Y_{i,t}=L_i-t.
\]

数据检查结果：上述四个时刻中34块calibration和33块test电芯均有观测，因此各时刻使用相同电芯总体，避免因短寿命电芯退出造成评价总体变化。

模型训练使用逐周期/稀疏逐周期标签。默认从第50周期开始，每10周期取一个监督点：

\[
t=50,60,70,\ldots,L_i-1.
\]

同一电芯的全部时刻只属于一个数据集合。训练损失按电芯平衡：先计算每块电芯各时刻平均L1损失，再对电芯求平均，避免长寿命电芯因监督点更多而获得更大权重。

模型输入在时刻`t`只能使用`1:t`循环。禁止使用`t`之后的数据计算输入、标准化参数或距离特征。

### 4. 第0步：冻结Mamba预测器

保留MambaLithium核心配置：

| 项目 | 固定值 |
|---|---|
| 主干 | 项目内Mamba实现 |
| hidden / d_model | 16 |
| Mamba layers | 2 |
| 回归损失 | L1 |
| 优化器 | Adam |
| 初始学习率 | 0.01 |
| 最大epoch | 100 |
| checkpoint | model-validation cell-balanced MAE最低 |

为了适配BatteryLife且消除泄漏，必须进行以下协议修正：

- 每块电芯是独立序列，不能把多块电芯首尾拼接成一条连续序列；
- 输入标准化参数只在Train电芯上拟合；
- RUL标签使用Train确定的一个全局尺度，不能除以测试电芯的真实总寿命；
- 推理时不能读取测试电芯完整序列长度；
- padding位置不参与损失和embedding导出。

第一版输入沿用现有BatteryLife管线的每循环31维统计特征，以隔离数据特征改动。`Qdlin`、协议特征和更复杂曲线特征留到点预测模型优化阶段，不在第0–4步中加入。

每个landmark导出Train、Model-validation、Calibration和Test的：`cell_id, t, true_rul, pred_rul, residual, embedding`。Test residual只用于最终评价。

### 5. 第1步：逐周期RUL点预测基准

训练一次冻结模型，生成所有监督时刻的RUL预测。在四个landmark分别报告：

- 电芯级RMSE、MAE、MAPE和R²；
- 平均有符号误差，检查系统性高估/低估；
- 残差中位数、90%分位数和最大值。

另外报告跨时刻汇总的cell-balanced指标：先对每块电芯在所有评价时刻求平均，再对电芯求平均。不能只把全部`(cell,t)`混合后计算一个overall指标。

验收条件：模型能够无泄漏地产生四个landmark的预测；所有区间方法读取完全相同的预测文件，不重复训练模型。

### 6. 第2步：逐landmark Split CP

在每个`t`单独使用34块calibration电芯：

\[
R_{i,t}=|Y_{i,t}-\hat Y_{i,t}|,
\qquad
q_t=R_{(32),t}.
\]

测试区间为：

\[
C_t(X_{*,t})=
[\max(0,\hat Y_{*,t}-q_t),\ \hat Y_{*,t}+q_t].
\]

下界截断到0，因为RUL物理上非负；不得根据数据集最大寿命随意截断上界。

每个landmark报告：目标覆盖率、实际覆盖率、覆盖电芯数/33、平均宽度、中位宽度、宽度90%分位数和`q_t`。Split CP是所有后续方法的唯一主基准。

### 7. 第3步：两类候选局部表示

所有表示变换仅在Train拟合，在Model-validation上选择，之后冻结；禁止使用Calibration/Test标签选择距离。

#### 7.1 低维观测状态表示（State-LCP）

从截至`t`的31维循环统计序列构造低维、预测时可见的状态：

- 当前放电容量相对前5循环基准的保持率；
- 最近10循环放电容量线性斜率；
- 最近10循环放电容量残差波动；
- 当前内阻或最近可用内阻；
- 最近10循环平均温度。

缺失率过高的内阻/温度维度根据Train统计决定是否删除，不允许事后根据Test效果决定。所有维度使用Train中位数填补和robust scaling。距离采用标准化后的欧氏距离。

#### 7.2 降维embedding表示（PCA-LCP）

使用冻结Mamba在`t`处的16维隐藏表示。对每个landmark分别：

1. 使用Train embedding拟合标准化器；
2. 使用Train拟合whitened PCA；
3. 主设置固定最多保留5个主成分；
4. 若5个成分累计解释方差低于80%，记录该事实但不利用Test修改维数；
5. 在PCA空间计算欧氏距离。

不直接在原始高维embedding上计算距离。State-LCP和PCA-LCP必须分别报告，不能只保留结果更好的一种而不披露选择过程。

#### 7.3 邻域有效性诊断

在使用LCP前，必须报告：

- Calibration到Test的最近邻距离分布；
- 每个测试点的有效样本量ESS；
- Model-validation上“距离近是否对应绝对残差更相近”的相关性；
- State/PCA空间中的batch混合程度。

这些诊断用于判断局部化是否有数据依据，不能把“数学上总能找到最近邻”等同于存在有效退化邻域。

### 8. 第4步：ESS约束的LCP及回退

对每个landmark、每个测试电芯计算高斯核权重：

\[
w_i(x)=\exp\left(-\frac{\|z_i-z(x)\|^2}{2h(x)^2}\right).
\]

带宽采用自适应k近邻规则：`h(x)`取第k个calibration邻居的距离。主设置预先固定`k=15`，对应34块calibration中的中等局部范围。不得使用Test标签选择k。

有效样本量定义为：

\[
ESS(x)=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\]

安全规则：

- 若`ESS >= 10`，计算局部加权conformal分位数；
- 若`ESS < 10`，逐步扩大带宽直到`ESS >= 10`；
- 若扩大到近似全局后仍不满足算法有限分位数条件，则回退到该landmark的Split CP区间；
- 每次回退都必须记录，不能静默处理。

LCP分位数必须包含测试点的conformal校正质量，不能简单计算校准残差的普通加权90%分位数后声称具有严格覆盖保证。具体实现应复用并核对项目`python_lcp`中的算法定义，另用合成数据验证边界情况。

每种距离分别产生State-LCP和PCA-LCP结果，并报告：

- coverage及覆盖数；
- mean/median/90% width；
- 相对Split CP的平均宽度变化；
- ESS的min/median/mean；
- 回退比例；
- 按batch和预测误差难度分组的探索性覆盖率。

### 9. 方法选择与成功标准

主结论不以单次“宽度最小”为准。第0–4步的成功标准为：

1. 四个landmark中不存在明显系统性欠覆盖；
2. LCP在多数landmark达到与Split CP相当的经验覆盖；
3. 在相当覆盖下，mean或median width稳定下降；
4. ESS和回退率表明缩窄不是由极少数邻居造成；
5. 多seed重复后方向一致。

33块Test下每块电芯约占3.03个百分点。因此90%目标附近的覆盖数只能是离散的，`30/33=90.91%`不应被解释为精确估计到小数点后两位。

### 10. 预期输出结构

```text
battery_lcp/outputs/rul_landmark_seed42/
  split.json
  config.json
  model.pt
  training_history.csv
  predictions_all.csv
  landmark_050.csv
  landmark_100.csv
  landmark_150.csv
  landmark_200.csv
  metrics_point.json
  metrics_split_cp.json
  metrics_state_lcp.json
  metrics_pca_lcp.json
  neighborhood_diagnostics.csv
```

实现完成前不得覆盖现有`outputs/matr_seed42`，旧的固定前50循环总寿命实验作为历史结果保留。

## CPTransformer residual audit

日期：2026-08-19

### 范围与使用边界

本报告汇总 seed 42、2021、2024 的现有 Calibration/Test 预测，只用于解释误差来源。Test 标签属于事后诊断信息，不得据此调参后再把同一 Test 结果当作无偏最终评价。

### 主要发现

- 共检查 67 块 Calibration/Test 电芯、4 个 landmark、3 个 seed，合计 804 条预测。
- 绝对残差中位数为 46.14 cycles，90% 分位为 187.40，但99%分位达到962.13，误差是明显的重尾分布。
- 最长寿命四分位的平均绝对残差为280.94，其余三个四分位为50.71，相差5.54倍。
- 绝对残差与真实总寿命的 Spearman 相关系数在各时刻约为0.43–0.56；现有状态量与误差的相关性更弱，容量保持率最高约0.40，内阻接近0。
- 最难电芯在三个 seed 和四个时刻重复出现，并非偶然训练波动。误差符号几乎始终为正，即 `true_rul - pred_rul > 0`，表现为系统性低估长寿命电芯。

### Split CP 宽度的直接来源

Calibration 有34块电芯，alpha=0.1时有限样本 conformal 阶数为：

\[
k=\lceil(34+1)(1-0.1)\rceil=32.
\]

即使用升序第32个残差，也就是第三大残差。三个 seed、四个时刻中，Calibration 的前两大残差稳定来自：

- `MATR_b1c2.pkl`，总寿命2237 cycles，绝对误差约847–1086；
- `MATR_b3c38.pkl`，总寿命1932 cycles，绝对误差约568–839。

它们位于 conformal 阈值之上，因此不会直接成为90%阈值。第三大残差稳定来自 `MATR_b2c47.pkl`，总寿命712 cycles，绝对误差约255–398；它基本决定 Split CP 半宽。

Test 中三块反复漏覆盖的长寿命电芯为：

- `MATR_b1c1.pkl`，总寿命2154，跨 seed/时刻平均绝对误差875.82；
- `MATR_b1c0.pkl`，总寿命1858，平均绝对误差727.20；
- `MATR_b1c4.pkl`，总寿命1705，平均绝对误差577.15。

因此 Split CP 常得到30/33=90.91%覆盖。Test 每多漏一块，覆盖率就下降3.03个百分点。

### 对当前 LCP 的解释

Diagonal-state LCP 的五个状态特征没有充分识别长寿命难例，学习结果趋向较大 ESS 的近全局核，所以常退化成 Split CP。Low-rank joint LCP 的平均 ESS 约14–19，在90个 OOF 电芯和34个 Calibration 电芯下方差过大。Spectral-band LCP 可以为部分长寿命电芯产生更宽区间并提高覆盖，但当前频域特征区分力不足，导致大量非难例也被扩宽。Quantile-profile LCP 在 t=200 出现稳定缩窄信号，说明直接学习“条件残差分布”比仅学习输入几何距离更符合当前误差结构。

### 下一阶段判断

首要矛盾是长寿命电芯的条件误差尺度，而不是普通样本的平均误差。因此下一步优先级应为：

1. 使用 Train OOF 残差训练误差尺度模型，实施 Normalized CP；
2. 训练上下条件分位数，实施 CQR；
3. 将 normalized score 或 CQR score 与保守的 LCP 权重组合；
4. 所有方法选择只使用 Train/Model-validation/Calibration，Test 保持冻结；
5. 完成后再在第二数据集做外部验证。

详细数据位于 `battery_lcp/outputs/residual_audit/`。

## BatteryLife-LCP 科研技术日志

> 用途：持续记录研究问题、实验口径、关键决策、代码变更和结果解释。每次修改实验设计或代码后，在“变更记录”顶部追加一条记录；正式指标必须注明数据划分、随机种子和配置。

### 1. 研究目标

在 BatteryLife 的 MATR 数据上研究锂离子电池寿命/RUL预测的不确定性量化。以 Mamba 类序列模型作为点预测器，比较 Split Conformal Prediction（Split CP）与 Localized Conformal Prediction（LCP），重点考察：

- 预测区间的边际覆盖率；
- 在相同覆盖水平下的区间宽度；
- 模型误差是否随可观测状态发生异方差；
- 局部表示能否识别残差分布相近的电芯；
- 跨批次分布漂移下方法的稳健性与失效边界。

### 2. 当前主干流程

```text
BatteryLife MATR
  -> 按电芯划分 Train / Validation / Calibration / Test
  -> 提取每块电芯前 50 个循环的统计特征
  -> Mamba 预测总循环寿命并输出 embedding
  -> Split CP / 基于 embedding 距离的 LCP
  -> 评价点预测、覆盖率和区间宽度
```

独立样本单位是电芯。同一电芯产生的所有循环或窗口必须归入同一数据集合，禁止跨 Train、Validation、Calibration、Test 划分。

### 3. 数据与当前任务

- 处理后数据：`data/BatteryLife_Processed`
- 数据子集：MATR
- 可用电芯：169 块，每个 `.pkl` 文件对应一块电芯
- 当前划分：Train 102、Validation 17、Calibration 17、Test 33
- 当前任务：使用一块电芯的前 50 个循环预测总循环寿命
- 当前任务不是逐周期 RUL。逐周期 RUL 应定义为：

  \[
  X_{i,1:t}\longrightarrow Y_{i,t}=L_i-t.
  \]

原始循环字段包括电流、电压、充/放电容量、温度、内阻和 `Qdlin`，另有充放电协议等电芯级元数据。当前代码对六类信号分别计算 mean、std、min、max、last，每个循环得到 31 维特征，前 50 循环构成序列。

### 4. 已完成工作

#### 4.1 MambaLithium原代码复现

- 完成 Case A、RUL 任务运行。
- seed 1 的一次结果约为 RMSE 15.72、R² 0.9991。
- 多个随机种子的 RMSE 约为 15.72–24.94。
- 该结果只证明原代码能够运行，不能与当前 BatteryLife 实验直接比较：任务定义、数据、样本组织及划分口径不一致。

#### 4.2 无泄漏BatteryLife主干实验

- 使用BatteryLife官方电芯划分；官方validation集合再分为模型验证集和conformal calibration集。
- 训练、验证、校准和测试电芯互不重叠。
- 已完成Mamba点预测、Split CP和LCP的端到端运行。

seed 42、目标覆盖率 90%（`alpha=0.1`）的初步结果：

| 项目 | 结果 |
|---|---:|
| RMSE | 226.65 cycles |
| MAE | 137.31 cycles |
| MAPE | 13.50% |
| R² | 0.6562 |
| Split CP覆盖率 | 90.91%（30/33） |
| Split CP平均宽度 | 830.07 cycles |
| LCP覆盖率 | 90.91%（30/33） |
| LCP平均宽度 | 778.70 cycles |

解释：本次运行中LCP在相同经验覆盖率下将平均宽度降低约6.2%，但区间仍过宽。只有一个随机种子，calibration仅17块、test仅33块，因此不能据此得出稳定优越性结论。

### 5. 当前方法的准确表述

当前LCP依据Mamba从前50循环学习到的32维embedding定义距离。因此应表述为：

> 使用早期循环观测的学习表示定义局部相似性。

目前不能直接称为“物理退化状态相近”，因为尚未证明embedding对应真实退化机制，也未验证embedding近邻具有相近的残差分布。

当前实现是基于学习表示的局部加权LCP，不是Hore–Barber定义的完整Randomized LCP（RLCP）。后续实现RLCP时必须单独记录算法公式、随机局部化步骤和理论适用条件。

### 6. 已知问题与风险

1. **任务不一致**：当前为固定前50循环的总寿命预测，MambaLithium为逐周期RUL，不能直接用RMSE比较。
2. **校准样本较少**：`n_cal=17`、90%目标覆盖时Split CP基本使用最大校准残差，区间容易被异常电芯拉宽。
3. **点预测误差较大**：Split CP阈值约415 cycles，导致完整区间宽度约830 cycles。
4. **局部性尚未验证**：需要验证embedding距离、观测轨迹相似度与残差分布相似度之间的关系。
5. **异方差尚未证实**：固定前50循环实验不能回答误差是否随退化过程变化。
6. **覆盖率分辨率有限**：33块测试电芯下，一个电芯约对应3.03个百分点。
7. **跨批次漂移**：b1 calibration、b4 test时普通可交换性可能失效；LCP/RLCP不能无条件解决条件分布漂移或支持集不重叠。

### 7. 下一阶段路线

#### 阶段A：建立正式逐周期RUL主干

- 按电芯固定Train/Validation/Calibration/Test归属；
- 在每块电芯内部生成多个观测时刻或窗口；
- 目标定义为 `RUL = total_life - t`；
- 在完全相同的数据、窗口和划分下运行Mamba、Mamba+Split CP、Mamba+LCP；
- 同时报告pointwise coverage与cell-balanced coverage。

#### 阶段B：验证LCP的研究动机

- 按可观测SOH、容量下降斜率、循环位置和embedding邻域统计out-of-sample残差；
- 使用按电芯cluster bootstrap估计不确定性，不能把同一电芯的循环当作独立样本；
- 检查邻居的容量/电压轨迹和充电协议；
- 报告最近邻距离、局部有效样本量（ESS）及超出支持集的测试点；
- 检验局部距离是否能解释绝对残差或残差分位数变化。

#### 阶段C：稳定性与扩展实验

- 多随机种子或电芯级重复划分；
- 绘制80%、85%、90%、95%的coverage-width曲线；
- 研究normalized/adaptive CP；
- 主实验稳定后再实现RLCP和跨batch压力测试。

### 8. 实验记录规范

每次正式运行至少记录：

- 日期、研究问题和代码入口；
- 数据路径与数据版本；
- 任务定义、观测窗口和标签定义；
- 电芯级划分及各集合电芯数；
- seed、模型超参数、训练轮数与早停规则；
- conformal方法、非一致性分数、`alpha`、距离和带宽；
- RMSE、MAE、MAPE、R²；
- 目标/经验覆盖率、平均/中位区间宽度、无限区间比例；
- 失败实验、异常现象和解释；
- 结果文件路径。

### 9. 变更记录

#### 2026-08-05：完成第0–4步seed 42正式运行

- 新增`battery_lcp/run_rul_landmarks.py`，旧`run_experiment.py`及其历史结果保持不变。
- 实现电芯独立长序列、每10周期RUL监督、按电芯平衡L1损失、四个landmark导出、逐landmark Split CP、State-LCP、whitened PCA-LCP、自适应带宽、ESS诊断和Split CP回退。
- 正式划分为Train 90、Model-validation 12、Calibration 34、Test 33；四个landmark均有12/34/33块完整的Model-validation/Calibration/Test电芯。
- 模型第64轮早停，最佳checkpoint位于第44轮，Model-validation cell-balanced MAE为54.96 cycles。

seed 42正式结果：

| t | Mamba RMSE | Split覆盖/宽度 | State-LCP覆盖/宽度 | PCA-LCP覆盖/宽度 |
|---:|---:|---:|---:|---:|
| 50 | 282.67 | 90.91% / 618.77 | 90.91% / 564.05 | 96.97% / 815.98 |
| 100 | 293.55 | 90.91% / 567.92 | 87.88% / 545.49 | 84.85% / 486.80 |
| 150 | 245.05 | 90.91% / 455.91 | 90.91% / 531.90 | 84.85% / 424.00 |
| 200 | 256.59 | 87.88% / 387.89 | 87.88% / 472.29 | 84.85% / 375.33 |

初步结论：State-LCP只在t=50实现“相同经验覆盖下缩短区间”，从618.77降至564.05（约8.8%）；t=100轻微缩窄但少覆盖1块；t=150和200反而变宽。PCA-LCP在t=100–200缩窄时均出现更明显欠覆盖，t=50提高覆盖则大幅变宽。四个时刻的validation残差相似性相关系数整体较弱或为负，尚不支持当前State/PCA距离能稳定识别残差分布相近的邻域。

该结果只来自一个固定划分和seed，不能作为最终优越性结论。输出位于`battery_lcp/outputs/rul_landmark_seed42/`。

#### 2026-08-05：完成第0–4步正式实验设计

- 新建设计文档`docs/history/MATR_RESEARCH.md`。
- 固定官方test 33块；官方train划分为Train 90和Model-validation 12；官方validation 34块全部作为Calibration。
- 主landmark固定为50、100、150、200周期；数据检查确认34块Calibration和33块Test在四个时刻均可用。
- 固定MambaLithium核心预测器，定义无泄漏逐周期RUL样本、逐landmark Split CP、State/PCA两种局部表示，以及`k=15`、最低`ESS=10`的自适应局部化与Split CP回退规则。
- 现有固定前50循环预测总寿命的输出保留，不被新实验覆盖。

#### 2026-08-05：确定第一阶段冻结Mamba预测器

- 第一阶段不进行预测模型横向比较或结构调优，先固定MambaLithium的核心网络配置，完成逐周期RUL、landmark校准以及Split CP/LCP系列实验。
- “固定原模型”指保留Mamba核心架构和主要训练超参数，不表示照搬原始`main.py`的数据处理。原脚本中依赖完整测试寿命长度的目标归一化/反归一化、单电芯全寿命标准化等步骤不适用于无泄漏的提前RUL预测。
- BatteryLife适配必须遵守：电芯级数据划分、只用训练数据拟合标准化参数、每个时刻只能使用截至该时刻的观测、推理时不得使用测试电芯总寿命。
- 在固定预测器下依次比较：新Split CP基准、低维状态LCP、PCA-embedding LCP、ESS约束的自适应局部化、Normalized CP；方法阶段完成后再更换或优化点预测模型。

#### 2026-08-05：项目清理并建立科研日志

- 审计项目目录；当前目录尚未初始化为Git仓库。
- 保留原始论文、第三方BatteryLife代码、MambaLithium基线、正式MATR实验结果和复现日志。
- 删除Python缓存以及可通过脚本重新生成的smoke数据和smoke输出。
- 建立本日志，汇总研究目标、当前实验、结论边界和后续路线。

#### 2026-08-19：CPTransformer 与可学习核端到端实验

- 研究目的：保持 MATR 逐周期 RUL、landmark 50/100/150/200、90% 目标覆盖率和电芯级数据隔离不变，用 BatteryLife 的 CPTransformer 组件替换 Mamba，并比较四种可学习局部核。
- 代码变更：新增 `battery_lcp/run_cptransformer_kernels.py`、`battery_lcp/analyze_cptransformer_kernels.py` 和根目录设计文档 `CPTRANSFORMER_KERNEL_DESIGN.md`。预测器复用 BatteryLife 的 MLPBlock、位置编码与 Transformer Encoder；增加 causal mask 和逐周期线性 RUL head。
- 数据/划分：每次为 Train 90、Model-validation 12、Calibration 34、Test 33；Calibration/Test 固定，Train/Model-validation 随 seed 在官方训练池内划分。核只由 Train 的 3 折 cell-level OOF 残差学习，Calibration/Test 标签不参与核训练。
- 方法：Split CP；五维状态的 learned diagonal kernel；状态+8维 PCA embedding 的 rank-3 low-rank kernel；残差 0.5/0.75/0.9 条件分位画像核；50-cycle 趋势+去趋势 RFFT 五频带的 spectral kernel。最低 ESS=10，不足时扩带，失败时回退 Split CP。
- 正式配置：seed 42/2021/2024，最终模型最多100 epoch、patience=20；3折 OOF 模型固定40 epoch；kernel 400 steps；d_model=64、d_ff=128、4 heads、2层 Transformer；alpha=0.1。
- 点预测：三 seed 平均 RMSE 在 t=50/100/150/200 分别为 240.09/244.21/233.94/213.18 cycles；平均 MAE 为 123.15/118.40/114.95/105.20；MAPE 为 11.43%/11.34%/11.96%/12.11%。相较旧 Mamba seed42，点预测总体改善，但这不是严格的同 seed 聚合模型比较。
- CP 结果：Split CP 三 seed 在四个时刻均为 30/33=90.91% 覆盖，平均宽度为 737.64/682.14/626.85/559.20。Diagonal kernel 基本退化为全局核，与 Split 几乎相同。Low-rank kernel 波动最大，t=150 三次均欠覆盖。Quantile-profile 在 t=200 三次均保持90.91%覆盖且平均宽度524.73，比 Split 平均缩短34.47 cycles（6.16%），但其余时刻不稳定。Spectral kernel 三次均达到目标覆盖，但平均区间没有稳定缩窄，且部分时刻明显变宽。
- 结论：目前唯一有重复实验支持的正向信号是 t=200 的 quantile-profile kernel；仍不足以声称 learnable LCP 整体优于 Split CP。小 test 集每块电芯对应3.03个百分点，87.88%与90.91%只差1块，必须同时报告 covered/total。
- 异常1：BatteryLife 的 `SelfAttention_Family.py` 无条件导入可选包 `reformer_pytorch`，CPTransformer 本身并不用 Reformer，却导致启动失败。已改为可选导入；仅实际构造 ReformerLayer 时才要求该依赖。
- 异常2：自动终端没有继承交互式 `my_torch` 环境，直接调用 `python` 报 `ModuleNotFoundError: torch`。正式复验统一使用 `python`，不影响已完成结果。
- 验证：对 causal mask 做过未来输入扰动测试，前50周期输出最大变化为0；三个正式运行均产生完整模型、OOF预测、kernel参数、逐电芯区间和诊断文件。
- 输出位置：`battery_lcp/outputs/cptransformer_learned_kernels_seed{42,2021,2024}/`；跨 seed 汇总为 `battery_lcp/outputs/cptransformer_*_summary.csv` 和 `cptransformer_*_runs.csv`。

#### 2026-08-19：第一阶段 residual audit

- 新增 `battery_lcp/audit_cptransformer_residuals.py` 和 `docs/history/MATR_RESEARCH.md`，汇总三次正式运行的 Calibration/Test 残差、寿命分层、相关性、反复难例、Calibration 尾部及 Test 漏覆盖电芯。
- 共804条预测的绝对残差中位数46.14、q90为187.40、q99为962.13；最长寿命四分位平均绝对误差280.94，其余四分位50.71，确认存在强烈的长寿命尾部异方差与系统性低估。
- n_cal=34、alpha=0.1时阈值为第三大残差。Calibration 前两大难例位于阈值以上，Test 则有三块长寿命难例稳定超阈值，解释了30/33覆盖和大区间同时出现的原因。
- Test 分析仅作事后诊断；后续方法训练、选择和超参数调整不得使用这些 Test 标签。
