# BatteryLife-LCP 第0–4步实验设计

版本：2026-08-05

## 1. 总体问题与控制变量

第一阶段研究问题：在固定的MambaLithium式点预测器下，重新设计电芯级calibration和局部距离，能否在达到目标边际覆盖率的同时缩短RUL预测区间？

第一阶段只比较区间方法，不更换点预测器。所有CP/LCP方法必须共享同一个模型checkpoint、相同点预测、相同电芯划分和相同landmark样本。不能把点预测器的变化计入LCP收益。

## 2. 固定数据划分

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

## 3. Landmark与样本定义

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

## 4. 第0步：冻结Mamba预测器

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

## 5. 第1步：逐周期RUL点预测基准

训练一次冻结模型，生成所有监督时刻的RUL预测。在四个landmark分别报告：

- 电芯级RMSE、MAE、MAPE和R²；
- 平均有符号误差，检查系统性高估/低估；
- 残差中位数、90%分位数和最大值。

另外报告跨时刻汇总的cell-balanced指标：先对每块电芯在所有评价时刻求平均，再对电芯求平均。不能只把全部`(cell,t)`混合后计算一个overall指标。

验收条件：模型能够无泄漏地产生四个landmark的预测；所有区间方法读取完全相同的预测文件，不重复训练模型。

## 6. 第2步：逐landmark Split CP

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

## 7. 第3步：两类候选局部表示

所有表示变换仅在Train拟合，在Model-validation上选择，之后冻结；禁止使用Calibration/Test标签选择距离。

### 7.1 低维观测状态表示（State-LCP）

从截至`t`的31维循环统计序列构造低维、预测时可见的状态：

- 当前放电容量相对前5循环基准的保持率；
- 最近10循环放电容量线性斜率；
- 最近10循环放电容量残差波动；
- 当前内阻或最近可用内阻；
- 最近10循环平均温度。

缺失率过高的内阻/温度维度根据Train统计决定是否删除，不允许事后根据Test效果决定。所有维度使用Train中位数填补和robust scaling。距离采用标准化后的欧氏距离。

### 7.2 降维embedding表示（PCA-LCP）

使用冻结Mamba在`t`处的16维隐藏表示。对每个landmark分别：

1. 使用Train embedding拟合标准化器；
2. 使用Train拟合whitened PCA；
3. 主设置固定最多保留5个主成分；
4. 若5个成分累计解释方差低于80%，记录该事实但不利用Test修改维数；
5. 在PCA空间计算欧氏距离。

不直接在原始高维embedding上计算距离。State-LCP和PCA-LCP必须分别报告，不能只保留结果更好的一种而不披露选择过程。

### 7.3 邻域有效性诊断

在使用LCP前，必须报告：

- Calibration到Test的最近邻距离分布；
- 每个测试点的有效样本量ESS；
- Model-validation上“距离近是否对应绝对残差更相近”的相关性；
- State/PCA空间中的batch混合程度。

这些诊断用于判断局部化是否有数据依据，不能把“数学上总能找到最近邻”等同于存在有效退化邻域。

## 8. 第4步：ESS约束的LCP及回退

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

## 9. 方法选择与成功标准

主结论不以单次“宽度最小”为准。第0–4步的成功标准为：

1. 四个landmark中不存在明显系统性欠覆盖；
2. LCP在多数landmark达到与Split CP相当的经验覆盖；
3. 在相当覆盖下，mean或median width稳定下降；
4. ESS和回退率表明缩窄不是由极少数邻居造成；
5. 多seed重复后方向一致。

33块Test下每块电芯约占3.03个百分点。因此90%目标附近的覆盖数只能是离散的，`30/33=90.91%`不应被解释为精确估计到小数点后两位。

## 10. 预期输出结构

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
