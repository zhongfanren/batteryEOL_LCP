# BatteryLife-LCP 科研技术日志

> 用途：持续记录研究问题、实验口径、关键决策、代码变更和结果解释。每次修改实验设计或代码后，在“变更记录”顶部追加一条记录；正式指标必须注明数据划分、随机种子和配置。

## 1. 研究目标

在 BatteryLife 的 MATR 数据上研究锂离子电池寿命/RUL预测的不确定性量化。以 Mamba 类序列模型作为点预测器，比较 Split Conformal Prediction（Split CP）与 Localized Conformal Prediction（LCP），重点考察：

- 预测区间的边际覆盖率；
- 在相同覆盖水平下的区间宽度；
- 模型误差是否随可观测状态发生异方差；
- 局部表示能否识别残差分布相近的电芯；
- 跨批次分布漂移下方法的稳健性与失效边界。

## 2. 当前主干流程

```text
BatteryLife MATR
  -> 按电芯划分 Train / Validation / Calibration / Test
  -> 提取每块电芯前 50 个循环的统计特征
  -> Mamba 预测总循环寿命并输出 embedding
  -> Split CP / 基于 embedding 距离的 LCP
  -> 评价点预测、覆盖率和区间宽度
```

独立样本单位是电芯。同一电芯产生的所有循环或窗口必须归入同一数据集合，禁止跨 Train、Validation、Calibration、Test 划分。

## 3. 数据与当前任务

- 处理后数据：`D:\BatteryLife_Processed`
- 数据子集：MATR
- 可用电芯：169 块，每个 `.pkl` 文件对应一块电芯
- 当前划分：Train 102、Validation 17、Calibration 17、Test 33
- 当前任务：使用一块电芯的前 50 个循环预测总循环寿命
- 当前任务不是逐周期 RUL。逐周期 RUL 应定义为：

  \[
  X_{i,1:t}\longrightarrow Y_{i,t}=L_i-t.
  \]

原始循环字段包括电流、电压、充/放电容量、温度、内阻和 `Qdlin`，另有充放电协议等电芯级元数据。当前代码对六类信号分别计算 mean、std、min、max、last，每个循环得到 31 维特征，前 50 循环构成序列。

## 4. 已完成工作

### 4.1 MambaLithium原代码复现

- 完成 Case A、RUL 任务运行。
- seed 1 的一次结果约为 RMSE 15.72、R² 0.9991。
- 多个随机种子的 RMSE 约为 15.72–24.94。
- 该结果只证明原代码能够运行，不能与当前 BatteryLife 实验直接比较：任务定义、数据、样本组织及划分口径不一致。

### 4.2 无泄漏BatteryLife主干实验

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

## 5. 当前方法的准确表述

当前LCP依据Mamba从前50循环学习到的32维embedding定义距离。因此应表述为：

> 使用早期循环观测的学习表示定义局部相似性。

目前不能直接称为“物理退化状态相近”，因为尚未证明embedding对应真实退化机制，也未验证embedding近邻具有相近的残差分布。

当前实现是基于学习表示的局部加权LCP，不是Hore–Barber定义的完整Randomized LCP（RLCP）。后续实现RLCP时必须单独记录算法公式、随机局部化步骤和理论适用条件。

## 6. 已知问题与风险

1. **任务不一致**：当前为固定前50循环的总寿命预测，MambaLithium为逐周期RUL，不能直接用RMSE比较。
2. **校准样本较少**：`n_cal=17`、90%目标覆盖时Split CP基本使用最大校准残差，区间容易被异常电芯拉宽。
3. **点预测误差较大**：Split CP阈值约415 cycles，导致完整区间宽度约830 cycles。
4. **局部性尚未验证**：需要验证embedding距离、观测轨迹相似度与残差分布相似度之间的关系。
5. **异方差尚未证实**：固定前50循环实验不能回答误差是否随退化过程变化。
6. **覆盖率分辨率有限**：33块测试电芯下，一个电芯约对应3.03个百分点。
7. **跨批次漂移**：b1 calibration、b4 test时普通可交换性可能失效；LCP/RLCP不能无条件解决条件分布漂移或支持集不重叠。

## 7. 下一阶段路线

### 阶段A：建立正式逐周期RUL主干

- 按电芯固定Train/Validation/Calibration/Test归属；
- 在每块电芯内部生成多个观测时刻或窗口；
- 目标定义为 `RUL = total_life - t`；
- 在完全相同的数据、窗口和划分下运行Mamba、Mamba+Split CP、Mamba+LCP；
- 同时报告pointwise coverage与cell-balanced coverage。

### 阶段B：验证LCP的研究动机

- 按可观测SOH、容量下降斜率、循环位置和embedding邻域统计out-of-sample残差；
- 使用按电芯cluster bootstrap估计不确定性，不能把同一电芯的循环当作独立样本；
- 检查邻居的容量/电压轨迹和充电协议；
- 报告最近邻距离、局部有效样本量（ESS）及超出支持集的测试点；
- 检验局部距离是否能解释绝对残差或残差分位数变化。

### 阶段C：稳定性与扩展实验

- 多随机种子或电芯级重复划分；
- 绘制80%、85%、90%、95%的coverage-width曲线；
- 研究normalized/adaptive CP；
- 主实验稳定后再实现RLCP和跨batch压力测试。

## 8. 实验记录规范

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

## 9. 变更记录

### 2026-08-05：完成第0–4步seed 42正式运行

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

### 2026-08-05：完成第0–4步正式实验设计

- 新建设计文档`EXPERIMENT_DESIGN_STEPS_0_4.md`。
- 固定官方test 33块；官方train划分为Train 90和Model-validation 12；官方validation 34块全部作为Calibration。
- 主landmark固定为50、100、150、200周期；数据检查确认34块Calibration和33块Test在四个时刻均可用。
- 固定MambaLithium核心预测器，定义无泄漏逐周期RUL样本、逐landmark Split CP、State/PCA两种局部表示，以及`k=15`、最低`ESS=10`的自适应局部化与Split CP回退规则。
- 现有固定前50循环预测总寿命的输出保留，不被新实验覆盖。

### 2026-08-05：确定第一阶段冻结Mamba预测器

- 第一阶段不进行预测模型横向比较或结构调优，先固定MambaLithium的核心网络配置，完成逐周期RUL、landmark校准以及Split CP/LCP系列实验。
- “固定原模型”指保留Mamba核心架构和主要训练超参数，不表示照搬原始`main.py`的数据处理。原脚本中依赖完整测试寿命长度的目标归一化/反归一化、单电芯全寿命标准化等步骤不适用于无泄漏的提前RUL预测。
- BatteryLife适配必须遵守：电芯级数据划分、只用训练数据拟合标准化参数、每个时刻只能使用截至该时刻的观测、推理时不得使用测试电芯总寿命。
- 在固定预测器下依次比较：新Split CP基准、低维状态LCP、PCA-embedding LCP、ESS约束的自适应局部化、Normalized CP；方法阶段完成后再更换或优化点预测模型。

### 2026-08-05：项目清理并建立科研日志

- 审计项目目录；当前目录尚未初始化为Git仓库。
- 保留原始论文、第三方BatteryLife代码、MambaLithium基线、正式MATR实验结果和复现日志。
- 删除Python缓存以及可通过脚本重新生成的smoke数据和smoke输出。
- 建立本日志，汇总研究目标、当前实验、结论边界和后续路线。

### 后续记录模板

```markdown
### YYYY-MM-DD：变更标题

- 研究目的：
- 代码变更：
- 数据/划分：
- 运行命令与配置：
- 结果：
- 结论：
- 已知问题：
- 输出位置：
```

### 2026-08-19：CPTransformer 与可学习核端到端实验

- 研究目的：保持 MATR 逐周期 RUL、landmark 50/100/150/200、90% 目标覆盖率和电芯级数据隔离不变，用 BatteryLife 的 CPTransformer 组件替换 Mamba，并比较四种可学习局部核。
- 代码变更：新增 `battery_lcp/run_cptransformer_kernels.py`、`battery_lcp/analyze_cptransformer_kernels.py` 和根目录设计文档 `CPTRANSFORMER_KERNEL_DESIGN.md`。预测器复用 BatteryLife 的 MLPBlock、位置编码与 Transformer Encoder；增加 causal mask 和逐周期线性 RUL head。
- 数据/划分：每次为 Train 90、Model-validation 12、Calibration 34、Test 33；Calibration/Test 固定，Train/Model-validation 随 seed 在官方训练池内划分。核只由 Train 的 3 折 cell-level OOF 残差学习，Calibration/Test 标签不参与核训练。
- 方法：Split CP；五维状态的 learned diagonal kernel；状态+8维 PCA embedding 的 rank-3 low-rank kernel；残差 0.5/0.75/0.9 条件分位画像核；50-cycle 趋势+去趋势 RFFT 五频带的 spectral kernel。最低 ESS=10，不足时扩带，失败时回退 Split CP。
- 正式配置：seed 42/2021/2024，最终模型最多100 epoch、patience=20；3折 OOF 模型固定40 epoch；kernel 400 steps；d_model=64、d_ff=128、4 heads、2层 Transformer；alpha=0.1。
- 点预测：三 seed 平均 RMSE 在 t=50/100/150/200 分别为 240.09/244.21/233.94/213.18 cycles；平均 MAE 为 123.15/118.40/114.95/105.20；MAPE 为 11.43%/11.34%/11.96%/12.11%。相较旧 Mamba seed42，点预测总体改善，但这不是严格的同 seed 聚合模型比较。
- CP 结果：Split CP 三 seed 在四个时刻均为 30/33=90.91% 覆盖，平均宽度为 737.64/682.14/626.85/559.20。Diagonal kernel 基本退化为全局核，与 Split 几乎相同。Low-rank kernel 波动最大，t=150 三次均欠覆盖。Quantile-profile 在 t=200 三次均保持90.91%覆盖且平均宽度524.73，比 Split 平均缩短34.47 cycles（6.16%），但其余时刻不稳定。Spectral kernel 三次均达到目标覆盖，但平均区间没有稳定缩窄，且部分时刻明显变宽。
- 结论：目前唯一有重复实验支持的正向信号是 t=200 的 quantile-profile kernel；仍不足以声称 learnable LCP 整体优于 Split CP。小 test 集每块电芯对应3.03个百分点，87.88%与90.91%只差1块，必须同时报告 covered/total。
- 异常1：BatteryLife 的 `SelfAttention_Family.py` 无条件导入可选包 `reformer_pytorch`，CPTransformer 本身并不用 Reformer，却导致启动失败。已改为可选导入；仅实际构造 ReformerLayer 时才要求该依赖。
- 异常2：自动终端没有继承交互式 `my_torch` 环境，直接调用 `python` 报 `ModuleNotFoundError: torch`。正式复验统一使用 `D:\Anaconda3\envs\my_torch\python.exe`，不影响已完成结果。
- 验证：对 causal mask 做过未来输入扰动测试，前50周期输出最大变化为0；三个正式运行均产生完整模型、OOF预测、kernel参数、逐电芯区间和诊断文件。
- 输出位置：`battery_lcp/outputs/cptransformer_learned_kernels_seed{42,2021,2024}/`；跨 seed 汇总为 `battery_lcp/outputs/cptransformer_*_summary.csv` 和 `cptransformer_*_runs.csv`。

### 2026-08-19：第一阶段 residual audit

- 新增 `battery_lcp/audit_cptransformer_residuals.py` 和 `RESIDUAL_AUDIT.md`，汇总三次正式运行的 Calibration/Test 残差、寿命分层、相关性、反复难例、Calibration 尾部及 Test 漏覆盖电芯。
- 共804条预测的绝对残差中位数46.14、q90为187.40、q99为962.13；最长寿命四分位平均绝对误差280.94，其余四分位50.71，确认存在强烈的长寿命尾部异方差与系统性低估。
- n_cal=34、alpha=0.1时阈值为第三大残差。Calibration 前两大难例位于阈值以上，Test 则有三块长寿命难例稳定超阈值，解释了30/33覆盖和大区间同时出现的原因。
- Test 分析仅作事后诊断；后续方法训练、选择和超参数调整不得使用这些 Test 标签。
