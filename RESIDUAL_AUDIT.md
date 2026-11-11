# CPTransformer residual audit

日期：2026-08-19

## 范围与使用边界

本报告汇总 seed 42、2021、2024 的现有 Calibration/Test 预测，只用于解释误差来源。Test 标签属于事后诊断信息，不得据此调参后再把同一 Test 结果当作无偏最终评价。

## 主要发现

- 共检查 67 块 Calibration/Test 电芯、4 个 landmark、3 个 seed，合计 804 条预测。
- 绝对残差中位数为 46.14 cycles，90% 分位为 187.40，但99%分位达到962.13，误差是明显的重尾分布。
- 最长寿命四分位的平均绝对残差为280.94，其余三个四分位为50.71，相差5.54倍。
- 绝对残差与真实总寿命的 Spearman 相关系数在各时刻约为0.43–0.56；现有状态量与误差的相关性更弱，容量保持率最高约0.40，内阻接近0。
- 最难电芯在三个 seed 和四个时刻重复出现，并非偶然训练波动。误差符号几乎始终为正，即 `true_rul - pred_rul > 0`，表现为系统性低估长寿命电芯。

## Split CP 宽度的直接来源

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

## 对当前 LCP 的解释

Diagonal-state LCP 的五个状态特征没有充分识别长寿命难例，学习结果趋向较大 ESS 的近全局核，所以常退化成 Split CP。Low-rank joint LCP 的平均 ESS 约14–19，在90个 OOF 电芯和34个 Calibration 电芯下方差过大。Spectral-band LCP 可以为部分长寿命电芯产生更宽区间并提高覆盖，但当前频域特征区分力不足，导致大量非难例也被扩宽。Quantile-profile LCP 在 t=200 出现稳定缩窄信号，说明直接学习“条件残差分布”比仅学习输入几何距离更符合当前误差结构。

## 下一阶段判断

首要矛盾是长寿命电芯的条件误差尺度，而不是普通样本的平均误差。因此下一步优先级应为：

1. 使用 Train OOF 残差训练误差尺度模型，实施 Normalized CP；
2. 训练上下条件分位数，实施 CQR；
3. 将 normalized score 或 CQR score 与保守的 LCP 权重组合；
4. 所有方法选择只使用 Train/Model-validation/Calibration，Test 保持冻结；
5. 完成后再在第二数据集做外部验证。

详细数据位于 `battery_lcp/outputs/residual_audit/`。
