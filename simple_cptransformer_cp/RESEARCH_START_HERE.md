# 当前研究

更新：2026-11-11。

固定电芯前 100 周期，预测新电芯的寿命区间。研究问题是：控制简单寿命尺度以后，早期信息能否识别剩余尾部风险，为局部校准提供额外价值？当前在 ISU-ILCC 开发，MATR 用于后续跨数据集复核。

## 最新判断

[DSCP-Local 复核](batterylife_main_table/dscp_local_20261111/REVIEW.md)：6 个 LSTM 编码器、3 个种子的区间评测与核查完成。归一化版略窄，interval score 仍落后简单尺度基线和随机编码器。后续先查开发集选择稳定性，以及辅助 OOF 与主预测器的误差差异。

前序证据：[多分位画像](batterylife_main_table/quantile_profile_20261111/REVIEW.md)、[手工曲线邻域](batterylife_main_table/curve_signal_20261104_v2/REVIEW.md)。训练侧画像用于信息初筛，区间评测单独报告。

## 阅读与执行

- [实验总账](EXPERIMENT_INDEX.md)：各批次的问题、结果和结论入口。
- [方法与数据定义](batterylife_main_table/LCP_EXPERIMENT_SPEC.md)：分数、局部性、总体与角色划分。
- [研究代码指南](README.md)：计算、核查、绘图和历史资料。
- [科研执行约定](../AGENTS.md)：假设、冻结、对照与汇报规则。
- [仓库目录与同步范围](../README.md#目录与同步范围)。

本地最新图位于 `batterylife_main_table/dscp_local_20261111/visuals/`，包含全部 61 块原测试电芯；多分位画像图位于 `batterylife_main_table/quantile_profile_20261111/visuals/`。

文献：[Lee 等深序列 CP 阅读笔记](literature_notes/lee_2026_sequence_cp/READING_NOTE.md)。
