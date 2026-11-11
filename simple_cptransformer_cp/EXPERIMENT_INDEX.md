# 当前实验总账

更新：2026-11-11。只追踪与当前问题有关的结论，不替代原始实验计划和结果。

|批次|问题|状态与结论|入口|
|---|---|---|---|
|20261111 仓库整理|简化文档与协作入口|工作约定、运行指南和历史材料归并；命令采用相对路径与当前 Python 环境|[仓库首页](../README.md)|
|MATR 历史局部化|核/学习相似性是否帮助区间？|有部分固定核收益，但学习核额外收益小且不稳定；旧basic构造不可混同当前RLCP|RESEARCH_PROGRESS_SUMMARY_20260930.md|
|ISU 18冻结设置稳定性|简单尺度是否有效？|寿命尺度收益重现；曲线局部性未显示稳定额外收益|batterylife_main_table/isu_stability_20261028/analysis_review_20261028/ANALYSIS.md|
|ISU 分歧尺度消融|收益来自尺度还是距离？|六模型分歧尺度有小幅收益；局部校准成本/∞仍有问题|batterylife_main_table/isu_stability_20261028/disagreement_cp_20261028/BRIEF.md|
|ISU OOF97|补充诚实风险监督能否学习尾部？|10辅助任务完成；尾部模型常数退化，平均风险模型未稳定胜强基线|batterylife_main_table/isu_crossfit_risk_20261028/analysis_review/ANALYSIS.md|
|MATR CDF诊断|小实际误差为什么也宽？|尾部权重与校准误差大空档共同放大半径；只解释机制，不证明带宽错误|interval_views_20261028/MATR_fixed_RBF_CDF/README.md|
|20261104 分数×局部性 第一轮|放松正则是否解决尾部；单独按分歧局部化是否有增量？|计算18/18完成：尾部非恒定但训练侧CV未改善；分歧局部化轻微改善高组覆盖但出现∞，未胜强尺度对照|batterylife_main_table/score_locality_20261104/BRIEF.md|
|20261104 曲线信息初筛|手工、冻结表示、直接曲线邻域能否改善归一化误差尾部估计？|train97+dev20完成，未评价cal/test；CPMLP手工邻域较全局OOF pinball改善4.9%，dev同向；CPLSTM OOF反向，尚未稳定。复杂表示和分位数回归不优先|batterylife_main_table/curve_signal_20261104_v2/REVIEW.md|
|20261111 多分位画像|学习0.1/0.5/0.9分位画像是否比手工邻域或单q90画像更好？|train97+dev20完成；主要候选CPMLP对手工OOF+1.4%而dev−21.0%，CPLSTM对全局OOF−0.3%，均未过门槛。单q90在CPLSTM有副线索但未跨预测器复现|batterylife_main_table/quantile_profile_20261111/REVIEW.md|
|20261111 DSCP-Local序列编码器|按用户要求提前试新论文多分位监督隐表示；raw signed及归一化适配|6编码器/3种子完成，CPMLP点预测器冻结；归一化版宽1021 vs尺度1053，但score1562 vs1545稍差，随机表示1542更好；核心机制已实现，未证学习收益；softmax与RLCP分列|batterylife_main_table/dscp_local_20261111/REVIEW.md|

下一步先查 DSCP 开发集选择稳定性和 OOF 到主预测器的误差差异，再依据训练侧证据推进手工邻域消融。批次文件与运行命令见[研究代码指南](README.md)。
