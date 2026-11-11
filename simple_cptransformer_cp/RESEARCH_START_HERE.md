# 从这里继续研究

更新：2026-11-11。这个文件是当前入口，历史记录继续保留。

仓库整理与 GitHub 同步范围：[仓库精简复核](../REPOSITORY_REVIEW.md)；代码、计划和结论同步，完整数据与产物保留本地。

2026-11-08文献补充：[Lee等深序列CP阅读笔记](literature_notes/lee_2026_sequence_cp/READING_NOTE.md)。2026-11-11已完成低容量输出分位画像适配，未稳定超过手工邻域；不是原文深encoder复现。

## 我们现在研究什么

固定前100周期，预测新电芯的寿命区间。核心问题是：扣除简单寿命尺度造成的误差差异以后，早期信息能否识别剩余的尾部风险，使局部校准提供额外价值？

以电芯为单位，不研究同一电芯随运行时间更新区间。当前在ISU-ILCC开发，MATR后续跨数据集复核；不把换数据集、换预测器和换区间方法混成一次收益。

## 当前只需打开三个入口

1. 最新结论：[DSCP-Local横向适配复核](batterylife_main_table/dscp_local_20261111/REVIEW.md)。用户优先试新论文，6个LSTM编码器、区间评测、CDF及独立核查均完成；归一化版略窄但综合分数未胜尺度基线，也未胜随机encoder。前序：[线性多分位画像](batterylife_main_table/quantile_profile_20261111/REVIEW.md)、[手工邻域线索](batterylife_main_table/curve_signal_20261104_v2/REVIEW.md)。
2. 最新图：[DSCP有符号残差CDF](batterylife_main_table/dscp_local_20261111/visuals/CDF_raw_signed_all61.pdf)；[归一化适配CDF](batterylife_main_table/dscp_local_20261111/visuals/CDF_abs_relative_all61.pdf)。论文式softmax图不含测试∞原子；另列RLCP见该批CSV。此前[训练侧画像CDF](batterylife_main_table/quantile_profile_20261111/visuals/PROFILE_CDF_all97.pdf)仅为信息初筛，不混同最终区间。
3. 实验总账：[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)。只有改变判断的结论进入总账，细节留在批次目录。

术语与数据划分：[LCP_EXPERIMENT_SPEC.md](batterylife_main_table/LCP_EXPERIMENT_SPEC.md)。跨聊天交接：[CONTEXT_HANDOFF.md](CONTEXT_HANDOFF.md)。不用每次重读全部历史。

## 每轮推进流程

1. **先写问题。** 每轮最多两个假设，明确分数、尺度、局部特征、校准构造、主对照和停止条件。分位数规则与覆盖校正来自采用的有效框架，不能按结果临时缩区间。
2. **先查信息。** 新学习分数/距离只用建模侧训练及诚实OOF信息开发。检查常数退化、尾部监督数量和跨折表现；非零系数、相关性好看都不自动代表有效。
3. **冻结再评价。** 写plan.json和源码/输入指纹后运行。分数变更先与全局CP比；局部性变更始终与同分数的全局CP比。保留普通CP、简单寿命尺度和六模型分歧尺度强对照。
4. **同时看成本和覆盖。** 保存覆盖、宽度、interval score、预定可观察组、原test与所有重复设置。无穷不删除。重复划分不是新独立样本，历史test不是重新封存后的新测试。
5. **固定诊断包。** 保存逐电芯预测/标签/尺度/上下界；校准预测与分数；每个query的权重、随机量、测试∞质量；加权CDF全点、过线位置、尾部权重和ESS。PDF固定包含所有主test，不按结果筛图。
6. **一页决策。** 每轮说明支持什么、不支持什么、收益来自分数还是局部性。未显示额外信息时停止该分支，禁止只为制造非恒定区间而继续加复杂度。下一轮计划另建目录，不覆盖旧结果。

用户把握问题、贡献目标与资源范围；助手负责具体实现、核查、可视化和留档。短CPU实验可当轮完成；长训练用可见终端，启动核验后不持续监听。

## 构造的边界

Guan的校正LCP、Hore–Barber的RLCP和旧basic加权经验分位数必须区分。当前新批次使用已有Gaussian RLCP实现，并保留测试+∞原子。旧MATR的32/34图是历史机制诊断，不是新批次的计算规则。

## 仓库整理原则

目前历史文件没有阻碍计算，先通过入口和批次索引降低认知负担。保留原始数据、冻结检查点、分折、OOF、计划、负结果和旧CDF诊断。新工作一律写新批次目录；本轮没有删除文件。可再生缓存/临时文件只有确认依赖和回收价值后才清理，不因旧结果不好而删证据。
