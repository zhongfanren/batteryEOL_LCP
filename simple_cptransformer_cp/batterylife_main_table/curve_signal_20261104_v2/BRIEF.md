# 曲线信息检查：先看这里

固定前100周期，只使用原train97和dev20。未训练任何预测器，未评价原cal/test。
分折冻结模型各自提取表示，不拼接不同模型的embedding坐标。归一化误差=绝对误差/max(1,预测寿命)。

## 检查什么

hand：归一化输入三通道的前10周期均值、后10与前10差、逐周期变化RMS，共9维；不是物理SoH。
embedding：对应分折冻结预测器的内部表示，验证电芯未参与该表示模型训练。
curve：复用旧57维时间曲线特征；直接距离按电压/电流/容量趋势、标准差、差分标准差、斜率六个特征块等权。六组是特征组，不是电芯类别。
各附加信息先在参考侧扣除log寿命二次项能拟合的关联；PCA3和标准化也只拟合参考侧。这是建模控制，不是条件独立证明。

## 两种检查

qr：固定alpha=.01的90%分位数回归，life-only与增加三维特征对照；预测目标直接为归一化误差，不是log误差。
kernel：直接用参考误差构造CDF，全局、仅寿命及寿命+特征分别比较；RBF带宽固定为参考两两距离中位数的一半，无监督选择。
这里是经验90%尾部估计诊断，没有加入CP有限样本修正或测试∞原子；fraction_below_q不是最终CP覆盖率，不能把这些CDF当成有保证的预测区间。

## 结果：pinball越低越好，增益与同路线life-only比较

|模型|证据|方法|pinball|相对寿命增益|胜出折/视角|
|---|---|---|---:|---:|---:|
|CPLSTM|dev20|kernel_curve|0.0251|+9.5%|5/5|
|CPLSTM|dev20|kernel_embedding|0.0292|-5.5%|1/5|
|CPLSTM|dev20|kernel_global|0.0292|-5.6%|1/5|
|CPLSTM|dev20|kernel_hand|0.0263|+4.9%|3/5|
|CPLSTM|dev20|kernel_life|0.0277|+0.0%|0/5|
|CPLSTM|dev20|qr_curve|0.0363|-19.6%|1/5|
|CPLSTM|dev20|qr_embedding|0.0341|-12.5%|2/5|
|CPLSTM|dev20|qr_hand|0.0379|-25.0%|1/5|
|CPLSTM|dev20|qr_life|0.0303|+0.0%|0/5|
|CPLSTM|oof_cv|kernel_curve|0.0626|+2.6%|4/5|
|CPLSTM|oof_cv|kernel_embedding|0.0600|+6.6%|3/5|
|CPLSTM|oof_cv|kernel_global|0.0596|+7.2%|3/5|
|CPLSTM|oof_cv|kernel_hand|0.0620|+3.6%|4/5|
|CPLSTM|oof_cv|kernel_life|0.0643|+0.0%|0/5|
|CPLSTM|oof_cv|qr_curve|0.0528|+9.8%|2/5|
|CPLSTM|oof_cv|qr_embedding|0.0537|+8.3%|4/5|
|CPLSTM|oof_cv|qr_hand|0.0518|+11.5%|2/5|
|CPLSTM|oof_cv|qr_life|0.0585|+0.0%|0/5|
|CPMLP|dev20|kernel_curve|0.0230|+14.9%|5/5|
|CPMLP|dev20|kernel_embedding|0.0260|+3.6%|4/5|
|CPMLP|dev20|kernel_global|0.0283|-4.9%|0/5|
|CPMLP|dev20|kernel_hand|0.0240|+11.1%|5/5|
|CPMLP|dev20|kernel_life|0.0270|+0.0%|0/5|
|CPMLP|dev20|qr_curve|0.0371|-32.0%|0/5|
|CPMLP|dev20|qr_embedding|0.0333|-18.8%|3/5|
|CPMLP|dev20|qr_hand|0.0336|-19.7%|1/5|
|CPMLP|dev20|qr_life|0.0281|+0.0%|0/5|
|CPMLP|oof_cv|kernel_curve|0.0578|+12.7%|5/5|
|CPMLP|oof_cv|kernel_embedding|0.0640|+3.3%|2/5|
|CPMLP|oof_cv|kernel_global|0.0580|+12.3%|5/5|
|CPMLP|oof_cv|kernel_hand|0.0552|+16.6%|5/5|
|CPMLP|oof_cv|kernel_life|0.0662|+0.0%|0/5|
|CPMLP|oof_cv|qr_curve|0.0603|+2.1%|1/5|
|CPMLP|oof_cv|qr_embedding|0.0578|+6.1%|3/5|
|CPMLP|oof_cv|qr_hand|0.0620|-0.6%|1/5|
|CPMLP|oof_cv|qr_life|0.0616|+0.0%|0/5|

## 预定门槛

OOF-CV较匹配寿命对照改善至少2%且至少3/5折胜；dev20方向为正且至少3/5模型视角胜，才列为下一步候选。所有方法报告，不在dev结果后改特征或带宽。
- CPMLP：kernel_hand、kernel_curve通过初筛门槛。
- CPLSTM：kernel_hand、kernel_curve通过初筛门槛。

## 证据边界

OOF-CV每个验证折使用未训练过该折电芯的冻结encoder，避免直接使用主模型训练集表示；但参考OOF残差来自其他辅助模型，这些模型训练集有重叠、可能见过当前验证折标签，所以不是端到端嵌套独立验证。
dev20没有进入任何辅助模型训练/epoch选择，但历史上已看过；同20块由5个模型重复预测，不是100块新电芯。参考误差来自5个辅助预测器的混合，最终主预测器的条件误差分布仍可能不同。
如果出现候选，只说明值得更严格复核及后续同分数CP消融，不证明已经提升最终LCP或具有条件覆盖保证。

## 可视化

visuals/OOF_CDF_all97.pdf：主CPMLP所有97块训练侧OOF查询的CDF，比较全局、寿命邻域、曲线/手工/embedding邻域。
邻域逐点权重和CDF、选中分位点、ESS、逐电芯误差与q均保留；标题明确这不是最终CP。没有按好坏筛选电芯。
