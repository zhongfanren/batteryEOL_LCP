# 训练侧交叉拟合风险学习：自动结果简报

所有辅助训练和风险拟合均在原train97或原dev20侧；OOF误差没有用于最终CP分位数。
主预测器仍为原CPMLP seed2024；CPLSTM seed2024作固定复核。原cal61/test61保持原成员，另外200次固定池重分仅作探索。

## 独立误差样本

每块训练电芯由没有见过它的辅助模型预测；内部验证选epoch后，在外折训练侧从头重训。
得到97块不同电芯的外折误差，并不意味着97个统计独立的拟合残差：不同辅助模型训练集重叠。也不是新获得97块数据。
辅助模型每折约77块训练、最终主模型97块，条件误差分布可能不同；分数最后仍由原61块校准电芯校准。

## 风险诊断

| 预测器 | OOF大误差数/97 | dev大误差数/20 | OOF风险与历史池相对误差rho |
|---|---:|---:|---:|
| CPMLP | 7 | 0 | nan |
| CPLSTM | 9 | 0 | nan |

大误差定义仍为误差>预测寿命50%；rho是风险排序诊断，不是90%条件分位数保证。

风险尺度同时保留Ridge的log误差平均趋势和固定正则化的90%分位数回归（tail）。LCP的learned_risk距离使用后者；估计尾部仍不自动保证条件90%覆盖。
## 冻结分数后的平均重分表现

| 预测器 | 方法 | 覆盖 | 宽度 | Interval score | 无穷比例 |
|---|---|---:|---:|---:|---:|
| CPLSTM | dev20_pair_scale_cp | 90.66% | 1917.2 | 2026.2 | 0.000% |
| CPLSTM | dev20_pair_tail_scale_cp | 90.61% | 1675.5 | 1804.7 | 0.000% |
| CPLSTM | gap6_scale_cp | 90.52% | 1045.2 | 1459.6 | 0.000% |
| CPLSTM | oof_life_scale_cp | 90.57% | 1063.2 | 1484.7 | 0.000% |
| CPLSTM | oof_pair_scale_cp | 90.25% | 1150.9 | 1446.5 | 0.000% |
| CPLSTM | oof_pair_tail_scale_cp | 90.53% | 1069.2 | 1523.6 | 0.000% |
| CPLSTM | oof_tail_rlcp_learned_risk_h3 | 90.53% | 1069.2 | 1523.6 | 0.000% |
| CPLSTM | oof_tail_rlcp_learned_risk_h6 | 90.53% | 1069.2 | 1523.6 | 0.000% |
| CPLSTM | oof_tail_rlcp_life_h3 | 91.00% | inf | inf | 0.008% |
| CPLSTM | oof_tail_rlcp_life_h6 | 90.85% | 1087.9 | 1533.2 | 0.000% |
| CPLSTM | pair_gap_scale_cp | 90.43% | 1073.7 | 1476.3 | 0.000% |
| CPLSTM | prediction_scale_cp | 90.48% | 991.7 | 1467.4 | 0.000% |
| CPLSTM | relative_cp | 90.53% | 1069.2 | 1523.6 | 0.000% |
| CPLSTM | relative_rlcp_learned_risk_h3 | 90.53% | 1069.2 | 1523.6 | 0.000% |
| CPLSTM | relative_rlcp_learned_risk_h6 | 90.53% | 1069.2 | 1523.6 | 0.000% |
| CPLSTM | relative_rlcp_life_h3 | 91.00% | inf | inf | 0.008% |
| CPLSTM | relative_rlcp_life_h6 | 90.85% | 1087.9 | 1533.2 | 0.000% |
| CPLSTM | split_cp | 89.98% | 1083.8 | 2172.9 | 0.000% |
| CPMLP | dev20_pair_scale_cp | 90.61% | 1895.8 | 2021.1 | 0.000% |
| CPMLP | dev20_pair_tail_scale_cp | 90.22% | 1877.9 | 1972.6 | 0.000% |
| CPMLP | gap6_scale_cp | 90.70% | 1035.8 | 1473.0 | 0.000% |
| CPMLP | oof_life_scale_cp | 90.63% | 1099.0 | 1549.0 | 0.000% |
| CPMLP | oof_pair_scale_cp | 90.56% | 1127.5 | 1503.0 | 0.000% |
| CPMLP | oof_pair_tail_scale_cp | 90.67% | 1052.9 | 1545.3 | 0.000% |
| CPMLP | oof_tail_rlcp_learned_risk_h3 | 90.67% | 1052.9 | 1545.3 | 0.000% |
| CPMLP | oof_tail_rlcp_learned_risk_h6 | 90.67% | 1052.9 | 1545.3 | 0.000% |
| CPMLP | oof_tail_rlcp_life_h3 | 91.25% | 1085.0 | 1559.9 | 0.000% |
| CPMLP | oof_tail_rlcp_life_h6 | 91.09% | 1065.7 | 1548.8 | 0.000% |
| CPMLP | pair_gap_scale_cp | 90.53% | 1068.0 | 1487.3 | 0.000% |
| CPMLP | prediction_scale_cp | 90.30% | 852.1 | 1596.2 | 0.000% |
| CPMLP | relative_cp | 90.67% | 1052.9 | 1545.3 | 0.000% |
| CPMLP | relative_rlcp_learned_risk_h3 | 90.67% | 1052.9 | 1545.3 | 0.000% |
| CPMLP | relative_rlcp_learned_risk_h6 | 90.67% | 1052.9 | 1545.3 | 0.000% |
| CPMLP | relative_rlcp_life_h3 | 91.25% | 1085.0 | 1559.9 | 0.000% |
| CPMLP | relative_rlcp_life_h6 | 91.09% | 1065.7 | 1548.8 | 0.000% |
| CPMLP | split_cp | 90.07% | 996.9 | 2218.2 | 0.000% |

## 原61块test（单独保留，不能被重分平均掩盖）

| 预测器 | 方法 | 覆盖 | 宽度 | Interval score |
|---|---|---:|---:|---:|
| CPMLP | relative_cp | 88.52% | 947.4 | 1265.6 |
| CPMLP | gap6_scale_cp | 86.89% | 968.1 | 1217.1 |
| CPMLP | pair_gap_scale_cp | 88.52% | 1010.9 | 1232.7 |
| CPMLP | oof_life_scale_cp | 88.52% | 992.2 | 1271.1 |
| CPMLP | oof_pair_scale_cp | 88.52% | 1053.8 | 1247.6 |
| CPMLP | dev20_pair_scale_cp | 90.16% | 1907.1 | 1976.1 |
| CPMLP | oof_pair_tail_scale_cp | 88.52% | 947.4 | 1265.6 |
| CPMLP | dev20_pair_tail_scale_cp | 91.80% | 1912.4 | 1966.5 |
| CPLSTM | relative_cp | 96.72% | 1203.4 | 1248.0 |
| CPLSTM | gap6_scale_cp | 96.72% | 1131.0 | 1164.6 |
| CPLSTM | pair_gap_scale_cp | 95.08% | 1165.3 | 1218.2 |
| CPLSTM | oof_life_scale_cp | 96.72% | 1210.2 | 1242.3 |
| CPLSTM | oof_pair_scale_cp | 95.08% | 1199.6 | 1253.5 |
| CPLSTM | dev20_pair_scale_cp | 93.44% | 1951.7 | 2032.0 |
| CPLSTM | oof_pair_tail_scale_cp | 96.72% | 1203.4 | 1248.0 |
| CPLSTM | dev20_pair_tail_scale_cp | 93.44% | 1728.4 | 1825.1 |

## 如何归因

- dev20_pair与oof_pair比较：同一低容量风险算法，残差参考样本及预测器训练规模不同，不能把差异全部归因于样本数。
- oof_life与oof_pair比较：相同97块OOF残差、相同寿命基函数，仅增加跨架构分歧。
- relative/gap6/pair_gap是无需残差拟合的强对照；旧prediction_scale保留。
- dev20_pair_tail与oof_pair_tail对比直接90%尾部尺度学习；模型正则强度、特征和目标保持相同。
- RLCP的relative分数对照relative_cp；oof_tail分数对照oof_pair_tail_scale_cp。每个分数下比较life与learned_risk距离，h3/h6全部报告。
- pair_gap分组使用dev可观察单种子模型对分歧；learned_risk分组用dev预测风险阈值，均未用cal/test误差定组。它们与此前六模型分歧组不同。
分组真实误差见group_support.csv，覆盖/宽度/score见group_metrics.csv；高预测风险组更宽本身不是有效证据。
全体无穷保留；历史标签早已查看，跨设置和重分有依赖，均不是新的确认性覆盖或每组90%保证。

已检查四个冻结基线与上一批精确一致、原test逐电芯指标算术一致。完整模型系数、分折身份和来源指纹均已保存。
