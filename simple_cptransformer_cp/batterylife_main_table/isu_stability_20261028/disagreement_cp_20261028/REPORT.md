# 分歧用于尺度还是LCP距离：冻结预测试验

本轮没有重新训练点预测器，单模型中心保持不变。
本研究方向来自历史结果，以下仍为探索性；本轮不使用此前在历史cal/test标签上拟合的风险模型。
方法公式不拟合cal/test标签，校准标签只计算当前分位数；历史标签暴露意味着这些不是新的确认性覆盖证据。

## 固定构造

p=max(1,当前预测寿命)，d=|三个CPMLP均值−三个CPLSTM均值|。
relative_cp尺度=p；gap_scale_cp尺度=p+d，系数预设为1，不调参；prediction_scale_cp沿用20块dev拟合的旧寿命尺度对照。
RLCP距离：log(p)一维，或[log(p),log(1+d/p)]二维，均只在原train97上拟合标准化。
每一种校准分数都分别配全局校准、仅寿命RLCP、寿命＋分歧RLCP；高斯随机中心，测试权重位于+∞，h=3和6全报告。
RLCP随机第一坐标在一维/二维对照中共用；尺度变化对照共用完整局部权重。不能将局部化与换评分混为一项效果。
每设置固定122块历史池，原61/61及200次随机重分；重分不产生新独立电芯。主设置仍是CPMLP seed2024、原划分。

## 主设置：原61块test

| 方法 | 覆盖 | 宽度/周期 | Interval score | 无穷数 |
|---|---:|---:|---:|---:|
| split_cp | 93.4% | 1023.5 | 1854.5 | 0 |
| relative_cp | 88.5% | 947.4 | 1265.6 | 0 |
| gap_scale_cp | 86.9% | 968.1 | 1217.1 | 0 |
| prediction_scale_cp | 93.4% | 903.5 | 1265.7 | 0 |
| relative_rlcp_life_h3 | 90.2% | 993.4 | 1285.4 | 0 |
| gap_scale_rlcp_life_h3 | 90.2% | 994.9 | 1223.4 | 0 |
| relative_rlcp_life_h6 | 88.5% | 960.5 | 1261.2 | 0 |
| gap_scale_rlcp_life_h6 | 90.2% | 975.5 | 1209.4 | 0 |
| relative_rlcp_life_gap_h3 | 86.9% | inf | inf | 1 |
| gap_scale_rlcp_life_gap_h3 | 88.5% | inf | inf | 1 |
| relative_rlcp_life_gap_h6 | 88.5% | 1004.0 | 1219.2 | 0 |
| gap_scale_rlcp_life_gap_h6 | 88.5% | 997.6 | 1187.9 | 0 |

## 全部设置与重分

均先在每个设置内平均200次，再对各模型9个设置等权平均。任何无穷值都保留；不计算‘只保留有限区间’的效率结论。

| 模型 | 方法 | 覆盖 | 宽度 | Interval score | 无穷电芯比例 |
|---|---|---:|---:|---:|---:|
| CPLSTM | gap_scale_cp | 90.05% | 1127.6 | 1649.7 | 0.000% |
| CPLSTM | gap_scale_rlcp_life_gap_h3 | 90.84% | inf | inf | 0.362% |
| CPLSTM | gap_scale_rlcp_life_gap_h6 | 90.59% | inf | inf | 0.003% |
| CPLSTM | gap_scale_rlcp_life_h3 | 90.53% | inf | inf | 0.002% |
| CPLSTM | gap_scale_rlcp_life_h6 | 90.35% | 1144.2 | 1654.2 | 0.000% |
| CPLSTM | prediction_scale_cp | 90.04% | 1131.7 | 1963.5 | 0.000% |
| CPLSTM | relative_cp | 90.00% | 1169.0 | 1735.3 | 0.000% |
| CPLSTM | relative_rlcp_life_gap_h3 | 90.89% | inf | inf | 0.362% |
| CPLSTM | relative_rlcp_life_gap_h6 | 90.64% | inf | inf | 0.003% |
| CPLSTM | relative_rlcp_life_h3 | 90.55% | inf | inf | 0.002% |
| CPLSTM | relative_rlcp_life_h6 | 90.38% | 1191.6 | 1740.3 | 0.000% |
| CPLSTM | split_cp | 90.05% | 1263.1 | 2359.3 | 0.000% |
| CPMLP | gap_scale_cp | 90.32% | 1101.9 | 1512.4 | 0.000% |
| CPMLP | gap_scale_rlcp_life_gap_h3 | 91.01% | inf | inf | 0.343% |
| CPMLP | gap_scale_rlcp_life_gap_h6 | 90.81% | inf | inf | 0.012% |
| CPMLP | gap_scale_rlcp_life_h3 | 90.78% | 1127.5 | 1524.8 | 0.000% |
| CPMLP | gap_scale_rlcp_life_h6 | 90.63% | 1114.4 | 1516.1 | 0.000% |
| CPMLP | prediction_scale_cp | 90.18% | 1021.9 | 1778.9 | 0.000% |
| CPMLP | relative_cp | 90.27% | 1117.4 | 1555.3 | 0.000% |
| CPMLP | relative_rlcp_life_gap_h3 | 91.08% | inf | inf | 0.343% |
| CPMLP | relative_rlcp_life_gap_h6 | 90.85% | inf | inf | 0.012% |
| CPMLP | relative_rlcp_life_h3 | 90.77% | 1146.9 | 1570.5 | 0.000% |
| CPMLP | relative_rlcp_life_h6 | 90.59% | 1131.0 | 1559.9 | 0.000% |
| CPMLP | split_cp | 90.18% | 1179.9 | 2171.9 | 0.000% |

## 分组解释

gap组阈值是20块dev的d/p三等分位点；life组是dev预测寿命三等分位点，均不使用误差。
高分歧组只是可提前观察的分歧分组，不能先验称为真实高难度；支持数据见group_support.csv中的实际误差与大误差比例。
group_metrics.csv保存每组覆盖、宽度与score；presentations为重分中的累计出现次数，不是独立样本量。
retrospective_large_error组用真实相对误差>50%定义，只做事后漏覆盖诊断，不是可部署分组，也没有该组90%覆盖保证。

具体归因对照见paired_by_job.csv与aggregate.csv：gap_scale_cp应与relative_cp比较；RLCP应与同score的全局方法比较。
只改善原test、某带宽或某个模型不算稳定收益；先报告全体正负结果，再决定是否值得获取更多诚实风险监督。
