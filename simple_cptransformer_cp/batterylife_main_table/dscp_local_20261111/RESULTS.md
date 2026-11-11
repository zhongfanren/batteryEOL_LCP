# DSCP-Local：固定T100横向适配

原CPMLP点预测器保持冻结。LSTM学习残差三个分位数，局部校准使用隐状态而非头输出。所有3随机种子与两种分数全部报告。
raw_signed为论文机制适配，使用有符号残差和经验5%/95%端点；abs_relative为本项目归一化绝对分数适配。
DSCP_*是论文式确定性softmax加权经验CDF，无有限样本覆盖保证声明；RLCP_*另用Gaussian随机中心及测试∞原子。

## 固定池200次重分平均（不是200个独立数据集）

|模式|encoder种子|方法|覆盖|宽度|Interval score|∞比例|
|---|---:|---|---:|---:|---:|---:|
|abs_relative|42|DSCP_relative|90.17%|1012.6|1568.0|0.000%|
|abs_relative|42|RLCP_relative|91.16%|1072.2|1550.5|0.000%|
|abs_relative|42|global_relative_empirical|88.93%|995.0|1540.1|0.000%|
|abs_relative|42|global_signed_empirical|86.44%|929.4|2199.6|0.000%|
|abs_relative|42|global_signed_finite|89.66%|1091.1|2213.1|0.000%|
|abs_relative|42|hand_relative_empirical|89.34%|993.1|1562.8|0.000%|
|abs_relative|42|random_relative|89.73%|1000.4|1547.4|0.000%|
|abs_relative|42|split_abs|90.07%|996.9|2218.2|0.000%|
|abs_relative|42|split_gap6|90.70%|1035.8|1473.0|0.000%|
|abs_relative|42|split_relative|90.67%|1052.9|1545.3|0.000%|
|abs_relative|2021|DSCP_relative|89.89%|1014.6|1550.8|0.000%|
|abs_relative|2021|RLCP_relative|91.13%|1072.6|1551.7|0.000%|
|abs_relative|2021|global_relative_empirical|88.93%|995.0|1540.1|0.000%|
|abs_relative|2021|global_signed_empirical|86.44%|929.4|2199.6|0.000%|
|abs_relative|2021|global_signed_finite|89.66%|1091.1|2213.1|0.000%|
|abs_relative|2021|hand_relative_empirical|89.34%|993.1|1562.8|0.000%|
|abs_relative|2021|random_relative|89.17%|1057.5|1537.7|0.000%|
|abs_relative|2021|split_abs|90.07%|996.9|2218.2|0.000%|
|abs_relative|2021|split_gap6|90.70%|1035.8|1473.0|0.000%|
|abs_relative|2021|split_relative|90.67%|1052.9|1545.3|0.000%|
|abs_relative|2024|DSCP_relative|90.14%|1035.8|1567.9|0.000%|
|abs_relative|2024|RLCP_relative|91.16%|1074.6|1553.4|0.000%|
|abs_relative|2024|global_relative_empirical|88.93%|995.0|1540.1|0.000%|
|abs_relative|2024|global_signed_empirical|86.44%|929.4|2199.6|0.000%|
|abs_relative|2024|global_signed_finite|89.66%|1091.1|2213.1|0.000%|
|abs_relative|2024|hand_relative_empirical|89.34%|993.1|1562.8|0.000%|
|abs_relative|2024|random_relative|89.77%|1052.0|1542.2|0.000%|
|abs_relative|2024|split_abs|90.07%|996.9|2218.2|0.000%|
|abs_relative|2024|split_gap6|90.70%|1035.8|1473.0|0.000%|
|abs_relative|2024|split_relative|90.67%|1052.9|1545.3|0.000%|
|raw_signed|42|DSCP_raw_absolute|92.16%|973.6|1782.5|0.000%|
|raw_signed|42|DSCP_raw_signed|89.95%|952.5|1920.9|0.000%|
|raw_signed|42|RLCP_raw_absolute|90.69%|1049.9|2201.0|0.000%|
|raw_signed|42|global_relative_empirical|88.93%|995.0|1540.1|0.000%|
|raw_signed|42|global_signed_empirical|86.44%|929.4|2199.6|0.000%|
|raw_signed|42|global_signed_finite|89.66%|1091.1|2213.1|0.000%|
|raw_signed|42|hand_relative_empirical|89.34%|993.1|1562.8|0.000%|
|raw_signed|42|random_raw_signed|90.07%|937.4|1919.8|0.000%|
|raw_signed|42|split_abs|90.07%|996.9|2218.2|0.000%|
|raw_signed|42|split_gap6|90.70%|1035.8|1473.0|0.000%|
|raw_signed|42|split_relative|90.67%|1052.9|1545.3|0.000%|
|raw_signed|2021|DSCP_raw_absolute|92.33%|1003.1|1848.2|0.000%|
|raw_signed|2021|DSCP_raw_signed|90.46%|975.7|1959.9|0.000%|
|raw_signed|2021|RLCP_raw_absolute|90.63%|1048.7|2206.6|0.000%|
|raw_signed|2021|global_relative_empirical|88.93%|995.0|1540.1|0.000%|
|raw_signed|2021|global_signed_empirical|86.44%|929.4|2199.6|0.000%|
|raw_signed|2021|global_signed_finite|89.66%|1091.1|2213.1|0.000%|
|raw_signed|2021|hand_relative_empirical|89.34%|993.1|1562.8|0.000%|
|raw_signed|2021|random_raw_signed|89.44%|969.5|1711.3|0.000%|
|raw_signed|2021|split_abs|90.07%|996.9|2218.2|0.000%|
|raw_signed|2021|split_gap6|90.70%|1035.8|1473.0|0.000%|
|raw_signed|2021|split_relative|90.67%|1052.9|1545.3|0.000%|
|raw_signed|2024|DSCP_raw_absolute|92.36%|922.9|1706.9|0.000%|
|raw_signed|2024|DSCP_raw_signed|90.43%|961.4|1825.6|0.000%|
|raw_signed|2024|RLCP_raw_absolute|90.80%|1054.3|2199.4|0.000%|
|raw_signed|2024|global_relative_empirical|88.93%|995.0|1540.1|0.000%|
|raw_signed|2024|global_signed_empirical|86.44%|929.4|2199.6|0.000%|
|raw_signed|2024|global_signed_finite|89.66%|1091.1|2213.1|0.000%|
|raw_signed|2024|hand_relative_empirical|89.34%|993.1|1562.8|0.000%|
|raw_signed|2024|random_raw_signed|89.57%|918.5|1893.5|0.000%|
|raw_signed|2024|split_abs|90.07%|996.9|2218.2|0.000%|
|raw_signed|2024|split_gap6|90.70%|1035.8|1473.0|0.000%|
|raw_signed|2024|split_relative|90.67%|1052.9|1545.3|0.000%|

## 解释范围

原始test61单独保存在original_intervals.csv。相同分数下比较global、random、trained；与split_relative/gap6强基线同时比较，不能仅看网络训练损失或挑最好种子。
原文的时序上下文含历史可观测残差，本任务没有同电芯历史EOL残差，因此用前100周期曲线代替；这是方法适配，不是原文数据集完整复现。
train97监督来自外折辅助模型的OOF误差，主模型训练规模和条件误差可能不同。dev20只用于选encoder epoch，cal/test不参与训练或温度选择；历史标签已查看，整体仍为探索。
有符号和对称区间分开列，经验CDF与有限样本校正分开列；不能把分数/区间形状变化全部算为学习距离收益。
输入标准化仅train97，温度仅train隐表示。图保留seed2024两模式全部test61。原始cal/test、模型与旧代码未修改。
论文：https://arxiv.org/abs/2610.02357。详细方法差异与配置见plan.json。
