# 分数 × 局部性：第一轮

本轮不训练点预测器，不改变数据集或划分。三种分数 × 三种校准构造全部报告，不按历史test选冠军。

## 训练侧尾部模型诊断

主要候选固定为pair alpha=.01；alpha=.05复现退化，alpha=0只检查去正则后的敏感性。
门槛：相对常数及同alpha寿命only的CV pinball均改善至少2%，各至少3/5折获胜，至少4折非恒定，全量预测std>.01。
本诊断复用固定OOF库，辅助模型训练集重叠，不是端到端嵌套验证。无论门槛结果，本批均不追加学习距离，不按历史test调参。

- CPMLP：进入下一轮学习尾部的门槛未通过；对常数pinball改善-6.9%，对寿命only改善-0.8%。
- CPLSTM：进入下一轮学习尾部的门槛未通过；对常数pinball改善-0.3%，对寿命only改善-1.3%。

## 所有18个冻结预测器设置，历史池200次重分平均

|预测器|分数__局部性|覆盖|宽度|Interval score|无穷比例|
|---|---|---:|---:|---:|---:|
|CPLSTM|absolute__gap|90.55%|inf|inf|0.002%|
|CPLSTM|absolute__global|90.05%|1263.1|2359.3|0.000%|
|CPLSTM|absolute__life|90.54%|1303.9|2340.2|0.000%|
|CPLSTM|gap6__gap|90.55%|inf|inf|0.002%|
|CPLSTM|gap6__global|90.05%|1127.6|1649.7|0.000%|
|CPLSTM|gap6__life|90.35%|1144.2|1654.2|0.000%|
|CPLSTM|relative__gap|90.58%|inf|inf|0.002%|
|CPLSTM|relative__global|90.00%|1169.0|1735.3|0.000%|
|CPLSTM|relative__life|90.38%|1191.6|1740.3|0.000%|
|CPMLP|absolute__gap|90.60%|inf|inf|0.011%|
|CPMLP|absolute__global|90.18%|1179.9|2171.9|0.000%|
|CPMLP|absolute__life|90.62%|1214.1|2155.1|0.000%|
|CPMLP|gap6__gap|90.74%|inf|inf|0.011%|
|CPMLP|gap6__global|90.32%|1101.9|1512.4|0.000%|
|CPMLP|gap6__life|90.63%|1114.4|1516.1|0.000%|
|CPMLP|relative__gap|90.76%|inf|inf|0.011%|
|CPMLP|relative__global|90.27%|1117.4|1555.3|0.000%|
|CPMLP|relative__life|90.59%|1131.0|1559.9|0.000%|

absolute=绝对误差；relative=误差/预测寿命尺度；gap6=误差/(预测寿命尺度+六模型架构均值分歧)。
global=普通split CP；life=只按log预测寿命局部化；gap=只按log1p(六模型分歧/预测寿命尺度)局部化。两个坐标均仅用train97标准化；h=6固定。
局部构造沿用Hore–Barber Gaussian RLCP，随机中心与测试+∞原子保留。不是旧MATR32/34阈值或Guan校正LCP的直接实现。

## 怎样阅读结果

先比较relative__global与gap6__global：这是分数/尺度增量；再逐分数比较life/gap与global：这是局部校准增量。
same_score_pairs.csv保存逐冻结设置的配对增益；group_metrics.csv保存按dev可观察变量三等分的组覆盖与成本。高分歧组不预先等同真实高难度，需查看真实误差。
original_intervals.csv完整保留原test，不能用重分均值掩盖。所有无穷保留。历史test已用于探索，200次重分、18设置均有重叠，不是独立新数据或每组90%保证。

## CDF诊断

primary_intermediates.npz实际保存全部校准预测、分数、query权重、测试+∞质量及随机中心噪声，不需反演。
绘图程序plot_score_locality.py输出主CPMLP2024全部61块原test的三份逐分数PDF、逐电芯区间图、CDF全点与归因CSV。CDF横轴是该分数，不一定是周期；纵轴含测试+∞质量的归一化，红线90%。

参考：[Guan LCP](https://arxiv.org/html/2106.08460v2)；[Hore–Barber RLCP](https://arxiv.org/html/2310.07850)；[Lei等 normalized CP](https://arxiv.org/html/1604.04173#S5.SS2)。
