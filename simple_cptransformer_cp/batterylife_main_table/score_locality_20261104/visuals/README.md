# 逐电芯分布诊断

先看relative_intervals_test61.png，再打开CDF_relative_all61.pdf。
三份PDF分别对应三种分数，每份按原test顺序包含全部61块电芯。

灰=普通split CP，蓝=寿命RLCP，绿=分歧RLCP。
每页三种方法使用同一校准分数、同一测试尺度，只改变权重。
横轴是分数，relative/gap6无量纲；半径=分位分数×测试尺度。
红横线为90%，有限CDF未达到1是因为剩余质量在测试+∞原子上。
若有限分数累计不足90%，分位数就是∞，不能重新归一化有限点来删除它。

与旧MATR图不同：本图保留测试+∞质量，使用RLCP的90%阈值，而非旧basic的32/34。
不要把两批曲线阈值直接混用。实际误差只在标题和区间图作事后诊断，未用于权重、尺度或参数选择。

weighted_CDF_points.csv有每个阶梯的校准ID、原始预测/标签、分数、权重、累计量；
quantile_attribution.csv有半径、过线前后、选中点、尾部质量和ESS。
