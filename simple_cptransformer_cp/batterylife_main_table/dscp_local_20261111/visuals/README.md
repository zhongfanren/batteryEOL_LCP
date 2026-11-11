# DSCP-Local可视化

两份PDF各含全部61块原test，固定encoder seed2024，不挑种子或电芯。
raw_signed横轴为有符号寿命预测残差，红线为5%/95%，取两端构造非对称区间。
abs_relative横轴为绝对误差/预测寿命尺度，红线90%，分位数乘测试尺度得到对称半径。
灰=全局，黄=同种子未训练encoder，绿=多分位残差训练后的encoder；所有cal分数固定，只换权重。
这些图对应论文式softmax经验CDF，没有测试∞原子，不声明有限样本保证。
单独RLCP结果及其∞计数见RESULTS.md和CSV，不与图中确定性权重混用。
右下是训练/验证损失，验证选epoch没有使用cal/test。实际EOL仅在标题作事后检查。
