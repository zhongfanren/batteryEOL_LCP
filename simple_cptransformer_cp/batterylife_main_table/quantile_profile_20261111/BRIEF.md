# 多分位误差画像：第一轮

固定原预测器，只用原train97 OOF与dev20；原cal/test未评价。三个线性分位头估计归一化绝对误差的0.1/0.5/0.9分位。
原手工9维摘要经过参考侧寿命关联扣除、PCA3，加入log预测寿命二次基函数。每个hand头5个输入+截距，三头共18个回归系数，不训练深序列网络。

## 主要对照

kernel_global=全局经验误差分布；kernel_hand=此前固定手工邻域；profile_hand_3q=主要候选；profile_hand_q90=同一画像只使用最后一维；profile_life_3q=不加入手工信息；qr_hand_q90=直接用分位回归输出。
画像参考点采用内层留辅助折拟合：一块参考电芯自己的误差不参与计算它的画像。每个内折对参考/查询用同一个映射，内折CDF按参考块大小混合；不是把不同坐标随意拼接。
独立分位头输出先截断为非负、再逐行排序；单分位画像对照也使用同一排序后的最后一维。核带宽固定为内层拟合点画像距离中位数的一半，原手工对照规则不变，不调参。

## 结果：主指标是90%归一化误差分位的pinball损失，越小越好

|预测器|阶段|方法|pinball|对全局改善|对手工改善|
|---|---|---|---:|---:|---:|
|CPLSTM|dev20|kernel_global|0.0292|+0.0%|-11.1%|
|CPLSTM|dev20|kernel_hand|0.0263|+10.0%|+0.0%|
|CPLSTM|dev20|kernel_life|0.0277|+5.3%|-5.2%|
|CPLSTM|dev20|profile_hand_3q|0.0243|+17.0%|+7.8%|
|CPLSTM|dev20|profile_hand_q90|0.0252|+13.9%|+4.3%|
|CPLSTM|dev20|profile_life_3q|0.0300|-2.4%|-13.8%|
|CPLSTM|dev20|qr_hand_q90|0.0379|-29.6%|-43.9%|
|CPLSTM|oof_cv|kernel_global|0.0596|+0.0%|+3.8%|
|CPLSTM|oof_cv|kernel_hand|0.0620|-3.9%|+0.0%|
|CPLSTM|oof_cv|kernel_life|0.0643|-7.8%|-3.7%|
|CPLSTM|oof_cv|profile_hand_3q|0.0598|-0.3%|+3.5%|
|CPLSTM|oof_cv|profile_hand_q90|0.0542|+9.1%|+12.6%|
|CPLSTM|oof_cv|profile_life_3q|0.0626|-5.0%|-1.1%|
|CPLSTM|oof_cv|qr_hand_q90|0.0518|+13.2%|+16.4%|
|CPMLP|dev20|kernel_global|0.0283|+0.0%|-18.0%|
|CPMLP|dev20|kernel_hand|0.0240|+15.2%|+0.0%|
|CPMLP|dev20|kernel_life|0.0270|+4.7%|-12.4%|
|CPMLP|dev20|profile_hand_3q|0.0291|-2.5%|-21.0%|
|CPMLP|dev20|profile_hand_q90|0.0281|+0.9%|-16.9%|
|CPMLP|dev20|profile_life_3q|0.0293|-3.3%|-21.9%|
|CPMLP|dev20|qr_hand_q90|0.0336|-18.6%|-39.9%|
|CPMLP|oof_cv|kernel_global|0.0580|+0.0%|-5.1%|
|CPMLP|oof_cv|kernel_hand|0.0552|+4.9%|+0.0%|
|CPMLP|oof_cv|kernel_life|0.0662|-14.1%|-19.9%|
|CPMLP|oof_cv|profile_hand_3q|0.0544|+6.2%|+1.4%|
|CPMLP|oof_cv|profile_hand_q90|0.0560|+3.5%|-1.4%|
|CPMLP|oof_cv|profile_life_3q|0.0672|-15.9%|-21.8%|
|CPMLP|oof_cv|qr_hand_q90|0.0620|-6.8%|-12.3%|

## 运行前固定的判断标准

学习几何门槛：OOF较全局与手工均改善至少2%，且各至少3/5折胜；dev20对两者均方向为正、各至少3/5模型视角胜。多分位数额外贡献另比较同模型的q90-only画像。
- CPMLP：学习几何门槛未通过；多个分位数增量检查未通过。
- CPLSTM：学习几何门槛未通过；多个分位数增量检查未通过。

## 边界

本批CDF是经验尾部估计，没有CP有限样本修正或测试∞原子，fraction_below_q不是最终CP覆盖保证。所有查询和负结果保留。
内层画像留出解决的是画像直接拟合自己的参考误差；底层OOF点预测器训练仍重叠，参考预测器可能见过外层查询标签，所以OOF-CV仍不是端到端独立嵌套验证。
dev20未进入辅助模型训练，但历史已用于探索；五个模型视角仍只有20块电芯。参考OOF为多个辅助模型的误差混合，与最终主模型可能不同。
此方法是借鉴Lee等的多分位残差监督思想、使用输出分位画像的低容量适配，不是原文深encoder或其理论保证的直接复现。

图入口：visuals/PROFILE_CDF_all97.pdf；逐电芯画像在profile_rows.csv，全部权重/参考画像/查询画像及模型参数保存在neighbourhoods与models。
