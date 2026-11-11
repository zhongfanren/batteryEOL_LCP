# CPTransformer + Learnable Kernel 实验设计

版本：2026-08-19

## 研究问题

在相同MATR电芯划分、相同landmark和相同90%目标覆盖率下，用BatteryLife原生CPTransformer组件替换Mamba预测器，并比较不同learnable kernel能否改善Split CP的coverage-width权衡。

## 数据隔离

- Train 90：训练最终预测器，并通过3折cell-level OOF生成核学习残差；
- Model-validation 12：最终预测器早停；
- Calibration 34：kernel冻结后才用于conformal calibration；
- Test 33：最终评价；
- landmark固定为50、100、150、200周期。

任何kernel均不得使用Calibration/Test标签训练。OOF的同一电芯所有landmark始终属于同一折。

## CPTransformer适配

复用BatteryLife的`MLPBlock`、`PositionalEmbedding`、`Encoder`、`EncoderLayer`、`AttentionLayer`和`FullAttention`。

改动理由：原模型将固定early-cycle序列flatten后预测总寿命，不能直接执行逐周期RUL；此外逐时刻预测必须阻止attention读取未来循环。

适配结构：

```text
31维逐循环统计特征
  -> Linear + BatteryLife MLPBlock
  -> PositionalEmbedding
  -> BatteryLife Transformer Encoder（causal mask）
  -> 每个循环位置的Linear RUL head
```

- 仅读取到最大landmark 200周期；
- 输出使用无界线性头，推理时只将负RUL截断为0；
- 不使用Mamba实验中受Train最大寿命限制的Sigmoid输出；
- L1损失按电芯平均，长寿命电芯不会因监督点更多而获得更大权重。

## 核学习数据

Train 90块进行3折OOF。每个OOF模型固定训练40轮，不使用被预测折的标签早停。得到：

\[
\{z_{i,t},\ |e^{OOF}_{i,t}|\},\quad t\in\{50,100,150,200\}.
\]

最终CPTransformer仍使用Train 90训练、Model-validation 12早停；所有区间方法共享该最终模型的Calibration/Test预测。

## 对照方法

1. **Split CP**：每个landmark使用34个绝对校准残差的有限样本分位数。
2. **Diagonal-State Kernel**：在容量保持率、容量斜率、容量波动、内阻、温度上学习非负对角Mahalanobis权重。
3. **Low-rank Joint Kernel**：拼接State与Train拟合的8维PCA embedding，学习3维低秩投影。
4. **Quantile-Profile Kernel**：从Joint表示学习绝对残差的0.5/0.75/0.9条件分位数，以学习到的分位数轮廓作为局部表示，再学习非负距离权重。
5. **Spectral-Band Kernel**：最近50周期的容量、内阻、温度轨迹先分离线性趋势，对去趋势残差做RFFT并汇总5个频带能量，再学习非负频带/趋势权重。

所有metric使用leave-one-cell-out局部残差预测目标训练，并加入最低ESS=10的惩罚。kernel训练完成后参数冻结。

## LCP安全规则

- 使用learned metric与learned global bandwidth；
- 测试点ESS不足10时只扩大带宽；
- LCP产生无限阈值时回退同landmark Split CP；
- 显式记录ESS、带宽扩张和回退比例。

## 评价

每个landmark报告：RMSE、MAE、MAPE、R²、覆盖数/33、coverage、mean/median/q90 width、ESS及回退率。重点比较相同或相近经验覆盖下的宽度，不能把欠覆盖造成的区间缩短称为有效改善。

