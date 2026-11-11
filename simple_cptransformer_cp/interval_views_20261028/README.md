# 固定第100周期：逐测试电芯区间查看

只读取和展示既有结果，不训练、不校准、不重算上下界，也不修改原文件。所有原测试电芯均展示；按真实寿命排序只为阅读。

- `ISU_CPMLP_test61.csv`：当前主设置CPMLP seed2024，原61块test，每块一行；含真实寿命、点预测及4种方法的上下界/覆盖。
- `ISU_test61.png`：同样61块，寿命尺度/六模型分歧尺度/OOF平均风险尺度三列区间。最新尾部模型退化成寿命尺度，故没有单列重复曲线；原文件保留全部18方法。
- `MATR_CPTransformer_seed2024_test33.csv`：早期MATR CPTransformer＋LCP试验，固定seed2024，原33块test，每块一行；普通CP、固定RBF和3种Hopfield方法。
- `MATR_test33.png`：上述历史试验的普通CP/固定RBF/自监督Hopfield三列，所有33块均保留。旧LCP是basic weighted诊断，不等于当前RLCP保证。
- `MATR_11models_splitCP_test33.csv`：后来的11模型普通Split CP基准，长表363行，每个模型33块；与前述历史CPTransformer不是同一个预测器检查点，切勿混比。

图中：横轴为寿命周期，纵轴为电芯ID；线段表示区间，蓝点表示点预测，黑叉表示真实EOL。红线表示未覆盖真实EOL。CSV带UTF-8 BOM，适合Excel打开；列名为中文。

这些都是总寿命EOL区间，不是随运行更新的轨迹；当前统一观察到第100周期。旧MATR负下界保留原样，未为画图裁剪。ISU的真实值微小小数来自原保存结果的逆标准化误差，未篡改成整数。

查看不改变训练结果；若据此选模型、距离、参数或筛选电芯，则仍属于使用历史test进行方法开发。三个来源本身都已有test暴露，不能称全新确认性检验。

原始来源和哈希见 `source_manifest.json`。横向任务定义见项目 RESEARCH_START_HERE.md。
