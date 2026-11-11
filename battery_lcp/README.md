# 早期电池寿命与 RUL 实验

当前主线见[研究入口](../simple_cptransformer_cp/RESEARCH_START_HERE.md)。本目录保存固定 50 周期寿命预测、逐 landmark RUL 和 CPTransformer 核方法的早期实现。

## 数据

数据来源：[BatteryLife Processed](https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed)。访问令牌通过 `HF_TOKEN` 环境变量提供。

以下命令均从仓库根目录、在已激活的 Python 环境中执行：

```sh
python battery_lcp/download_matr.py --output-dir data/BatteryLife_Processed
```

预期数据结构为 `data/BatteryLife_Processed/MATR/` 和 `data/BatteryLife_Processed/Life labels/MATR_labels.json`。

## 运行入口

|任务|脚本|配置与结果说明|
|---|---|---|
|固定 50 周期 EOL|`run_experiment.py`|官方 test 33 块，validation 划分为模型开发与校准|
|逐 landmark RUL|`run_rul_landmarks.py`|landmark 50/100/150/200；train 90、dev 12、calibration 34、test 33|
|CPTransformer 与核方法|`run_cptransformer_kernels.py`|[冻结设计](../CPTRANSFORMER_KERNEL_DESIGN.md)|

```sh
python battery_lcp/run_experiment.py --data-root data/BatteryLife_Processed --seed 42
python battery_lcp/run_rul_landmarks.py --data-root data/BatteryLife_Processed --landmarks 50 100 150 200 --seed 42
python battery_lcp/run_cptransformer_kernels.py --data-root data/BatteryLife_Processed --seed 42
```

点预测、区间、模型和训练记录保存在 `battery_lcp/outputs/`。早期设计、残差诊断与技术记录合并在 [MATR 历史资料](../docs/history/MATR_RESEARCH.md)。
