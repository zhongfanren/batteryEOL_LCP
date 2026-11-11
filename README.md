# Battery-LCP

电池寿命区间预测研究。固定电芯前 100 周期，在 ISU-ILCC 研究简单寿命尺度以外的尾部风险信息及局部校准；MATR 用于跨数据集复核。

从[研究入口](simple_cptransformer_cp/RESEARCH_START_HERE.md)了解当前判断，从[实验总账](simple_cptransformer_cp/EXPERIMENT_INDEX.md)定位各批次结论。

## 获取与运行

```sh
git clone --recurse-submodules https://github.com/zhongfanren/batteryEOL_LCP.git
cd batteryEOL_LCP
python tools/setup_dependencies.py
```

以下命令从仓库根目录运行，`python` 使用当前激活的环境。依赖见 [BatteryLife requirements](third_party/BatteryLife/requirements.txt) 和 [LCP requirements](python_lcp/requirements.txt)。

数据目录示例为 `data/BatteryLife_Processed/` 和 `data/BatteryLife_v4/`，包含各数据源子目录及 `Life labels/`。运行时通过 `--data-root` 指定位置：

```sh
python simple_cptransformer_cp/run_t100.py --data-root data/BatteryLife_Processed
```

当前方法的运行、核查与绘图入口见[研究代码指南](simple_cptransformer_cp/README.md)。

## 目录与同步范围

|目录|内容|
|---|---|
|`simple_cptransformer_cp/`|当前研究、实验总账和分批复核|
|`simple_cptransformer_cp/batterylife_main_table/`|ISU 方法、评测、绘图与科学契约测试|
|`battery_lcp/`、`python_lcp/`|早期实验与 CP/LCP 实现|
|`docs/history/`|合并后的早期设计与研究记录|
|`third_party/`、`baselines/`|固定版本的上游依赖与基线|
|`tools/`、`patches/`|同步核查与上游兼容补丁|

GitHub 保存源码、计划、配置、分折、审计记录和 Markdown 结论，主仓库约 8 MB。数据、模型、拟合参数、OOF 数值、CSV、CDF 和日志保存在本地。上游通过 submodule 固定版本，兼容修改通过补丁恢复。

同步前运行：

```sh
git add .
python tools/repository_audit.py --staged --write-local
git diff --cached --stat
```

完整清单与同步记录保存在 `.local/repository_audit/`。项目采用 [MIT 许可证](LICENSE)，第三方代码沿用各自许可证。
