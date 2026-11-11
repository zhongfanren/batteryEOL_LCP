# Battery-LCP

电池寿命区间预测研究。当前任务固定电芯前 100 周期，在 ISU-ILCC 开发，研究简单寿命尺度以外的尾部风险信息能否改善局部校准；MATR 用于后续跨数据集复核。

先读 [研究入口](simple_cptransformer_cp/RESEARCH_START_HERE.md)，再看其指向的最新复核和 [实验总账](simple_cptransformer_cp/EXPERIMENT_INDEX.md)。实验限制与负结果以批次复核为准。

|目录|用途|
|---|---|
|`simple_cptransformer_cp/`|当前研究及冻结历史实验；新工作使用新批次目录|
|`simple_cptransformer_cp/batterylife_main_table/`|当前 ISU 方法、独立核查、绘图和科学契约测试|
|`battery_lcp/`、`python_lcp/`|早期实验和已有 CP/LCP 实现，保留历史路径|
|`third_party/BatteryLife/`|固定版本的上游模型、数据加载和分折实现|
|`baselines/`|MambaLithium 上游及独立基线实现|
|`tools/`、`patches/`|仓库核查与上游兼容补丁|

## 获取代码

克隆本仓库时加 `--recurse-submodules`；已有克隆执行：

```powershell
git submodule update --init --recursive
python tools/setup_dependencies.py
```

上游版本由 Git submodule 固定。`setup_dependencies.py` 幂等应用 BatteryLife 的可选 Reformer 依赖补丁，不重置已有修改。第三方代码沿用各自许可证；本仓库不替上游重新授权。

本仓库保留 GitHub 初始提交中的 [MIT 许可证](LICENSE)。

## 环境与数据

项目原有运行环境为 `D:/Anaconda3/envs/my_torch/python.exe`。上游依赖见 [BatteryLife requirements](third_party/BatteryLife/requirements.txt)，早期 LCP 依赖见 [python_lcp/requirements.txt](python_lcp/requirements.txt)。GPU/PyTorch 安装需匹配本机环境；本次整理没有重建或重新验证训练环境。

完整数据、模型、OOF、CSV、CDF 和日志留在研究工作区，不随 GitHub 仓库发布。**仅克隆源码不能立即复现已有数值或图。** 数据获取方式、权限和历史运行说明见各目录文档；不要上传访问令牌。当前脚本含历史绝对路径和冻结输入指纹，迁移机器时需恢复依赖与数据路径，另建批次后再运行，不覆盖旧结果。

GitHub 中保留源码、计划、配置、分折 JSON、来源/核查记录和 Markdown 结论。完整的本地研究入口仍含只在原工作区可用的产物链接。

## 同步前检查

```powershell
git add .
python tools/repository_audit.py --staged --write-local
git diff --cached --stat
```

核查会检查主仓库暂存文件的体积、被排除的产物、常见令牌，以及两个上游的版本和兼容补丁。完整清单写入被忽略的 `.local/repository_audit/`。核查通过后再提交、推送。精简范围和本次量化结果见 [仓库精简复核](REPOSITORY_REVIEW.md)。
