# ISU-ILCC 固定划分的预测器与 CP/LCP 试验

2026-10-27 两模型已完成。独立复算与研究分析见 [analysis_20261027/ANALYSIS.md](analysis_20261027/ANALYSIS.md)。预测器及原始区间输出保持冻结；补充的尺度消融明确标为查看测试结果后的探索性诊断。

研究问题：在材料、规格一致但实验协议不同的电芯总体中，基于早期信息和协议的相似性，能否识别误差较大的电芯，并在保持合理覆盖率时把更宽的区间分配给它们？

首轮固定 CPMLP 与 CPLSTM 两个预测器。前者在本地混合 Li-ion 复现中表现较好，后者在此前 MATR 筛选中表现较好；这不预设它们在 ISU-ILCC 上的排名。全部模型和全部预定方法一起报告，不用本次 test 来选择赢家并重新调参。239 块电芯沿用已建候选清单：97 train、20 dev、61 calibration、61 test，种子 20261026。SOC 元数据修订不妨碍先用曲线训练预测器；当前局部距离只用预测值和已核查的充放电倍率。

原 BatteryLife 模型从头训练，train 使用 1..100 周期前缀，dev 仅用 T=100 选 epoch；最多 100 epoch，至少 5 epoch 后启动耐心为 8 的早停。标准化只拟合 train。模型冻结之后再预测 calibration/test，每块电芯仅一条 T=100 分数。训练均值和中位数常数也作为点预测对照。开发集只有 20 块，早停和难度估计均会有波动。

区间比较使用名义 90%：绝对残差 Split CP、开发集误差估计难度后缩放的 normalized CP、预测值加倍率距离的探索性 baseLCP，以及 [Hore–Barber RLCP](https://arxiv.org/html/2310.07850#S3) 的预测值距离与预测值加倍率距离。高斯带宽固定为标准化坐标中的 1.5，RLCP 随机种子固定为 20261027。两个预测器采用相同设置。RLCP 包含随机中心和测试点在无穷大的权重；探索性 baseLCP 不声称边际覆盖保证。Normalized CP 的尺度函数只用开发残差拟合，再用独立 calibration 做归一化分数校准。

难度是开发集绝对误差的核回归估计，只是期望误差的代理，不是真实条件 90% 分位数。风险低/中/高的阈值来自开发集留一难度预测，不按测试误差分箱。结果同时报告难度与真实误差的 Spearman 相关、风险组覆盖/宽度、interval score、局部有效样本量和逐协议诊断。组内样本少，只作描述。无穷区间保留；若存在无穷区间，整体平均宽度和 interval score 在 JSON 中为 null，并列出数量，不以只计算有限区间的均值替代整体结果。

这些是同一批历史数据上的探索性试验。总体随机划分支持固定池内随机测试电芯的解释；面向未来电芯仍需校准/测试来自同一目标混合总体等假设。Split/normalized CP 的边际覆盖和 RLCP 的论文保证均不能解释成每组或每个电芯都达到 90%。改变数据总体后的 MAPE 或宽度也不是相对 MATR/混合数据的受控改进。

## 启动和查看

在工作区 PowerShell 运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File D:\Battery-LCP\simple_cptransformer_cp\batterylife_main_table\start_curated_isu.ps1
```

可见终端显示数据预处理、每 100 batch 的进度、每个 epoch 的开发集 MAPE；两个模型串行执行，结束后自动完成区间评估。启动后不需要聊天助手轮询进程。

- `queue_state.json`：队列 PID、各模型进程、完成或失败状态。
- `CPMLP/progress.json`、`CPLSTM/progress.json`：每 100 batch 和每个阶段/epoch 更新，数据预处理期间可查看终端进度条。
- `CPMLP.log`、`CPLSTM.log`：完整终端输出。
- `COMPARISON.md`：队列结束后的总表。
- 各模型目录的 `REPORT.md`、`results.json`、`group_diagnostics.csv`、`test_intervals.csv`：冻结点预测和区间分析。

复算已完成模型的区间分析无需训练，例如：

```powershell
& D:\Anaconda3\envs\my_torch\python.exe simple_cptransformer_cp/batterylife_main_table/run_curated_isu.py --model CPMLP --evaluate-only
```

本轮保留全部已完成文件。若训练被中断，先确认原进程已停止；当前脚本不会覆盖已开始的训练，重新完整训练应使用新的 `--out` 目录。预处理缓存用于节省第二个预测器重复读取曲线的时间。
