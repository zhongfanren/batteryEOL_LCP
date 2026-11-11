# 研究代码指南

[当前研究](RESEARCH_START_HERE.md) · [实验总账](EXPERIMENT_INDEX.md) · [方法与数据定义](batterylife_main_table/LCP_EXPERIMENT_SPEC.md)

## 当前方法入口

路径相对于 `simple_cptransformer_cp/batterylife_main_table/`。

|批次|计算|核查|绘图|
|---|---|---|---|
|DSCP-Local|`run_dscp_local.py`|`review_dscp_local.py`|`plot_dscp_local.py`|
|多分位画像|`run_quantile_profile.py`|`review_quantile_profile.py`|`plot_quantile_profile.py`|
|曲线信息初筛|`check_curve_signal.py`|`review_curve_signal.py`|`plot_curve_signal.py`|
|分数与局部性|`run_score_locality.py`|`analyze_score_locality.py`|`plot_score_locality.py`|

激活 Python 环境后，从仓库根目录运行。新批次使用新的输出目录，例如：

```sh
python simple_cptransformer_cp/batterylife_main_table/run_score_locality.py --out simple_cptransformer_cp/batterylife_main_table/score_locality_NEW
python simple_cptransformer_cp/batterylife_main_table/plot_score_locality.py --out simple_cptransformer_cp/batterylife_main_table/score_locality_NEW
python simple_cptransformer_cp/batterylife_main_table/analyze_score_locality.py --out simple_cptransformer_cp/batterylife_main_table/score_locality_NEW
```

PowerShell 的 `start_*.ps1` 同样使用当前环境的 `python`，工作目录由脚本位置确定。

## 批次文件

|文件|内容|
|---|---|
|`plan.json`|假设、配置、判断标准与输入/源码指纹|
|`REVIEW.md` 或 `BRIEF.md`|该批次结论|
|`REPORT.md` 或 `RESULTS.md`|自动生成的完整表格|
|`audit.json`、`review_audit.json`|来源、基线重放与独立核查|
|`state.json`|运行状态|
|`metrics.csv`、`original_intervals.csv`、`visuals/`|本地完整指标、区间与 CDF|

## 历史资料

- [MATR 固定 T100 基线：实现与首轮结果](MATR_BASELINE.md)
- [2026-09-30 阶段总结](RESEARCH_PROGRESS_SUMMARY_20260930.md)
- [2026-10-28 从 MATR 到 ISU](RESEARCH_PROGRESS_SUMMARY_20261028.md)
- [MATR 早期设计与研究记录](../docs/history/MATR_RESEARCH.md)
