"""Independent short review and group accounting for the frozen first round."""
import argparse
import numpy as np
import pandas as pd
import run_score_locality as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=run.Path, default=run.DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    audit = run.cp.read(out / "audit.json")
    assert all(run.cp.digest(row["path"]) == row["sha256"] for row in audit["inputs"])
    metric = pd.read_csv(out / "metrics.csv")
    original = pd.read_csv(out / "original_intervals.csv")
    groups = pd.read_csv(out / "group_metrics.csv")
    by_job = pd.read_csv(out / "by_job.csv")
    aggregate = pd.read_csv(out / "aggregate.csv")
    assert metric.groupby(["job", "repetition"]).method.nunique().eq(9).all()
    assert len(original) == 18 * 9 * 61
    # Every predefined three-group partition must recover the overall metrics.
    grouped = original.groupby(["job", "method"])
    checked = 0
    for (job, method), rows in grouped:
        for axis in ("life", "gap"):
            subset = groups[(groups.job == job) & (groups.method == method) & (groups.phase == "original") & (groups.axis == axis)]
            assert subset.presentations.sum() == len(rows)
            for target, column in (("coverage", "covered"), ("mean_width", "width"), ("interval_score", "interval_score")):
                np.testing.assert_allclose(np.sum(subset[target].to_numpy()*subset.presentations.to_numpy())/len(rows),
                                           np.mean(rows[column].to_numpy()), rtol=1e-10)
                checked += 1
    pooled = []
    for (model, phase, method, axis, group), rows in groups.groupby(["model", "phase", "method", "axis", "group"]):
        count = rows.presentations.to_numpy()
        pooled.append({"model": model, "phase": phase, "method": method, "axis": axis, "group": group,
                       "presentations": int(count.sum()),
                       **{col: float(np.sum(rows[col].to_numpy()*count)/count.sum())
                          for col in ("coverage", "mean_width", "interval_score", "relative_width")}})
    pd.DataFrame(pooled).to_csv(out / "groups_pooled.csv", index=False)
    contrasts = []
    for model in ("CPMLP", "CPLSTM"):
        rows = by_job[by_job.model == model]
        baseline = rows[rows.method == "relative__global"].set_index("job")
        for method in run.METHODS:
            part = rows[rows.method == method].set_index("job").loc[baseline.index]
            contrasts.append({"model": model, "method": method,
                 "width_gain_pct_vs_relative": float(100*(1-np.mean(part.mean_width.to_numpy())/np.mean(baseline.mean_width.to_numpy()))),
                 "score_gain_pct_vs_relative": float(100*(1-np.mean(part.interval_score.to_numpy())/np.mean(baseline.interval_score.to_numpy()))),
                 "width_win_fits_vs_relative": int((part.mean_width < baseline.mean_width).sum()),
                 "score_win_fits_vs_relative": int((part.interval_score < baseline.interval_score).sum())})
    pd.DataFrame(contrasts).to_csv(out / "contrasts_vs_relative.csv", index=False)
    # Diagnostics of groups on actual original test cells, not assumed difficulty.
    rows = original[original.method == "relative__global"].copy()
    rows["absolute_error"] = np.abs(rows.eol-rows.prediction)
    rows["relative_error"] = rows.absolute_error/np.maximum(1., rows.prediction)
    support = rows.groupby(["model", "gap_group"])[["absolute_error", "relative_error"]].agg(["count", "mean"])
    support.to_csv(out / "original_gap_group_error_support.csv")
    lines = ["# 本轮先看这里", "", "研究问题：扣除寿命尺度后，模型分歧还能否改善区间分配？",
             "本轮只用ISU现有预测，不做GPU训练。三分数×三构造、18冻结设置、200次历史池重分全部保留。", "",
             "**结论：尺度改进仍有依据，新的局部权重尚未超过强基线；降低尾部模型正则化也没有带来更好的训练侧泛化。**", "",
             "## 1. 可以保留的正面结果", "",
             "简单寿命尺度relative仍作基线；分歧尺度gap6只把s从p改为p+d，权重保持全局。"]
    for model in ("CPMLP", "CPLSTM"):
        part = aggregate[aggregate.model == model].set_index("method")
        base, scaled = part.loc["relative__global"], part.loc["gap6__global"]
        lines.append(f"- {model}：分歧尺度使宽度{base.mean_width:.1f}→{scaled.mean_width:.1f}，interval score {base.interval_score:.1f}→{scaled.interval_score:.1f}，覆盖{base.coverage:.2%}→{scaled.coverage:.2%}。这是重现既有尺度收益，不是新LCP成果。")
    lines += ["", "## 2. 本轮新增局部性没有达到目标", "",
              "在同一分数下，只按分歧局部化，比全局法略提高高分歧组覆盖，但改善不足以解决漏覆盖，并出现无穷区间。高分歧组按dev协变量三分位预定，不能称每块都难。"]
    pooled_table = pd.DataFrame(pooled)
    for model in ("CPMLP", "CPLSTM"):
        part = pooled_table[(pooled_table.model == model) & (pooled_table.phase == "resplits") & (pooled_table.axis == "gap") & (pooled_table.group == 2)].set_index("method")
        count = int(metric[(metric.model == model) & (metric.method == "relative__gap") & (metric.repetition > 0)].infinite.sum())
        lines.append(f"- {model}：relative分数的高分歧组覆盖{part.loc['relative__global','coverage']:.2%}→{part.loc['relative__gap','coverage']:.2%}；9设置×200×61次测试呈现中{count}次∞。同一权重用于三分数，故∞位置相同，不是三批独立事件。")
    lines += ["∞全部计入总成本，故分歧局部法总体平均宽度/score为∞；不删去少数失败来报有限均值。原test主模型本轮九法都有限，不代表重分时不会失败。",
              "按寿命局部化没有∞，但在relative/gap6同分数对照中平均宽度增加约1%–2%，score也稍差。",
              "主CPMLP原61test：relative全局覆盖88.52%、宽947.4；relative分歧局部覆盖90.16%、宽1023.0。单次覆盖上升伴随成本增加，不能代替跨设置结论。", "",
              "## 3. 尾部模型：非恒定不等于有用", "",
              "原alpha=.05在全量和各风险训练折均系数全零。预设alpha=.01后能产生变化，但验证pinball（越小越好）CPMLP比常数差6.9%，CPLSTM比常数差0.3%；对相同alpha寿命only两者也更差。alpha=0敏感性更差。",
              "两模型均未通过训练侧门槛，因此不继续把这个尾部模型包装成新分数/距离。固定OOF库的风险CV不是端到端嵌套独立验证，不能把这些损失当最终泛化成绩。", "",
              "## 4. 下一步只推进一条", "",
              "不继续围绕同一模型对分歧反复调alpha或按历史test搜带宽。下一批先在建模侧检查低维早期曲线特征：它们能否区分归一化残差的上尾，而不是只预测平均误差或寿命。",
              "先固定少量物理可解释候选（例如前100周期容量变化/曲线差异），与寿命only做训练侧对照，确认稳定信号后才接局部校准。若仍无额外信息，保留简单尺度CP作为方法结论，不为了LCP继续加复杂度。",
              "当前尚未实施这项新特征实验，也不据本轮否定所有LCP。", "",
              "图入口：visuals/relative_intervals_test61.png → visuals/CDF_relative_all61.pdf；再对照absolute/gap6两份PDF。",
              "所有图固定展示全部61块主test，不挑好看样本。历史test与重复划分仍只提供探索证据。详细数值见REPORT.md及CSV。"]
    (out / "BRIEF.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    run.save(out / "review_audit.json", {"group_reaggregation_checks": checked, "all_methods_present": True,
              "source_hashes_verified": True, "original_rows": len(original), "metric_rows": len(metric),
              "review_script_sha256": run.cp.digest(__file__)})
    if (out / "visualization_complete.json").exists():
        state = run.cp.read(out / "state.json")
        state["plots"] = "complete; see visualization_complete.json"
        state["review"] = "complete; see review_audit.json"
        state["updated_at"] = run.cp.stamp()
        run.save(out / "state.json", state)
    print(out / "BRIEF.md")


if __name__ == "__main__":
    main()
