"""Audit numerical outputs and compare ALL predeclared controls, including global."""
import argparse
import numpy as np
import pandas as pd
import check_curve_signal as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=run.Path, default=run.OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    audit = run.cp.read(out / "audit.json")
    assert all(run.cp.digest(item["path"]) == item["sha256"] for item in audit["inputs"])
    records = pd.read_csv(out / "predictions.csv")
    folds = pd.read_csv(out / "fold_metrics.csv")
    summary = pd.read_csv(out / "summary.csv")
    loss = .9*np.maximum(records.relative_error-records.predicted_q90, 0)+.1*np.maximum(records.predicted_q90-records.relative_error, 0)
    np.testing.assert_allclose(records.pinball, loss, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(records.below_q, records.relative_error <= records.predicted_q90)
    assert len(records) == 2*9*(97+5*20)
    assert records[records.phase == "oof_cv"].groupby(["model", "method"]).cell_file.nunique().eq(97).all()
    checks = 0
    counts = []
    for path in sorted((out / "neighbourhoods").glob("*.npz")):
        bank = dict(np.load(path, allow_pickle=False))
        pieces = path.stem.split("_")
        fold, model = int(pieces[1]), pieces[2]
        phase = "oof_cv" if path.stem.endswith("oof_cv") else "dev20"
        # CV query names are disjoint references; dev is disjoint all train97.
        assert not set(bank["reference_names"]) & set(bank["query_names"])
        order = np.argsort(bank["reference_scores"], kind="stable")
        scores = bank["reference_scores"][order]
        for method in [name for name in run.METHODS if name.startswith("kernel_")]:
            weights = bank[method]
            np.testing.assert_allclose(weights.sum(axis=1), 1.)
            expected = []
            for row in weights[:, order]:
                cumulative, chosen = 0., scores[-1]
                for value, weight in zip(scores, row):
                    cumulative += weight
                    if cumulative >= .9:
                        chosen = value
                        break
                expected.append(chosen)
            actual = records[(records.model == model) & (records.fold == fold) & (records.phase == phase) & (records.method == method)].set_index("cell_file").loc[bank["query_names"]]
            np.testing.assert_allclose(expected, actual.predicted_q90, rtol=1e-10, atol=1e-12)
            checks += len(expected)
        counts.append({"model": model, "fold": fold, "phase": phase, "query_cells": len(bank["query_names"]),
                       "reference_cells": len(bank["reference_names"]), "reference_relative_error_over_half": int(np.sum(bank["reference_scores"] > .5)),
                       "query_relative_error_over_half": int(np.sum(bank["query_scores"] > .5))})
    pd.DataFrame(counts).to_csv(out / "tail_support.csv", index=False)
    global_comparisons = []
    for row in summary[summary.method.str.startswith("kernel_")].itertuples():
        part = records[(records.model == row.model) & (records.phase == row.phase)]
        candidate = part[part.method == row.method].set_index(["fold", "cell_file"])
        control = part[part.method == "kernel_global"].set_index(["fold", "cell_file"]).loc[candidate.index]
        fm = folds[(folds.model == row.model) & (folds.phase == row.phase)].pivot(index="fold", columns="method", values="pinball")
        global_comparisons.append({"model": row.model, "phase": row.phase, "method": row.method,
            "gain_pct_vs_global": 100*(1-candidate.pinball.mean()/control.pinball.mean()),
            "fold_wins_vs_global": int((fm[row.method] < fm.kernel_global).sum()),
            "mean_q": row.mean_q, "fraction_below_q": row.fraction_below_q})
    comparison = pd.DataFrame(global_comparisons)
    comparison.to_csv(out / "comparisons_vs_global.csv", index=False)
    cell_gains = []
    for (model, phase), part in records.groupby(["model", "phase"]):
        pivot = part.pivot(index=["fold", "cell_file"], columns="method", values="pinball")
        for method in ("kernel_hand", "kernel_embedding", "kernel_curve"):
            for (fold, cell), gain in (pivot.kernel_global-pivot[method]).items():
                cell_gains.append({"model": model, "phase": phase, "method": method, "fold": fold,
                                   "cell_file": cell, "loss_reduction_vs_global": gain})
    gains = pd.DataFrame(cell_gains)
    gains.to_csv(out / "per_cell_gain_vs_global.csv", index=False)
    primary = gains[(gains.model == "CPMLP") & (gains.phase == "oof_cv") & (gains.method == "kernel_hand")]
    first_fold_share = primary[primary.fold == 0].loss_reduction_vs_global.sum()/primary.loss_reduction_vs_global.sum()
    lines = ["# 独立复核：曲线邻域有局部线索，尚未稳定胜强对照", "",
        "主结论：先保留低维手工特征邻域作为候选；当前证据不支持立即训练复杂表示或宣布LCP改进成功。原cal/test完全未评价。", "",
        "## 初筛通过不等于超过所有强基线", "",
        "预先门槛比较的是同路线life-only。手工与57维曲线邻域在两架构都过门槛，但OOF-CV中全局分布本身也比寿命邻域好。必须检查新增特征是否超过全局，而不是只修复一个差的寿命邻域。", "",
        "|模型|证据|邻域|相对寿命pinball改善|相对全局改善|相对全局胜折/视角|", "|---|---|---|---:|---:|---:|"]
    for model in ("CPMLP", "CPLSTM"):
        for phase in ("oof_cv", "dev20"):
            for method in ("kernel_hand", "kernel_curve", "kernel_embedding"):
                a = summary[(summary.model == model) & (summary.phase == phase) & (summary.method == method)].iloc[0]
                b = comparison[(comparison.model == model) & (comparison.phase == phase) & (comparison.method == method)].iloc[0]
                lines.append(f"|{model}|{phase}|{method}|{a.gain_pct_vs_life:+.1f}%|{b.gain_pct_vs_global:+.1f}%|{int(b.fold_wins_vs_global)}/5|")
    lines += ["", "## 怎样判断", "",
        "- CPMLP：手工邻域OOF较全局有小幅改善，dev20也同向；这是目前最值得保留的线索。57维曲线邻域OOF对全局几乎无额外收益，不能把对寿命邻域的两位数改善全算成新信息。",
        "- CPLSTM：手工/曲线邻域dev20有改善，但OOF相对全局反而更差；跨预测器不一致。",
        "- 冻结embedding：部分OOF改善没有一致通过dev检查，不优先推进。分位数回归的附加特征在dev20均恶化；不能把训练侧分位数拟合变好当成稳定风险函数。",
        "- 本批检查的直接邻域无需先学单一难度值，说明这条检查路线与此前尾部回归常数退化不同。但它还只是参考误差分布的诊断，不是最终CP。", "",
        f"CPMLP手工邻域约{first_fold_share:.1%}净损失改善来自第0折；并非所有折同等贡献。dev20在每架构仅1块电芯曾在辅助模型视角下出现>50%相对误差，尾部验证支撑仍很薄，不能把五视角一致当成五个独立数据集。", "",
        "## 接下来", "",
        "先围绕手工邻域检查哪些参考尾点被加/减权、它们是否跨折一致；与全局和寿命邻域一同看CDF。不要据少数好图改特征或挑带宽。",
        "若继续方法实验，应冻结这套简单规则，再对主预测器做同分数的有效CP消融，并保留不同预测器复核；现有初筛收益不能直接搬成最终主模型收益。", "",
        "## 已核对", "",
        f"重新计算{len(records)}条pinball及分位覆盖指标，逐查询独立累计核对{checks}个核分位数；参考与查询名单隔离、原输入/冻结模型哈希均通过。",
        "97块OOF的参考模型训练有交叠，dev20五视角仍是同20块；dev历史已用过，不是新确认性数据。辅助模型约77块训练，与最终97块主预测器可能有条件误差差异。",
        "提取时首次混合推理批次造成CPLSTM最多0.0126周期的数值差，严格重放检查失败；保留失败批次curve_signal_20261104。v2按原heldout分组和原batch size推理，194个旧OOF预测最大偏差5.7e-14周期。没有放宽容差或重训。"]
    (out / "REVIEW.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    run.cp.write(out / "review_audit.json", {"prediction_rows_checked": len(records), "independent_kernel_quantiles": checks,
        "original_inputs_unchanged": True, "original_cal_test_not_evaluated": True,
        "review_sha256": run.cp.digest(__file__)})
    print(out / "REVIEW.md")


if __name__ == "__main__":
    main()
