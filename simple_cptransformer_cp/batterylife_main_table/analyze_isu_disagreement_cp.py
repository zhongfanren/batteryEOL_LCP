"""Independent arithmetic checks and a concise review of the frozen pilot."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_isu_disagreement_cp import ROOT, DEFAULT_OUT, PRIMARY, digest, read


def independent_radius(scores, cal, test, h, noise):
    result = []
    order = np.argsort(scores)
    for query, epsilon in zip(test, noise):
        center = query + h * epsilon
        logs = -np.r_[np.sum((cal - center)**2, axis=1), np.sum((query - center)**2)] / (2*h*h)
        weight = np.exp(logs - logs.max())
        target = .9 * weight.sum()
        cumulative, value = 0., np.inf
        for i in order:
            cumulative += weight[i]
            if cumulative >= target:
                value = scores[i]
                break
        result.append(value)
    return np.asarray(result)


def main():
    out = DEFAULT_OUT
    metrics = pd.read_csv(out / "metrics.csv")
    original = pd.read_csv(out / "original_intervals.csv")
    groups = pd.read_csv(out / "group_metrics.csv")
    support = pd.read_csv(out / "group_support.csv")
    paired = pd.read_csv(out / "paired_by_job.csv")
    aggregate = pd.read_csv(out / "aggregate.csv")
    plan = read(ROOT / "plan.json")
    assert read(out / "state.json")["state"] == "complete"
    assert digest(Path(__file__).with_name("run_isu_disagreement_cp.py")) == read(out / "plan.json")["code_sha256"]
    assert len(metrics) == 18 * 201 * 12
    np.testing.assert_allclose(original.width, original.upper - original.lower)
    expected_coverage = (original.lower <= original.eol) & (original.eol <= original.upper)
    np.testing.assert_array_equal(expected_coverage, original.covered)
    score = original.width + 20 * (np.maximum(original.lower - original.eol, 0) + np.maximum(original.eol - original.upper, 0))
    np.testing.assert_allclose(score, original.interval_score)
    baseline_rows, local_intervals = 0, 0
    loaded = {s["key"]: dict(np.load(ROOT / "analysis" / s["key"] / "features.npz")) for s in plan["jobs"]}
    for spec in plan["jobs"]:
        job = spec["key"]
        old = pd.read_csv(ROOT / "analysis" / job / "resplit_metrics.csv").set_index(["method", "repetition"])
        new = metrics[(metrics.job == job) & metrics.method.isin(["split_cp", "relative_cp", "prediction_scale_cp"])].set_index(["method", "repetition"])
        for column in ("coverage", "mean_width", "interval_score"):
            np.testing.assert_allclose(new[column], old.loc[new.index, column], rtol=1e-10, atol=1e-8)
        baseline_rows += len(new)
        data = loaded[job]
        cohort = {k: v for k, v in loaded.items() if k.startswith(f"s{spec['split_seed']}_")}
        phase_values = {}
        for phase in ("train", "calibration", "test"):
            pred = np.maximum(1., data[f"{phase}_prediction"])
            means = {model: np.mean([v[f"{phase}_prediction"] for k, v in cohort.items() if k.endswith(model)], axis=0) for model in ("CPMLP", "CPLSTM")}
            gap = np.abs(means["CPMLP"] - means["CPLSTM"])
            phase_values[phase] = pred, gap, np.column_stack((np.log(pred), np.log1p(gap / pred)))
        train_z = phase_values["train"][2]
        mean, sd = train_z.mean(axis=0), train_z.std(axis=0)
        sd[sd == 0] = 1.
        cal_z, test_z = [(phase_values[p][2] - mean) / sd for p in ("calibration", "test")]
        cal_error = np.abs(data["calibration_eol"] - data["calibration_prediction"])
        noise = np.random.default_rng(20261028).normal(size=(61, 2))
        for scale_name in ("relative", "gap_scale"):
            cal_scale = phase_values["calibration"][0].copy()
            test_scale = phase_values["test"][0].copy()
            if scale_name == "gap_scale":
                cal_scale += phase_values["calibration"][1]
                test_scale += phase_values["test"][1]
            for location, dim in (("life", 1), ("life_gap", 2)):
                for h in (3., 6.):
                    method = f"{scale_name}_rlcp_{location}_h{h:g}"
                    radius = independent_radius(cal_error / cal_scale, cal_z[:, :dim], test_z[:, :dim], h, noise[:, :dim]) * test_scale
                    actual = original[(original.job == job) & (original.method == method)].set_index("cell_file").loc[data["test_names"]]
                    np.testing.assert_allclose(radius, actual.radius, rtol=1e-10, atol=1e-8)
                    local_intervals += len(radius)
        # Each covariate partition must recover the complete interval metrics.
        for phase, keep in (("original", metrics.repetition.eq(0)), ("resplits", metrics.repetition.gt(0))):
            for method in metrics.method.unique():
                ref = metrics[(metrics.job == job) & (metrics.method == method) & keep]
                for axis in ("gap", "life"):
                    rows = groups[(groups.job == job) & (groups.phase == phase) & (groups.method == method) & (groups.axis == axis)]
                    assert rows.presentations.sum() == 61 * len(ref)
                    for group_metric, whole_metric in (("coverage", "coverage"), ("mean_width", "mean_width"), ("interval_score", "interval_score")):
                        combined = np.sum(rows[group_metric] * rows.presentations) / rows.presentations.sum()
                        np.testing.assert_allclose(combined, ref[whole_metric].mean(), rtol=1e-10, atol=1e-8)
    audit = read(out / "audit.json")
    assert all(digest(r["path"]) == r["sha256"] for r in audit["inputs"])
    verification = {"baseline_rows_matched_prior_batch": baseline_rows, "original_cell_rows_recomputed": len(original),
                    "independent_rlcp_intervals_checked": local_intervals, "group_reaggregation_passed": True,
                    "source_files_unchanged": True}
    (out / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")

    selected_methods = ["relative_cp", "gap_scale_cp"]
    main_rows = aggregate[aggregate.method.isin(selected_methods)]
    scale_gain = paired[paired.method.eq("gap_scale_cp")]
    group_rows = groups[(groups.phase == "resplits") & groups.method.isin(selected_methods) & groups.axis.eq("gap")]
    group_summary = group_rows.groupby(["model", "method", "group"])[["coverage", "mean_width", "interval_score", "relative_width"]].mean()
    group_summary.to_csv(out / "group_summary.csv")
    lines = ["# 这一轮试下来，应该怎么判断", "", "**分歧作为误差尺度有小幅、一致的线索；作为当前RLCP距离，尚未体现稳定优势。**",
             "这仍是历史ISU数据上的探索，未训练点预测器，原单模型预测中心不变。", "",
             "## 全部设置的尺度对照", "", "先对每个设置200次重分平均，再对每模型9个设置等权平均。", "",
             "| 模型 | 方法 | 覆盖 | 平均宽度 | Interval score |", "|---|---|---:|---:|---:|"]
    for r in main_rows.itertuples():
        lines.append(f"| {r.model} | {'寿命＋分歧尺度' if r.method == 'gap_scale_cp' else '简单寿命尺度'} | {r.mean_coverage:.2%} | {r.mean_width:.1f} | {r.mean_interval_score:.1f} |")
    lines += ["", "这里的分歧尺度是 max(1,预测寿命)＋两个架构预测均值的绝对差，系数固定为1，没有根据结果调参。",
              "两个架构各用已有3种子预测；没有把六模型均值改作点预测中心。"]
    for model in ("CPMLP", "CPLSTM"):
        rows = scale_gain[scale_gain.model == model]
        lines += [f"- {model}：interval score在{int((rows.interval_score_gain_pct > 0).sum())}/9设置改善；平均宽度在{int((rows.mean_width_gain_pct > 0).sum())}/9改善。设置重叠，不是9个独立统计复现。"]
    lines += ["", "## 是否真的照顾了困难电芯", "", "高分歧组由dev的可观察分歧阈值定义，没有按测试误差分组。下面为各设置重分组覆盖的平均。", "",
              "| 模型 | 低分歧覆盖：原→加分歧 | 高分歧覆盖：原→加分歧 |", "|---|---:|---:|"]
    for model in ("CPMLP", "CPLSTM"):
        values = []
        for group in (0, 2):
            before = group_summary.loc[(model, "relative_cp", group), "coverage"]
            after = group_summary.loc[(model, "gap_scale_cp", group), "coverage"]
            values.append(f"{before:.1%} → {after:.1%}")
        lines.append(f"| {model} | {values[0]} | {values[1]} |")
    lines += ["", "高分歧组获得更多宽度，低分歧组区间收窄；高分歧组实际误差和大误差比例也较高（完整数据见group_support.csv）。",
              "但高分歧组覆盖仍只有约83–84%，所以还没有解决困难区域漏覆盖。低分歧组覆盖下降是宽度重新分配的代价，不能忽略。", "",
              "## 原主测试集并没有同步获胜", "", "原61块test上，简单寿命尺度覆盖88.5%、宽度947.4；加分歧后86.9%、宽度968.1。高分歧组20块，两种方法都只覆盖15块（75%）。",
              "所以不能把总体重分上的小幅改善说成主测试集已成功。每个原设置与原test结果均保留，没有按结果重选模型。", "",
              "## 距离这条路的结果", "", "相同分数下加入分歧的RLCP距离，在h=3和6都出现无穷区间。h=6逐电芯比例虽仅约0.003%–0.012%，完整平均宽度和score仍为无穷，不能删去再报效率。",
              "仅寿命RLCP h=6全体有限，但比同分数全局CP更宽，score也没有改善。本轮不支持把这一距离作为有效LCP改进。", "",
              "## 下一步的研究取舍", "", "保留分歧尺度作为新增简单对照，主数据集和预测器继续固定。下一轮若推进，应先补充建模侧的诚实残差监督，判断分歧是否能更准确地区分大误差尾部，而不是继续增加带宽或网络搜索。",
              "训练侧交叉拟合需要重新拟合少量辅助模型；必须先固定它们的用途、epoch选择和最终校准隔离。其误差分布也未必等于最终全训练模型，不能把OOF残差直接冒充最终模型的校准残差。",
              "本轮没有启动这类训练，没有后台任务。所有原模型与预测哈希保持不变。", "",
              "详细公式及全部方法：[REPORT.md](REPORT.md)。核查记录：[verification.json](verification.json)。"]
    (out / "BRIEF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for column, model in enumerate(("CPMLP", "CPLSTM")):
        x = np.arange(3)
        for method, label, color in (("relative_cp", "Lifetime scale", "#5579a6"), ("gap_scale_cp", "Lifetime + gap scale", "#d18443")):
            table = group_summary.loc[(model, method)].reindex([0, 1, 2])
            axes[0, column].plot(x, table.coverage * 100, "o-", label=label, color=color)
            axes[1, column].plot(x, table.relative_width, "o-", label=label, color=color)
        axes[0, column].axhline(90, color="grey", linestyle="--", linewidth=.8)
        axes[0, column].set_ylim(75, 100)
        axes[0, column].set_title(model)
        for ax in axes[:, column]:
            ax.set_xticks(x, ["Low gap", "Middle gap", "High gap"])
            ax.grid(alpha=.15)
    axes[0, 0].set_ylabel("Group coverage (%)")
    axes[1, 0].set_ylabel("Mean width / predicted lifetime")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Exploratory ISU audit: width allocation improves modestly; high-gap coverage remains low", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "allocation.png", dpi=170)
    plt.close(fig)
    print(json.dumps(verification, indent=2))
    print(f"REVIEW COMPLETE {out / 'BRIEF.md'}")


if __name__ == "__main__":
    main()
