"""Independent checks and a research review of the completed stability batch.

No predictor is retrained, no bandwidth is selected, and original results stay
frozen. Correlation ablations on the 122-cell pools are post-hoc diagnostics.
"""

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
ROOT = HERE / "isu_stability_20261028"
OUT = ROOT / "analysis_review_20261028"
MODELS = ("CPMLP", "CPLSTM")
FINITE = ("split_cp", "relative_cp", "prediction_scale_cp", "protocol_scale_cp", "curve_scale_cp")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(reference, values, query, leave_out=False, floor=1.):
    weights = -np.sum((query[:, None, :] - reference[None, :, :])**2, axis=2) / 4.5
    if leave_out:
        np.fill_diagonal(weights, -np.inf)
    weights = np.exp(weights - weights.max(axis=1, keepdims=True))
    return np.maximum(max(floor, .1 * np.median(values)), (weights @ values) / weights.sum(axis=1))


def finite_quantile(scores):
    index = math.ceil((len(scores) + 1) * .9) - 1
    return sorted(scores)[index] if index < len(scores) else np.inf


def local_quantile(cal_z, test_z, scores, h, seed):
    rng = np.random.default_rng(seed)
    q, effective, self_mass = [], [], []
    ordering = np.argsort(scores)
    for query in test_z:
        center = query + h * rng.standard_normal(len(query))
        distances = np.r_[np.sum((cal_z - center)**2, axis=1), np.sum((query - center)**2)]
        log_weights = -distances / (2 * h*h)
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        cumulative, radius = 0., np.inf
        for i in ordering:
            cumulative += weights[i]
            if cumulative >= .9:
                radius = scores[i]
                break
        q.append(radius)
        self_mass.append(weights[-1])
        effective.append(weights[:-1].sum()**2 / np.sum(weights[:-1]**2))
    return np.asarray(q), np.asarray(effective), np.asarray(self_mass)


def rank_correlation(x, y):
    return float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else np.nan


def main():
    OUT.mkdir(exist_ok=True)
    plan, state = read(ROOT / "plan.json"), read(ROOT / "state.json")
    assert state["state"] == "complete" and state["completed_jobs"] == 18
    assert digest(HERE / "run_isu_stability.py") == plan["runner_sha256"]
    assert digest(HERE / "run_curated_isu.py") == plan["training_runner_sha256"]
    published = pd.read_csv(ROOT / "summary.csv")
    metric_tables, point_rows, risk_rows, audit_rows = [], [], [], []
    labels = None
    for spec in plan["jobs"]:
        job_dir = ROOT / "analysis" / spec["key"]
        completion = read(job_dir / "complete.json")
        assert completion["runner_sha256"] == plan["runner_sha256"]
        assert completion["resplits"] == 200
        assert state["jobs"][spec["key"]]["exit_code"] == 0
        source = Path(spec["source"])
        model_dir = source / spec["model"]
        manifest, frozen, config = (read(p) for p in (source / "dataset_manifest.json", model_dir / "frozen_predictor.json", model_dir / "config.json"))
        phases = manifest["split"]["phases"]
        assert phases == read(spec["candidate"])["split"]["phases"]
        assert len(set(sum(phases.values(), []))) == 239
        assert digest(model_dir / "best_model.pt") == frozen["checkpoint_sha256"]
        assert digest(source / "dataset_manifest.json") == frozen["manifest_sha256"]
        assert read(source / "experiment_plan.json")["train_seed"] == spec["train_seed"]
        identity = read(job_dir / "feature_identity.json")
        assert identity["checkpoint_sha256"] == frozen["checkpoint_sha256"]
        if labels is None:
            labels = read(Path(manifest["data_root"]) / "Life labels" / "ISU-ILCC_labels.json")
        data = {k: v for k, v in np.load(job_dir / "features.npz").items()}
        for phase in phases:
            assert list(data[f"{phase}_names"]) == phases[phase]
            assert np.all(np.isfinite(data[f"{phase}_embedding"]))
            if phase != "train":
                np.testing.assert_allclose(data[f"{phase}_eol"], [labels[n] for n in phases[phase]], atol=.002, rtol=0)
        result = read(model_dir / "results.json")
        scale = result["scales"]
        train_y = np.array([labels[n] for n in phases["train"]])
        np.testing.assert_allclose([train_y.mean(), train_y.std()], [scale["prediction_mean"], scale["prediction_std"]])
        history = read(model_dir / "history.json")
        assert min(history, key=lambda x: x["mape_fraction"])["epoch"] == frozen["best_epoch"]
        point_rows.append({**spec, **result["point"]["test"], "epochs": len(history), "best_epoch": frozen["best_epoch"]})

        pca = PCA(n_components=3, whiten=True, svd_solver="full").fit(data["train_embedding"])
        metadata = {row["cell_file"]: row["metadata"] for row in manifest["cells"]}
        z = {}
        for phase in ("dev", "calibration", "test"):
            pred = data[f"{phase}_prediction"]
            rates = np.array([[metadata[n][k][0]["rate_in_C"] for k in ("charge_protocol", "discharge_protocol")]
                              for n in data[f"{phase}_names"]])
            level = ((pred - scale["prediction_mean"]) / scale["prediction_std"])[:, None]
            z[phase] = (level, np.column_stack((level, (rates - scale["rate_mean"]) / scale["rate_std"])),
                        np.column_stack((level, pca.transform(data[f"{phase}_embedding"]))))
        pool_z = tuple(np.r_[a, b] for a, b in zip(z["calibration"], z["test"]))
        pred = np.r_[data["calibration_prediction"], data["test_prediction"]]
        y = np.r_[data["calibration_eol"], data["test_eol"]]
        errors = np.abs(y - pred)
        dev_error = np.abs(data["dev_eol"] - data["dev_prediction"])
        scales = {"relative_cp": np.maximum(1, pred),
                  "prediction_scale_cp": smooth(z["dev"][0], dev_error, pool_z[0]),
                  "protocol_scale_cp": smooth(z["dev"][1], dev_error, pool_z[1]),
                  "curve_scale_cp": smooth(z["dev"][2], dev_error, pool_z[2])}
        dev_loo_scale = smooth(z["dev"][0], dev_error, z["dev"][0], leave_out=True)
        dev_normalized = dev_error / dev_loo_scale
        normalized = errors / scales["prediction_scale_cp"]
        risk_curve = smooth(z["dev"][2], dev_normalized, pool_z[2], floor=1e-6)
        risk_prediction = smooth(z["dev"][0], dev_normalized, pool_z[0], floor=1e-6)
        cutpoints = np.quantile(smooth(z["dev"][2], dev_normalized, z["dev"][2], leave_out=True, floor=1e-6), [1/3, 2/3])
        groups = np.searchsorted(cutpoints, risk_curve, side="right")
        risk_rows.append({"job": spec["key"], "model": spec["model"],
                          "original_test_curve_risk_rho": rank_correlation(risk_curve[61:], normalized[61:]),
                          "pool122_curve_risk_rho": rank_correlation(risk_curve, normalized),
                          "pool122_prediction_risk_rho_posthoc": rank_correlation(risk_prediction, normalized),
                          "posthoc_note": "pool correlations and prediction-only risk ablation are post-hoc diagnostics"})
        table = pd.read_csv(job_dir / "resplit_metrics.csv")
        assert len(table) == 201 * 9 and table.groupby("repetition").size().eq(9).all()
        assert set(table.repetition) == set(range(201)) and table.n.eq(61).all()
        assert table.groupby(["repetition", "method"]).size().eq(1).all()
        original = pd.read_csv(job_dir / "original_test_intervals.csv")
        assert len(original) == 61 * 9
        rng = np.random.default_rng(20261028)
        permutations = {0: np.arange(122)}
        for repetition in range(1, 201):
            value = rng.permutation(122)
            if repetition in (1, 100, 200):
                permutations[repetition] = value
        for repetition, permutation in permutations.items():
            ci, ti = permutation[:61], permutation[61:]
            radii = {"split_cp": np.full(61, finite_quantile(errors[ci]))}
            for method, s in scales.items():
                radii[method] = finite_quantile(errors[ci] / s[ci]) * s[ti]
            ess = {}
            for h in (1.5, 3.):
                for feature_name, coords in (("prediction", pool_z[0]), ("curve", pool_z[2])):
                    method = f"normalized_rlcp_{feature_name}_h{h:g}"
                    q, e, mass = local_quantile(coords[ci], coords[ti], normalized[ci], h, 20261028 + repetition)
                    assert np.all(mass[~np.isfinite(q)] > .1 - 1e-12)
                    radii[method], ess[method] = q * scales["prediction_scale_cp"][ti], e
            for method, radius in radii.items():
                row = table[(table.repetition == repetition) & (table.method == method)].iloc[0]
                lower, upper = np.maximum(0, pred[ti] - radius), pred[ti] + radius
                covered = (lower <= y[ti]) & (y[ti] <= upper)
                width = upper - lower
                score = width + 20 * (np.maximum(lower - y[ti], 0) + np.maximum(y[ti] - upper, 0))
                np.testing.assert_allclose([covered.mean(), width.mean(), score.mean()],
                                           [row.coverage, row.mean_width, row.interval_score], rtol=2e-5, atol=.03)
                assert int((~np.isfinite(radius)).sum()) == row.infinite
                if method in ess:
                    np.testing.assert_allclose(ess[method].mean(), row.mean_ess, rtol=2e-5, atol=.001)
                for group in (0, 1, 2):
                    keep = groups[ti] == group
                    assert int(keep.sum()) == row[f"group_{group}_n"]
                    if keep.any():
                        np.testing.assert_allclose(covered[keep].mean(), row[f"group_{group}_coverage"])
                if repetition == 0:
                    cells = original[original.method == method].set_index("cell_file").loc[phases["test"]]
                    np.testing.assert_allclose(radius, cells.radius, rtol=2e-5, atol=.03)
        for (method, is_original), g in table.groupby(["method", table.repetition.eq(0)]):
            row = published[(published.job == spec["key"]) & (published.method == method) & (published.original_test == is_original)].iloc[0]
            np.testing.assert_allclose([g.coverage.mean(), np.mean(g.mean_width), np.mean(g.interval_score)],
                                       [row.mean_coverage, row.mean_width, row.mean_interval_score])
        table = table[table.repetition > 0].copy()
        table["job"], table["model"], table["split_seed"], table["train_seed"] = spec["key"], spec["model"], spec["split_seed"], spec["train_seed"]
        metric_tables.append(table)
        audit_rows.append({"job": spec["key"], "status": "pass", "checked_repetitions": [0, 1, 100, 200],
                           "complete_resplits": 200, "checkpoint_sha256": frozen["checkpoint_sha256"]})
        print(f"AUDIT_PASS {spec['key']}", flush=True)

    all_metrics = pd.concat(metric_tables, ignore_index=True)
    original_summary = published[~published.original_test].copy()
    aggregate_rows, paired_rows, group_rows = [], [], []
    for (model, method), g in all_metrics.groupby(["model", "method"]):
        group = original_summary[(original_summary.model == model) & (original_summary.method == method)]
        aggregate_rows.append({"model": model, "method": method, "n_predictors": len(group),
                               "mean_coverage": g.coverage.mean(), "mean_width": np.mean(g.mean_width),
                               "mean_interval_score": np.mean(g.interval_score),
                               "fraction_sets_with_infinite": (g.infinite > 0).mean(),
                               "fraction_cell_intervals_infinite": g.infinite.sum() / g.n.sum(),
                               "mean_ess": g.mean_ess.mean()})
        baseline = original_summary[(original_summary.model == model) & (original_summary.method == "prediction_scale_cp")].set_index("job").loc[group.job]
        paired_rows.append({"model": model, "method": method,
                            "width_better_fits": int((group.mean_width.to_numpy() < baseline.mean_width.to_numpy()).sum()),
                            "interval_score_better_fits": int((group.mean_interval_score.to_numpy() < baseline.mean_interval_score.to_numpy()).sum()),
                            "mean_width_delta": np.mean(group.mean_width.to_numpy() - baseline.mean_width.to_numpy()),
                            "mean_interval_score_delta": np.mean(group.mean_interval_score.to_numpy() - baseline.mean_interval_score.to_numpy())})
        for risk in (0, 1, 2):
            per_fit = []
            for _, fit in g.groupby("job"):
                n = fit[f"group_{risk}_n"]
                if n.sum():
                    per_fit.append((n * fit[f"group_{risk}_coverage"]).sum() / n.sum())
            n = g[f"group_{risk}_n"]
            group_rows.append({"model": model, "method": method, "risk_group": risk,
                               "mean_per_fit_group_coverage": np.mean(per_fit),
                               "pooled_descriptive_group_coverage": (n*g[f"group_{risk}_coverage"]).sum()/n.sum(),
                               "average_group_n_per_testset": n.mean(),
                               "warning": "overlapping cell assignments; not independent test counts"})
    aggregate, paired, risk = pd.DataFrame(aggregate_rows), pd.DataFrame(paired_rows), pd.DataFrame(risk_rows)
    group_table, points = pd.DataFrame(group_rows), pd.DataFrame(point_rows)
    for name, table in (("aggregate", aggregate), ("paired_fit_comparison", paired), ("risk_signal", risk),
                        ("group_coverage", group_table), ("point_predictors", points)):
        table.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = ("#2a6f97", "#d2691e")
    short = ["Split", "Lifetime", "Dev scale", "Protocol", "Curve"]
    for offset, model in enumerate(MODELS):
        g = aggregate[aggregate.model == model].set_index("method").loc[list(FINITE)]
        x = np.arange(5) + (offset-.5)*.36
        axes[0, 0].bar(x, g.mean_width, width=.34, color=colors[offset], label=model)
        axes[0, 1].bar(x, g.mean_interval_score, width=.34, color=colors[offset], label=model)
        high = group_table[(group_table.model == model) & (group_table.risk_group == 2)].set_index("method").loc[list(FINITE)]
        axes[1, 0].plot(np.arange(5), high.mean_per_fit_group_coverage*100, "o-", color=colors[offset], label=model)
        local_names = ("normalized_rlcp_prediction_h1.5", "normalized_rlcp_curve_h1.5",
                       "normalized_rlcp_prediction_h3", "normalized_rlcp_curve_h3")
        local = aggregate[aggregate.model == model].set_index("method").loc[list(local_names)]
        axes[1, 1].bar(np.arange(4)+(offset-.5)*.36, local.fraction_cell_intervals_infinite*100,
                       width=.34, color=colors[offset], label=model)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2); ax.legend()
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(np.arange(5), short)
    axes[0, 0].set(title="Average width: finite CP baselines", ylabel="Cycles")
    axes[0, 1].set(title="Interval score (lower is better)", ylabel="Cycles + miss penalties")
    axes[1, 0].set(title="Remaining high-risk group coverage", ylabel="Coverage (%)", ylim=(65, 96))
    axes[1, 0].axhline(90, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(np.arange(4), ["Pred h=1.5", "Curve h=1.5", "Pred h=3", "Curve h=3"])
    axes[1, 1].set(title="RLCP: fraction of cell intervals that are infinite", ylabel="Infinite cell intervals (%)")
    fig.suptitle("ISU-ILCC: 3 partitions x 3 initializations x 2 predictors; exploratory", fontsize=14)
    fig.savefig(OUT / "stability_findings.png", dpi=180)
    plt.close(fig)

    def value(model, method):
        return aggregate[(aggregate.model == model) & (aggregate.method == method)].iloc[0]
    lines = ["# ISU-ILCC 稳定性批次：结果核查与研究判断", "",
             "2026-10-28 11:41:50–13:06:48，18/18设置完成，约85分钟，无失败或超时。包含16次新增拟合和2个历史模型复用。", "",
             "## 运行与结果核查", "",
             "已核对全部模型、代码与划分指纹；每次239块电芯四阶段互斥；特征逐电芯ID、标签和训练尺度一致。各任务均有200次重分和完整9种方法。独立重算全部任务的原test及第1、100、200次重分，包括尺度、分位数、RLCP自权重/ESS、逐电芯半径及分组覆盖；汇总表亦从原始指标复算通过。", "",
             "200次重分仅在每个预测器固定的122块cal+test电芯上进行。三次train/dev划分及不同模型的电芯也相互重叠。以下均是探索性描述，不把重复次数当作新增独立样本量。", "",
             "## H1：寿命尺度CP的收益重复出现", "",
             "这里区分直接寿命尺度 s=max(1,预测寿命) 与开发集拟合尺度 s=开发绝对误差关于预测寿命的核回归。后者只有20块开发电芯，但原始cal/test独立于其拟合。", "",
             "| 模型 | 方法 | 平均覆盖 | 平均宽度（周期） | Interval score（越低越好） |",
             "|---|---|---:|---:|---:|"]
    for model in MODELS:
        for method in FINITE:
            row = value(model, method)
            lines.append(f"| {model} | {method} | {row.mean_coverage:.2%} | {row.mean_width:.1f} | {row.mean_interval_score:.1f} |")
    lines += ["", "上述平均先在每个固定预测器的200次重分上计算，再对9个预测器设置等权汇总。边际覆盖接近90%符合固定池随机重分及有限样本分位数的设计预期，不能当作未来新电芯覆盖已被证明的证据。", ""]
    for model in MODELS:
        split, dev_scale = value(model, "split_cp"), value(model, "prediction_scale_cp")
        lines.append(f"- {model}：开发寿命尺度相对Split CP，宽度减少{1-dev_scale.mean_width/split.mean_width:.1%}，interval score减少{1-dev_scale.mean_interval_score/split.mean_interval_score:.1%}；9/9设置的平均宽度和interval score均改善。")
    lines += ["", "直接寿命尺度CP的平均宽度稍大于开发尺度CP，但interval score反而更低。开发尺度CP不是所有指标上的赢家；需要兼顾漏覆盖严重程度与局部表现。", "",
              "## H2：当前曲线表示尚未形成稳定的LCP效率收益", ""]
    for model in MODELS:
        baseline, curve = value(model, "prediction_scale_cp"), value(model, "curve_scale_cp")
        count = paired[(paired.model == model) & (paired.method == "curve_scale_cp")].iloc[0]
        original_risk = risk[risk.model == model].original_test_curve_risk_rho
        lines.append(f"- {model}：曲线尺度CP的宽度增加{curve.mean_width/baseline.mean_width-1:.1%}，9/9设置均更宽；interval score平均减少{1-curve.mean_interval_score/baseline.mean_interval_score:.1%}，仅{int(count.interval_score_better_fits)}/9设置改善。曲线估计的归一化风险在原test上的Spearman平均{original_risk.mean():.3f}，{int((original_risk>0).sum())}/9为正。")
    lines += ["", "因此曲线表示存在弱到中等的剩余风险排序信号，但当前固定3维PCA、高斯距离和20块开发残差的组合没有把它稳定转化为更有效的区间。该结果限定于本轮表示与构造，不证明早期曲线无用或所有LCP无效。", "",
              "| 模型 | RLCP特征/带宽 | 至少出现一个无穷区间的测试集合比例 | 无穷区间占逐电芯区间的比例 | 平均ESS |",
              "|---|---|---:|---:|---:|"]
    for model in MODELS:
        for method in ("normalized_rlcp_prediction_h1.5", "normalized_rlcp_curve_h1.5", "normalized_rlcp_prediction_h3", "normalized_rlcp_curve_h3"):
            row = value(model, method)
            lines.append(f"| {model} | {method} | {row.fraction_sets_with_infinite:.2%} | {row.fraction_cell_intervals_infinite:.2%} | {row.mean_ess:.1f} |")
    lines += ["", "例如曲线h=1.5几乎每组61块test都出现至少一个无穷区间，逐电芯比例约9%，不是每块电芯都无穷。h=3减少了这一问题但未消除。不能删掉这些区间只计算有限区间平均宽度，再宣称方法有效；完整平均宽度与interval score均为无穷。", "",
              "## 更关键的问题：整体90%仍掩盖剩余困难区域", "",
              "本批风险组根据开发集估计的**归一化残差风险**定义，与上轮绝对误差难度组不同。这里归一化只指除以指定尺度函数，并没有保证残差已独立于预测寿命。各设置均先固定组别，再比较各方法。下表是每个预测器设置内对组成员重复抽样后的覆盖，再跨9设置取平均；不是逐组统计保证。", "",
              "| 模型 | Split CP高风险组覆盖 | 开发寿命尺度CP | 直接寿命尺度CP | 曲线尺度CP |",
              "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        selected = group_table[(group_table.model == model) & (group_table.risk_group == 2)].set_index("method")
        vals = [selected.loc[m, "mean_per_fit_group_coverage"] for m in ("split_cp", "prediction_scale_cp", "relative_cp", "curve_scale_cp")]
        lines.append(f"| {model} | " + " | ".join(f"{v:.1%}" for v in vals) + " |")
    lines += ["", "这是平均宽度不能单独决定结论的直接证据：开发寿命尺度CP更窄、整体约90%，高风险组仍约78%。直接寿命尺度与曲线尺度能部分改善该区域；直接寿命尺度用更简单的构造就达到相近组覆盖，且总体interval score更低，必须保留为强对照。", "",
              "补充的事后诊断把同一个122块电芯池上的曲线风险估计与仅预测寿命的风险估计比较：CPMLP与归一化误差的平均Spearman分别为0.296/0.296，CPLSTM为0.346/0.330。曲线风险的正相关仍不能直接证明曲线提供了寿命之外的新信息；简单尺度估计可能未充分表达寿命与误差的关系。该诊断没有搜索参数或重训模型，明确属于查看结果后的解释性消融。", "",
              "本轮主要原始报告只有每组覆盖，未保存200次重分中每组的宽度/interval score，不能据这些汇总声称高风险组的条件效率也改善。若需补充，应明确为冻结预测上的追加诊断。", "",
              "## 下一步取舍", "",
              "保留全部18个预测器及本轮负结果，当前无需再扩展11模型或重复训练点预测器。先用同一组简单尺度基线在MATR冻结预测上复核，同时把困难区域覆盖列为主要诊断。", "",
              "若继续推进局部方法，应先检验为什么剩余高风险区域仍漏覆盖：20块开发残差的尺度/尾部估计是否不足，以及当前距离是否借到了不相似的残差。可在proper-training内用交叉拟合获取更多诚实残差，再研究小容量风险/尾部分布模型；calibration/test继续隔离。不要从这次结果直接跳到更复杂的表示或按test选择带宽。", "",
              "这批实验支持‘简单尺度适应确有价值’，尚不支持‘当前曲线距离已带来稳定LCP增益’。", "",
              "![Stability findings](stability_findings.png)"]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"REPORT={OUT / 'ANALYSIS.md'}")
    print(risk.groupby("model").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
