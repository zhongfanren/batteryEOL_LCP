"""Frozen-prediction factorial pilot: architecture gap as scale vs RLCP distance.

Exploratory after historical label exposure. This program does not import or
use the supervised risk estimators trained on historical calibration/test in
the preceding information audit. No point predictor is fitted or changed.
"""
import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE / "isu_stability_20261028"
DEFAULT_OUT = ROOT / "disagreement_cp_20261028"
PRIMARY = "s20261026_r2024_CPMLP"
PHASES = ("train", "dev", "calibration", "test")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def finite_quantile(scores):
    rank = math.ceil((len(scores) + 1) * .9)
    return np.partition(scores, rank - 1)[rank - 1] if rank <= len(scores) else np.inf


def random_weights(cal_z, test_z, h, normals):
    """Gaussian RLCP: same first random coordinate for 1D/2D comparisons."""
    center = test_z + h * normals[:, :test_z.shape[1]]
    distances = np.column_stack((np.sum((cal_z[None, :, :] - center[:, None, :])**2, axis=2),
                                 np.sum((test_z - center)**2, axis=1)))
    logs = -distances / (2 * h**2)
    weights = np.exp(logs - logs.max(axis=1, keepdims=True))
    self_mass = weights[:, -1] / weights.sum(axis=1)
    effective = weights[:, :-1].sum(axis=1)**2 / np.square(weights[:, :-1]).sum(axis=1)
    return weights, effective, self_mass


def weighted_quantiles(scores, weights):
    """One last weight per query belongs to a test atom at +infinity."""
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[:, :-1][:, order], axis=1)
    threshold = .9 * weights.sum(axis=1)
    # >= is the left quantile convention, including the infinity test atom.
    indices = np.sum(cumulative < threshold[:, None], axis=1)
    return np.r_[scores[order], np.inf][indices]


def construct(data, cohort, scales):
    """Only train covariates and dev labels may affect this frozen setup."""
    prediction = {p: np.maximum(0., data[f"{p}_prediction"]) for p in PHASES}
    gap = {}
    for p in PHASES:
        values = {}
        for key, item in cohort.items():
            assert np.array_equal(item[f"{p}_names"], data[f"{p}_names"])
            values.setdefault(key.rsplit("_", 1)[-1], []).append(item[f"{p}_prediction"])
        assert all(len(v) == 3 for v in values.values())
        gap[p] = np.abs(np.mean(values["CPMLP"], axis=0) - np.mean(values["CPLSTM"], axis=0))
    life = {p: np.maximum(1., prediction[p]) for p in PHASES}
    relative_gap = {p: gap[p] / life[p] for p in PHASES}
    raw_z = {p: np.column_stack((np.log(life[p]), np.log1p(relative_gap[p]))) for p in PHASES}
    scaler = StandardScaler().fit(raw_z["train"])
    z = {p: scaler.transform(raw_z[p]) for p in PHASES}
    # Pre-observation risk groups: thresholds use dev covariates, not errors.
    gap_cutpoints = np.quantile(relative_gap["dev"], [1/3, 2/3])
    life_cutpoints = np.quantile(prediction["dev"], [1/3, 2/3])
    groups = {p: {"gap": np.searchsorted(gap_cutpoints, relative_gap[p], side="right"),
                   "life": np.searchsorted(life_cutpoints, prediction[p], side="right")} for p in PHASES}
    dev_error = np.abs(data["dev_eol"] - prediction["dev"])
    dev_z = (prediction["dev"] - scales["prediction_mean"]) / scales["prediction_std"]
    learned_scale = {}
    for p in PHASES:
        query_z = (prediction[p] - scales["prediction_mean"]) / scales["prediction_std"]
        logs = -(query_z[:, None] - dev_z[None, :])**2 / 4.5
        w = np.exp(logs - logs.max(axis=1, keepdims=True))
        learned_scale[p] = np.maximum(max(1., .1 * np.median(dev_error)), w @ dev_error / w.sum(axis=1))
    return {"prediction": prediction, "gap": gap, "relative_gap": relative_gap, "z": z,
            "life_scale": life, "gap_scale": {p: life[p] + gap[p] for p in PHASES},
            "dev_scale": learned_scale, "groups": groups,
            "gap_cutpoints": gap_cutpoints.tolist(), "life_cutpoints": life_cutpoints.tolist(),
            "feature_mean": scaler.mean_.tolist(), "feature_std": scaler.scale_.tolist()}


def method_list():
    methods = ["split_cp", "prediction_scale_cp", "relative_cp", "gap_scale_cp"]
    for score in ("relative", "gap_scale"):
        for loc in ("life", "life_gap"):
            for h in (3., 6.):
                methods.append(f"{score}_rlcp_{loc}_h{h:g}")
    return methods


def intervals(bundle, y, ci, ti, repetition):
    pred = bundle["pool_prediction"]
    error = np.abs(y[ci] - pred[ci])
    q = finite_quantile(error)
    radii = {"split_cp": np.full(len(ti), q)}
    for name, scale in (("relative_cp", bundle["pool_life_scale"]),
                        ("gap_scale_cp", bundle["pool_gap_scale"]),
                        ("prediction_scale_cp", bundle["pool_dev_scale"])):
        radii[name] = finite_quantile(error / scale[ci]) * scale[ti]
    diagnostics = {}
    normals = np.random.default_rng(20261028 + repetition).normal(size=(len(ti), 2))
    for loc, dim in (("life", 1), ("life_gap", 2)):
        for h in (3., 6.):
            w, ess, mass = random_weights(bundle["pool_z"][ci, :dim], bundle["pool_z"][ti, :dim], h, normals)
            for score, scale in (("relative", bundle["pool_life_scale"]), ("gap_scale", bundle["pool_gap_scale"])):
                name = f"{score}_rlcp_{loc}_h{h:g}"
                radii[name] = weighted_quantiles(error / scale[ci], w) * scale[ti]
                diagnostics[name] = {"mean_ess": ess.mean(), "mean_self_mass": mass.mean(),
                                     "self_mass_above_alpha": int((mass > .1).sum())}
    return radii, diagnostics


def cell_metrics(y, prediction, radius):
    lower, upper = np.maximum(0., prediction - radius), prediction + radius
    width = upper - lower
    covered = (lower <= y) & (y <= upper)
    score = width + 20 * (np.maximum(lower - y, 0.) + np.maximum(y - upper, 0.))
    return lower, upper, width, covered, score


def add_group(accum, key, keep, width, covered, score, pred):
    if not keep.any():
        return
    values = np.array([keep.sum(), covered[keep].sum(), width[keep].sum(), score[keep].sum(),
                       np.sum(width[keep] / np.maximum(1., pred[keep]))], dtype=float)
    accum[key] = accum.get(key, np.zeros(5)) + values


def run_job(spec, loaded, repetitions):
    source = Path(spec["source"])
    data = loaded[spec["key"]]
    cohort = {k: v for k, v in loaded.items() if k.startswith(f"s{spec['split_seed']}_")}
    scales = read(source / spec["model"] / "results.json")["scales"]
    setup = construct(data, cohort, scales)
    for name in ("prediction", "gap", "relative_gap", "z", "life_scale", "gap_scale", "dev_scale"):
        setup[f"pool_{name}"] = np.concatenate([setup[name][p] for p in ("calibration", "test")])
    y = np.r_[data["calibration_eol"], data["test_eol"]]
    pred = setup["pool_prediction"]
    names = np.r_[data["calibration_names"], data["test_names"]]
    group_values = {axis: np.r_[setup["groups"]["calibration"][axis], setup["groups"]["test"][axis]] for axis in ("gap", "life")}
    large = np.abs(y - pred) / np.maximum(1., pred) > .5
    rng = np.random.default_rng(20261028)
    metrics, original, support, allocation = [], [], [], {}
    for axis, values in group_values.items():
        for group in range(3):
            keep = values == group
            support.append({"job": spec["key"], "model": spec["model"], "axis": axis, "group": group,
                            "pool_n": int(keep.sum()), "mean_absolute_error": np.abs(y[keep] - pred[keep]).mean() if keep.any() else np.nan,
                            "mean_relative_error": np.mean(np.abs(y[keep] - pred[keep]) / np.maximum(1., pred[keep])) if keep.any() else np.nan,
                            "large_error_fraction": large[keep].mean() if keep.any() else np.nan})
    for repetition in range(repetitions + 1):
        permutation = np.arange(122) if repetition == 0 else rng.permutation(122)
        ci, ti = permutation[:61], permutation[61:]
        radii, diagnostic = intervals(setup, y, ci, ti, repetition)
        for method, radius in radii.items():
            lower, upper, width, covered, score = cell_metrics(y[ti], pred[ti], radius)
            metrics.append({"job": spec["key"], "model": spec["model"], "split_seed": spec["split_seed"],
                            "train_seed": spec["train_seed"], "method": method, "repetition": repetition,
                            "coverage": covered.mean(), "mean_width": width.mean(), "interval_score": score.mean(),
                            "relative_width": np.mean(width / np.maximum(1., pred[ti])),
                            "relative_interval_score": np.mean(score / np.maximum(1., pred[ti])),
                            "infinite": int(np.isinf(radius).sum()), **diagnostic.get(method, {})})
            phase = "original" if repetition == 0 else "resplits"
            for axis, values in group_values.items():
                for group in range(3):
                    add_group(allocation, (phase, method, axis, group), values[ti] == group, width, covered, score, pred[ti])
            add_group(allocation, (phase, method, "retrospective_large_error", 1), large[ti], width, covered, score, pred[ti])
            if repetition == 0:
                for j, i in enumerate(ti):
                    original.append({"job": spec["key"], "method": method, "cell_file": names[i], "prediction": pred[i], "eol": y[i],
                                     "gap": setup["pool_gap"][i], "relative_gap": setup["pool_relative_gap"][i],
                                     "gap_group": group_values["gap"][i], "life_group": group_values["life"][i],
                                     "large_error": bool(large[i]), "radius": radius[j], "lower": lower[j], "upper": upper[j],
                                     "covered": bool(covered[j]), "width": width[j], "interval_score": score[j]})
    groups = []
    for (phase, method, axis, group), values in allocation.items():
        n, covered, width, score, relative_width = values
        groups.append({"job": spec["key"], "model": spec["model"], "phase": phase, "method": method,
                       "axis": axis, "group": group, "presentations": int(n), "coverage": covered / n,
                       "mean_width": width / n, "interval_score": score / n, "relative_width": relative_width / n})
    return metrics, original, groups, support, {k: setup[k] for k in ("gap_cutpoints", "life_cutpoints", "feature_mean", "feature_std")}


def summarize(metrics, groups):
    results, paired = [], []
    for (job, method), rows in metrics[metrics.repetition > 0].groupby(["job", "method"]):
        global_name = "gap_scale_cp" if method.startswith("gap_scale_rlcp") else "relative_cp"
        controls = metrics[(metrics.job == job) & (metrics.method == global_name) & (metrics.repetition > 0)].set_index("repetition").loc[rows.repetition]
        # infinity - infinity is not a finite improvement and is not averaged.
        row = {"job": job, "model": rows.model.iloc[0], "method": method, "score_control": global_name,
               "mean_coverage": rows.coverage.mean(), "mean_width": rows.mean_width.mean(), "mean_interval_score": rows.interval_score.mean(),
               "mean_relative_width": rows.relative_width.mean(), "mean_relative_interval_score": rows.relative_interval_score.mean(),
               "infinite_cell_fraction": rows.infinite.mean() / 61, "infinite_set_fraction": (rows.infinite > 0).mean(),
               "coverage_delta_pp": 100 * (rows.coverage.to_numpy() - controls.coverage.to_numpy()).mean()}
        for metric in ("mean_width", "interval_score"):
            a, b = rows[metric].to_numpy(), controls[metric].to_numpy()
            row[f"{metric}_gain_pct"] = 100 * (1 - a.mean() / b.mean()) if np.isfinite(a).all() and np.isfinite(b).all() else np.nan
        results.append(row)
    by_job = pd.DataFrame(results)
    for (model, method), rows in by_job.groupby(["model", "method"]):
        row = {"model": model, "method": method, "fits": len(rows)}
        for name in ("mean_coverage", "mean_width", "mean_interval_score", "mean_relative_width", "mean_relative_interval_score",
                     "infinite_cell_fraction", "infinite_set_fraction", "coverage_delta_pp", "mean_width_gain_pct", "interval_score_gain_pct"):
            values = rows[name].to_numpy()
            # Never silently drop NaN gains from fits whose full mean is infinite.
            row[name] = np.mean(values)
        row["width_wins_vs_score_control"] = int((rows.mean_width_gain_pct > 0).sum())
        row["score_wins_vs_score_control"] = int((rows.interval_score_gain_pct > 0).sum())
        paired.append(row)
    return by_job, pd.DataFrame(paired)


def report(out, metrics, by_job, aggregate, groups, support):
    lines = ["# 分歧用于尺度还是LCP距离：冻结预测试验", "", "本轮没有重新训练点预测器，单模型中心保持不变。",
             "本研究方向来自历史结果，以下仍为探索性；本轮不使用此前在历史cal/test标签上拟合的风险模型。",
             "方法公式不拟合cal/test标签，校准标签只计算当前分位数；历史标签暴露意味着这些不是新的确认性覆盖证据。", "",
             "## 固定构造", "", "p=max(1,当前预测寿命)，d=|三个CPMLP均值−三个CPLSTM均值|。",
             "relative_cp尺度=p；gap_scale_cp尺度=p+d，系数预设为1，不调参；prediction_scale_cp沿用20块dev拟合的旧寿命尺度对照。",
             "RLCP距离：log(p)一维，或[log(p),log(1+d/p)]二维，均只在原train97上拟合标准化。",
             "每一种校准分数都分别配全局校准、仅寿命RLCP、寿命＋分歧RLCP；高斯随机中心，测试权重位于+∞，h=3和6全报告。",
             "RLCP随机第一坐标在一维/二维对照中共用；尺度变化对照共用完整局部权重。不能将局部化与换评分混为一项效果。",
             "每设置固定122块历史池，原61/61及200次随机重分；重分不产生新独立电芯。主设置仍是CPMLP seed2024、原划分。", "",
             "## 主设置：原61块test", "", "| 方法 | 覆盖 | 宽度/周期 | Interval score | 无穷数 |", "|---|---:|---:|---:|---:|"]
    primary = metrics[(metrics.job == PRIMARY) & (metrics.repetition == 0)]
    for r in primary.itertuples():
        lines.append(f"| {r.method} | {r.coverage:.1%} | {r.mean_width:.1f} | {r.interval_score:.1f} | {r.infinite} |")
    lines += ["", "## 全部设置与重分", "", "均先在每个设置内平均200次，再对各模型9个设置等权平均。任何无穷值都保留；不计算‘只保留有限区间’的效率结论。", "",
              "| 模型 | 方法 | 覆盖 | 宽度 | Interval score | 无穷电芯比例 |", "|---|---|---:|---:|---:|---:|"]
    for r in aggregate.itertuples():
        lines.append(f"| {r.model} | {r.method} | {r.mean_coverage:.2%} | {r.mean_width:.1f} | {r.mean_interval_score:.1f} | {r.infinite_cell_fraction:.3%} |")
    lines += ["", "## 分组解释", "", "gap组阈值是20块dev的d/p三等分位点；life组是dev预测寿命三等分位点，均不使用误差。",
              "高分歧组只是可提前观察的分歧分组，不能先验称为真实高难度；支持数据见group_support.csv中的实际误差与大误差比例。",
              "group_metrics.csv保存每组覆盖、宽度与score；presentations为重分中的累计出现次数，不是独立样本量。",
              "retrospective_large_error组用真实相对误差>50%定义，只做事后漏覆盖诊断，不是可部署分组，也没有该组90%覆盖保证。", "",
              "具体归因对照见paired_by_job.csv与aggregate.csv：gap_scale_cp应与relative_cp比较；RLCP应与同score的全局方法比较。",
              "只改善原test、某带宽或某个模型不算稳定收益；先报告全体正负结果，再决定是否值得获取更多诚实风险监督。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise FileExistsError("Use a new output directory to retain frozen plans")
    write(out / "plan.json", {"created_at": stamp(), "code_sha256": digest(__file__), "primary": PRIMARY,
                              "hypotheses": ["Hscale: additive architecture-gap scale improves allocation beyond lifetime alone",
                                             "Hlocal: adding gap to RLCP geometry helps beyond lifetime localization, with the same score"],
                              "methods": method_list(), "bandwidths": [3, 6], "gap_scale": "max(1,prediction)+absolute architecture mean gap; coefficient1 fixed",
                              "geometry": "train97 standardized log prediction and log1p(gap/max(1,prediction))",
                              "grouping": "dev20 covariate tertiles; no errors used", "repetitions": args.repetitions,
                              "seed": 20261028, "point_training": False, "supervised_historical_risk_model_reused": False,
                              "analysis": "same-score global/local pairs, all 18 fits and h settings; retain infinity; no outcome-based choice",
                              "interpretation": "post-hoc historical exploration; no new confirmatory guarantee or per-group90% claim"})
    write(out / "state.json", {"state": "running", "pid": os.getpid(), "updated_at": stamp()})
    plan = read(ROOT / "plan.json")
    loaded, inputs = {}, []
    for spec in plan["jobs"]:
        source = Path(spec["source"])
        model = source / spec["model"]
        frozen = read(model / "frozen_predictor.json")
        assert digest(model / "best_model.pt") == frozen["checkpoint_sha256"]
        assert digest(source / "dataset_manifest.json") == frozen["manifest_sha256"]
        path = ROOT / "analysis" / spec["key"] / "features.npz"
        loaded[spec["key"]] = dict(np.load(path))
        manifest = read(source / "dataset_manifest.json")
        phases = manifest["split"]["phases"]
        assert len(set(sum(phases.values(), []))) == 239
        for p in PHASES:
            assert loaded[spec["key"]][f"{p}_names"].tolist() == phases[p]
        for p in (path, model / "best_model.pt", model / "results.json", source / "dataset_manifest.json"):
            inputs.append({"path": str(p), "sha256": digest(p)})
    rows, originals, group_rows, support_rows, construction = [], [], [], [], {}
    for index, spec in enumerate(plan["jobs"]):
        print(f"EXPERIMENT {index + 1}/18 {spec['key']}", flush=True)
        a, b, c, d, e = run_job(spec, loaded, args.repetitions)
        rows.extend(a); originals.extend(b); group_rows.extend(c); support_rows.extend(d); construction[spec["key"]] = e
        write(out / "state.json", {"state": "running", "completed_jobs": index + 1, "total_jobs": 18, "updated_at": stamp()})
    metrics, groups, support = pd.DataFrame(rows), pd.DataFrame(group_rows), pd.DataFrame(support_rows)
    by_job, aggregate = summarize(metrics, groups)
    for name, table in (("metrics", metrics), ("original_intervals", pd.DataFrame(originals)), ("group_metrics", groups),
                        ("group_support", support), ("paired_by_job", by_job), ("aggregate", aggregate)):
        table.to_csv(out / f"{name}.csv", index=False)
    write(out / "construction.json", construction)
    assert all(digest(r["path"]) == r["sha256"] for r in inputs)
    write(out / "audit.json", {"inputs_unchanged": True, "inputs": inputs, "all_methods_reported": True, "no_point_training": True})
    report(out, metrics, by_job, aggregate, groups, support)
    write(out / "state.json", {"state": "complete", "completed_jobs": 18, "updated_at": stamp(), "report": str(out / "REPORT.md")})
    print(f"COMPLETE {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
