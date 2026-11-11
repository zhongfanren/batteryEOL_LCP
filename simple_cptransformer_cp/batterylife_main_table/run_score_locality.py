"""Predeclared score/locality ablation on frozen ISU predictions; CPU only.

Train-only OOF risk diagnostics never select a method using holdout labels.
The interval experiment uses Hore--Barber Gaussian RLCP, not legacy baseLCP.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler

import run_isu_disagreement_cp as cp

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "score_locality_20261104"
OOF_ROOT = HERE / "isu_crossfit_risk_20261028"
SCORES = ("absolute", "relative", "gap6")
LOCATIONS = ("global", "life", "gap")
METHODS = [f"{s}__{loc}" for s in SCORES for loc in LOCATIONS]
H = 6.


def save(path, obj):
    cp.write(Path(path), obj)


def prepare(data, cohort):
    """Covariates only. No phase's outcome is accessed, including development."""
    prediction, gap, scales, raw = {}, {}, {}, {}
    for phase in cp.PHASES:
        prediction[phase] = np.maximum(0., data[f"{phase}_prediction"])
        by_arch = {m: [] for m in ("CPMLP", "CPLSTM")}
        for key, item in cohort.items():
            assert np.array_equal(item[f"{phase}_names"], data[f"{phase}_names"])
            by_arch[key.rsplit("_", 1)[-1]].append(item[f"{phase}_prediction"])
        assert all(len(v) == 3 for v in by_arch.values())
        gap[phase] = np.abs(np.mean(by_arch["CPMLP"], axis=0) - np.mean(by_arch["CPLSTM"], axis=0))
        p = np.maximum(1., prediction[phase])
        scales[phase] = np.array([np.ones(len(p)), p, p + gap[phase]])
        raw[phase] = np.column_stack((np.log(p), np.log1p(gap[phase] / p)))
    scaler = StandardScaler().fit(raw["train"])
    z = {phase: scaler.transform(value) for phase, value in raw.items()}
    cuts = {axis: np.quantile(raw["dev"][:, k], [1/3, 2/3]) for k, axis in enumerate(("life", "gap"))}
    groups = {phase: {axis: np.searchsorted(cuts[axis], value[:, k], side="right")
                      for k, axis in enumerate(("life", "gap"))} for phase, value in raw.items()}
    return {"prediction": prediction, "gap": gap, "scales": scales, "z": z, "groups": groups,
            "parameters": {"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist(),
                           "cutpoints": {k: v.tolist() for k, v in cuts.items()}}}


def get_weights(z, ci, ti, repetition):
    # Preserve the first-coordinate draws of the previous 1D/2D ablation.
    normals = np.random.default_rng(20261028 + repetition).normal(size=(len(ti), 2))[:, :1]
    weights = {"global": np.ones((len(ti), len(ci) + 1))}
    for k, axis in enumerate(("life", "gap")):
        weights[axis] = cp.random_weights(z[ci, k:k+1], z[ti, k:k+1], H, normals)[0]
    return {k: v / v.sum(axis=1, keepdims=True) for k, v in weights.items()}, normals


def construct_intervals(prediction, scales, truth, ci, ti, weights):
    """Only truth[ci] is inspected. The last weight is the +infinity atom."""
    error = np.abs(truth[ci] - prediction[ci])
    result = {}
    for k, score in enumerate(SCORES):
        cal_scores = error / scales[k, ci]
        for loc in LOCATIONS:
            q = cp.weighted_quantiles(cal_scores, weights[loc])
            result[f"{score}__{loc}"] = q * scales[k, ti]
            if loc == "global":
                np.testing.assert_allclose(q, cp.finite_quantile(cal_scores))
    return result


def raw_features(p, gap):
    return np.column_stack((np.log(np.maximum(1., p)), np.log1p(gap / np.maximum(1., p))))


def risk_fit(p, gap, y, query_p, query_gap, use_gap, alpha):
    target = np.log(np.maximum(.02, np.abs(y - p) / np.maximum(1., p)))
    if use_gap is None:
        value = float(np.quantile(target, .9))
        return np.full(len(query_p), value), {"intercept": value, "coefficient": []}
    scaler = StandardScaler().fit(raw_features(p, gap))
    def design(pred, g):
        z = scaler.transform(raw_features(pred, g))
        return np.column_stack((z[:, 0], z[:, 0]**2, z[:, 1])) if use_gap else np.column_stack((z[:, 0], z[:, 0]**2))
    estimator = QuantileRegressor(quantile=.9, alpha=alpha, solver="highs").fit(design(p, gap), target)
    prediction = np.clip(estimator.predict(design(query_p, query_gap)), -4., 2.)
    return prediction, {"coefficient": estimator.coef_.tolist(), "intercept": float(estimator.intercept_),
                        "feature_mean": scaler.mean_.tolist(), "feature_std": scaler.scale_.tolist()}


def train_diagnostic(out, oof, train_names):
    variants = {"constant": (None, 0.), "life_a01": (False, .01),
                "pair_a05_old": (True, .05), "pair_a01_primary": (True, .01),
                "pair_a0_sensitivity": (True, 0.)}
    rows, predictions, parameters = [], [], {}
    for model in ("CPMLP", "CPLSTM"):
        tables = {m: oof[oof.model == m].set_index("cell_file").loc[train_names] for m in ("CPMLP", "CPLSTM")}
        table = tables[model]
        p, y, fold = np.maximum(0., table.prediction.to_numpy()), table.eol.to_numpy(), table.fold.to_numpy()
        np.testing.assert_array_equal(tables["CPMLP"].fold, tables["CPLSTM"].fold)
        gap = np.abs(np.maximum(0., tables["CPMLP"].prediction.to_numpy()) - np.maximum(0., tables["CPLSTM"].prediction.to_numpy()))
        target = np.log(np.maximum(.02, np.abs(y-p) / np.maximum(1., p)))
        parameters[model] = {}
        for name, (use_gap, alpha) in variants.items():
            heldout = np.full(len(p), np.nan)
            for f in sorted(set(fold)):
                fit, val = fold != f, fold == f
                values, pars = risk_fit(p[fit], gap[fit], y[fit], p[val], gap[val], use_gap, alpha)
                heldout[val] = values
                rows.append({"model": model, "variant": name, "fold": int(f), "n": int(val.sum()),
                             "pinball": mean_pinball_loss(target[val], values, alpha=.9),
                             "nonzero_coefficients": int(np.count_nonzero(np.abs(pars["coefficient"]) > 1e-10))})
            full, pars = risk_fit(p, gap, y, p, gap, use_gap, alpha)
            parameters[model][name] = {**pars, "full_fit_prediction_std": float(full.std())}
            for i, cell in enumerate(train_names):
                predictions.append({"model": model, "variant": name, "cell_file": cell, "fold": int(fold[i]),
                                    "target_log_relative_error": target[i], "crossvalidated_log_risk": heldout[i]})
    folds, pred = pd.DataFrame(rows), pd.DataFrame(predictions)
    folds.to_csv(out / "train_risk_folds.csv", index=False)
    pred.to_csv(out / "train_risk_predictions.csv", index=False)
    decisions, summary = {}, []
    for model in ("CPMLP", "CPLSTM"):
        part = folds[folds.model == model]
        pivot = part.pivot(index="fold", columns="variant", values="pinball")
        losses = {}
        for variant in variants:
            p = pred[(pred.model == model) & (pred.variant == variant)]
            losses[variant] = mean_pinball_loss(p.target_log_relative_error, p.crossvalidated_log_risk, alpha=.9)
            summary.append({"model": model, "variant": variant, "cv_pinball": losses[variant],
                            "fold_wins_vs_constant": int((pivot[variant] < pivot.constant).sum()),
                            "fold_wins_vs_life": int((pivot[variant] < pivot.life_a01).sum()),
                            "nonconstant_folds": int((part[part.variant == variant].nonzero_coefficients > 0).sum()),
                            "full_fit_prediction_std": parameters[model][variant]["full_fit_prediction_std"]})
        primary = "pair_a01_primary"
        nonconstant = int((part[part.variant == primary].nonzero_coefficients > 0).sum())
        gate = (losses[primary] <= .98 * losses["constant"] and losses[primary] <= .98 * losses["life_a01"]
                and (pivot[primary] < pivot.constant).sum() >= 3 and (pivot[primary] < pivot.life_a01).sum() >= 3
                and nonconstant >= 4 and parameters[model][primary]["full_fit_prediction_std"] > .01)
        decisions[model] = {"advance_learned_tail": bool(gate), "primary_variant": primary,
                            "relative_pinball_gain_vs_constant": 1-losses[primary]/losses["constant"],
                            "relative_pinball_gain_vs_life": 1-losses[primary]/losses["life_a01"]}
    pd.DataFrame(summary).to_csv(out / "train_risk_summary.csv", index=False)
    save(out / "train_risk_models.json", parameters)
    save(out / "train_risk_decision.json", {"decisions": decisions,
         "interpretation": "Training-side model-selection diagnostic, not nested end-to-end validation. Auxiliary OOF models overlap and may have seen another risk fold's labels. No cal/test used; no new independence claim.",
         "action": "No learned score or distance is added in this batch, regardless of gate. A passing gate warrants a separate frozen follow-up."})
    return decisions


def evaluate_job(spec, loaded, repetitions, out):
    data = loaded[spec["key"]]
    cohort = {k: v for k, v in loaded.items() if k.startswith(f"s{spec['split_seed']}_")}
    setup = prepare(data, cohort)
    prediction = np.concatenate([setup["prediction"][p] for p in ("calibration", "test")])
    truth = np.r_[data["calibration_eol"], data["test_eol"]]
    names = np.r_[data["calibration_names"], data["test_names"]]
    scales = np.concatenate([setup["scales"][p] for p in ("calibration", "test")], axis=1)
    z = np.concatenate([setup["z"][p] for p in ("calibration", "test")])
    groups = {axis: np.r_[setup["groups"]["calibration"][axis], setup["groups"]["test"][axis]] for axis in ("life", "gap")}
    rng = np.random.default_rng(20261028)
    metrics, original, allocations = [], [], {}
    for rep in range(repetitions + 1):
        order = np.arange(len(names)) if rep == 0 else rng.permutation(len(names))
        ci, ti = order[:61], order[61:]
        weights, normals = get_weights(z, ci, ti, rep)
        radii = construct_intervals(prediction, scales, truth, ci, ti, weights)
        if rep == 0 and spec["key"] == cp.PRIMARY:
            np.savez_compressed(out / "primary_intermediates.npz", cal_names=names[ci], test_names=names[ti],
                                cal_prediction=prediction[ci], test_prediction=prediction[ti], cal_eol=truth[ci], test_eol=truth[ti],
                                cal_scales=scales[:, ci], test_scales=scales[:, ti],
                                cal_scores=np.abs(truth[ci]-prediction[ci])[None, :] / scales[:, ci],
                                weights=np.array([weights[loc] for loc in LOCATIONS]), normals=normals,
                                cal_coordinates=z[ci], test_coordinates=z[ti])
        for method, radius in radii.items():
            score_name, loc = method.split("__")
            lower, upper, width, covered, score = cp.cell_metrics(truth[ti], prediction[ti], radius)
            w = weights[loc]
            ess = (1-w[:, -1])**2 / np.square(w[:, :-1]).sum(axis=1)
            metrics.append({"job": spec["key"], "model": spec["model"], "method": method, "repetition": rep,
                            "coverage": covered.mean(), "mean_width": width.mean(), "interval_score": score.mean(),
                            "relative_width": np.mean(width/np.maximum(1., prediction[ti])),
                            "relative_interval_score": np.mean(score/np.maximum(1., prediction[ti])),
                            "infinite": int(np.isinf(radius).sum()), "mean_ess": ess.mean(), "mean_self_mass": w[:, -1].mean()})
            for axis, group in groups.items():
                for g in range(3):
                    cp.add_group(allocations, ("original" if rep == 0 else "resplits", method, axis, g),
                                 group[ti] == g, width, covered, score, prediction[ti])
            if rep == 0:
                for j, i in enumerate(ti):
                    original.append({"job": spec["key"], "model": spec["model"], "method": method, "cell_file": names[i],
                                     "prediction": prediction[i], "eol": truth[i], "radius": radius[j], "lower": lower[j],
                                     "upper": upper[j], "width": width[j], "covered": bool(covered[j]), "interval_score": score[j],
                                     "gap_group": groups["gap"][i], "life_group": groups["life"][i]})
    group_rows = []
    for (phase, method, axis, group), (n, cov, width, score, relwidth) in allocations.items():
        group_rows.append({"job": spec["key"], "model": spec["model"], "phase": phase, "method": method, "axis": axis,
                           "group": group, "presentations": int(n), "coverage": cov/n, "mean_width": width/n,
                           "interval_score": score/n, "relative_width": relwidth/n})
    return metrics, original, group_rows, setup["parameters"]


def summarize(metrics, out):
    rows = []
    for (job, method), part in metrics[metrics.repetition > 0].groupby(["job", "method"]):
        rows.append({"job": job, "model": part.model.iloc[0], "method": method,
                     **{k: float(np.mean(part[k].to_numpy())) for k in ("coverage", "mean_width", "interval_score", "relative_width", "relative_interval_score", "mean_ess", "mean_self_mass")},
                     "infinite_cell_fraction": float(part.infinite.mean()/61)})
    by_job = pd.DataFrame(rows)
    by_job.to_csv(out / "by_job.csv", index=False)
    rows = []
    for (model, method), part in by_job.groupby(["model", "method"]):
        rows.append({"model": model, "method": method, "fits": len(part),
                     **{k: float(np.mean(part[k].to_numpy())) for k in part.columns if k not in ("job", "model", "method")}})
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(out / "aggregate.csv", index=False)
    pairs = []
    for (job, method), part in metrics[metrics.repetition > 0].groupby(["job", "method"]):
        base = metrics[(metrics.job == job) & (metrics.method == method.split("__")[0] + "__global") & (metrics.repetition > 0)].set_index("repetition").loc[part.repetition]
        pairs.append({"job": job, "model": part.model.iloc[0], "method": method,
                      "coverage_delta_pp": float(100*np.mean(part.coverage.to_numpy()-base.coverage.to_numpy())),
                      "width_gain_pct": float(100*(1-np.mean(part.mean_width.to_numpy())/np.mean(base.mean_width.to_numpy()))),
                      "interval_score_gain_pct": float(100*(1-np.mean(part.interval_score.to_numpy())/np.mean(base.interval_score.to_numpy())))})
    pd.DataFrame(pairs).to_csv(out / "same_score_pairs.csv", index=False)
    return aggregate


def verify(out, metrics, original):
    # Independent order-statistic baseline and existing independently saved results.
    prior = pd.read_csv(cp.ROOT / "disagreement_cp_20261028/metrics.csv").set_index(["job", "method", "repetition"])
    mappings = {"absolute__global": "split_cp", "relative__global": "relative_cp", "gap6__global": "gap_scale_cp",
                "relative__life": "relative_rlcp_life_h6", "gap6__life": "gap_scale_rlcp_life_h6"}
    maxdiff = 0.
    for name, previous in mappings.items():
        part = metrics[metrics.method == name]
        ref = prior.loc[pd.MultiIndex.from_arrays([part.job, [previous]*len(part), part.repetition])]
        for column in ("coverage", "mean_width", "interval_score"):
            a, b = part[column].to_numpy(), ref[column].to_numpy()
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-7)
            maxdiff = max(maxdiff, float(np.max(np.abs(a-b))))
    np.testing.assert_allclose(original.width, original.upper-original.lower)
    np.testing.assert_array_equal(original.covered, (original.lower <= original.eol) & (original.eol <= original.upper))
    np.testing.assert_allclose(original.interval_score, original.width+20*(np.maximum(original.lower-original.eol, 0)+np.maximum(original.eol-original.upper, 0)))
    return {"existing_method_replay_max_absolute_difference": maxdiff, "existing_methods_replayed": list(mappings),
            "original_interval_rows_checked": len(original)}


def report(out, aggregate, decisions):
    lines = ["# 分数 × 局部性：第一轮", "", "本轮不训练点预测器，不改变数据集或划分。三种分数 × 三种校准构造全部报告，不按历史test选冠军。",
             "", "## 训练侧尾部模型诊断", "", "主要候选固定为pair alpha=.01；alpha=.05复现退化，alpha=0只检查去正则后的敏感性。",
             "门槛：相对常数及同alpha寿命only的CV pinball均改善至少2%，各至少3/5折获胜，至少4折非恒定，全量预测std>.01。",
             "本诊断复用固定OOF库，辅助模型训练集重叠，不是端到端嵌套验证。无论门槛结果，本批均不追加学习距离，不按历史test调参。", ""]
    for model, decision in decisions.items():
        lines.append(f"- {model}：进入下一轮学习尾部的门槛{'通过' if decision['advance_learned_tail'] else '未通过'}；对常数pinball改善{decision['relative_pinball_gain_vs_constant']:.1%}，对寿命only改善{decision['relative_pinball_gain_vs_life']:.1%}。")
    lines += ["", "## 所有18个冻结预测器设置，历史池200次重分平均", "", "|预测器|分数__局部性|覆盖|宽度|Interval score|无穷比例|", "|---|---|---:|---:|---:|---:|"]
    for row in aggregate.itertuples():
        lines.append(f"|{row.model}|{row.method}|{row.coverage:.2%}|{row.mean_width:.1f}|{row.interval_score:.1f}|{row.infinite_cell_fraction:.3%}|")
    lines += ["", "absolute=绝对误差；relative=误差/预测寿命尺度；gap6=误差/(预测寿命尺度+六模型架构均值分歧)。",
              "global=普通split CP；life=只按log预测寿命局部化；gap=只按log1p(六模型分歧/预测寿命尺度)局部化。两个坐标均仅用train97标准化；h=6固定。",
              "局部构造沿用Hore–Barber Gaussian RLCP，随机中心与测试+∞原子保留。不是旧MATR32/34阈值或Guan校正LCP的直接实现。",
              "", "## 怎样阅读结果", "", "先比较relative__global与gap6__global：这是分数/尺度增量；再逐分数比较life/gap与global：这是局部校准增量。",
              "same_score_pairs.csv保存逐冻结设置的配对增益；group_metrics.csv保存按dev可观察变量三等分的组覆盖与成本。高分歧组不预先等同真实高难度，需查看真实误差。",
              "original_intervals.csv完整保留原test，不能用重分均值掩盖。所有无穷保留。历史test已用于探索，200次重分、18设置均有重叠，不是独立新数据或每组90%保证。",
              "", "## CDF诊断", "", "primary_intermediates.npz实际保存全部校准预测、分数、query权重、测试+∞质量及随机中心噪声，不需反演。",
              "绘图程序plot_score_locality.py输出主CPMLP2024全部61块原test的三份逐分数PDF、逐电芯区间图、CDF全点与归因CSV。CDF横轴是该分数，不一定是周期；纵轴含测试+∞质量的归一化，红线90%。",
              "", "参考：[Guan LCP](https://arxiv.org/html/2106.08460v2)；[Hore–Barber RLCP](https://arxiv.org/html/2310.07850)；[Lei等 normalized CP](https://arxiv.org/html/1604.04173#S5.SS2)。"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("At least one resplit is required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise FileExistsError("Use a new output directory; existing experiment remains frozen")
    plan = {"created_at": cp.stamp(), "script_sha256": cp.digest(__file__), "dependency_sha256": cp.digest(cp.__file__),
            "primary": cp.PRIMARY, "hypotheses": ["Htail: fixed weaker-regularized tail model generalizes on train-only OOF diagnostics beyond lifetime-only", "Hlocal: same-score 1D gap RLCP adds value beyond global and life RLCP"],
            "score_scales": ["1", "max(1,prediction)", "max(1,prediction)+gap6"], "methods": METHODS,
            "locality": "train-standardized log prediction or log1p(gap6/p); 1D Gaussian RLCP; h=6 fixed",
            "risk_variants": ["constant", "life_a01", "pair_a05_old", "pair_a01_primary", "pair_a0_sensitivity"],
            "risk_gate": "Primary pair_a01: pooled CV pinball >=2% better than BOTH constant and life_a01; >=3/5 fold wins against EACH; >=4/5 nonconstant folds; full predicted std>.01. No downstream model selection in this batch.",
            "risk_cv": "leave original auxiliary outer-fold out; scaler refit per risk fold; fixed OOF overlap precludes end-to-end nested validation",
            "repetitions": args.repetitions, "resplit_seed": 20261028, "random_center_seed": 20261028,
            "readout": "All same-score pairs, coverage, width, interval score, predefined gap/life groups, infinity and per-cell CDFs; never pick a winner using test",
            "point_training": False, "dataset_change": False, "exploratory": True,
            "plot_selection": "All primary original test61, all three scores; no error-based cell selection"}
    save(out / "plan.json", plan)
    save(out / "state.json", {"state": "running", "stage": "verify_inputs", "pid": os.getpid(), "updated_at": cp.stamp()})
    try:
        inputs = []
        def track(path):
            path = Path(path)
            inputs.append({"path": str(path), "sha256": cp.digest(path)})
        track(__file__); track(cp.__file__); track(cp.ROOT / "plan.json")
        parent = cp.read(cp.ROOT / "plan.json")
        oof_plan = cp.read(OOF_ROOT / "plan.json")
        assert cp.digest(OOF_ROOT / "dataset_manifest.json") == oof_plan["manifest_sha256"]
        for dependency in oof_plan["dependencies"]:
            assert cp.digest(dependency["path"]) == dependency["sha256"]
        track(OOF_ROOT / "plan.json"); track(OOF_ROOT / "oof_predictions.csv"); track(OOF_ROOT / "dataset_manifest.json")
        phases = cp.read(OOF_ROOT / "dataset_manifest.json")["split"]["phases"]
        oof = pd.read_csv(OOF_ROOT / "oof_predictions.csv")
        assert len(oof) == 194
        for model in ("CPMLP", "CPLSTM"):
            part = oof[oof.model == model]
            assert len(part) == part.cell_file.nunique() == 97
            assert set(part.cell_file) == set(phases["train"])
        print("STAGE A: train-only OOF risk diagnostic", flush=True)
        decisions = train_diagnostic(out, oof, phases["train"])
        # Freeze training-side diagnosis before any interval evaluation.
        loaded = {}
        for spec in parent["jobs"]:
            source = Path(spec["source"])
            model_dir = source / spec["model"]
            frozen = cp.read(model_dir / "frozen_predictor.json")
            assert cp.digest(model_dir / "best_model.pt") == frozen["checkpoint_sha256"]
            assert cp.digest(source / "dataset_manifest.json") == frozen["manifest_sha256"]
            path = cp.ROOT / "analysis" / spec["key"] / "features.npz"
            loaded[spec["key"]] = dict(np.load(path, allow_pickle=False))
            manifest = cp.read(source / "dataset_manifest.json")["split"]["phases"]
            assert len(set(sum(manifest.values(), []))) == 239
            for phase in cp.PHASES:
                assert loaded[spec["key"]][f"{phase}_names"].tolist() == manifest[phase]
            for path in (path, model_dir / "best_model.pt", model_dir / "frozen_predictor.json", source / "dataset_manifest.json"):
                track(path)
        rows, cells, groups, construction = [], [], [], {}
        for i, spec in enumerate(parent["jobs"]):
            print(f"STAGE B: {i+1}/{len(parent['jobs'])} {spec['key']}", flush=True)
            a, b, c, parameters = evaluate_job(spec, loaded, args.repetitions, out)
            rows.extend(a); cells.extend(b); groups.extend(c); construction[spec["key"]] = parameters
            save(out / "state.json", {"state": "running", "stage": "intervals", "completed_jobs": i+1,
                 "total_jobs": len(parent["jobs"]), "pid": os.getpid(), "updated_at": cp.stamp()})
        metrics, original = pd.DataFrame(rows), pd.DataFrame(cells)
        for name, table in (("metrics", metrics), ("original_intervals", original), ("group_metrics", pd.DataFrame(groups))):
            table.to_csv(out / f"{name}.csv", index=False)
        aggregate = summarize(metrics, out)
        verification = verify(out, metrics, original)
        assert all(cp.digest(row["path"]) == row["sha256"] for row in inputs)
        save(out / "construction.json", construction)
        save(out / "audit.json", {**verification, "inputs_unchanged": True, "inputs": inputs,
             "no_point_training": True, "all_methods_reported": True})
        report(out, aggregate, decisions)
        save(out / "state.json", {"state": "complete", "completed_jobs": len(parent["jobs"]), "updated_at": cp.stamp(), "plots": "pending; run plot_score_locality.py"})
        print(f"COMPLETE {out / 'REPORT.md'}", flush=True)
    except Exception as error:
        save(out / "state.json", {"state": "failed", "updated_at": cp.stamp(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
