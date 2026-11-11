"""Low-capacity, cross-fitted quantile-profile neighbourhood screening.

This is a train/dev-only diagnostic, not a calibrated CP procedure. It reuses
the fixed OOF bank and its existing dependence caveats. No predictor training.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

import check_curve_signal as base

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "curve_signal_20261104_v2"
DEFAULT_OUT = HERE / "quantile_profile_20261111"
LEVELS = (.1, .5, .9)
METHODS = ("kernel_global", "kernel_life", "kernel_hand", "qr_hand_q90",
           "profile_life_3q", "profile_hand_q90", "profile_hand_3q")
WEIGHT_METHODS = tuple(name for name in METHODS if name != "qr_hand_q90")


def design(reference_p, query_p, reference_hand, query_hand):
    life_r, life_q, scaler = base.life_design(reference_p, query_p)
    normalizer = StandardScaler().fit(life_r)
    lr, lq = normalizer.transform(life_r), normalizer.transform(life_q)
    pr, pq, dr, dq, parameters = base.transform_extra(reference_hand, query_hand, life_r, life_q, "hand")
    return {"life": lr, "hand": np.column_stack((lr, pr)), "life_geometry": life_r[:, :1],
            "hand_geometry": np.column_stack((life_r[:, :1], dr))}, {
            "life": lq, "hand": np.column_stack((lq, pq)), "life_geometry": life_q[:, :1],
            "hand_geometry": np.column_stack((life_q[:, :1], dq))}, {
            "log_life_mean": scaler.mean_.tolist(), "log_life_scale": scaler.scale_.tolist(),
            "life_design_mean": normalizer.mean_.tolist(), "life_design_scale": normalizer.scale_.tolist(),
            "hand_transform": parameters}


def profile_fit(reference_x, target, query_x):
    """Three small linear quantile heads; all preprocessing already fit-only."""
    models = [QuantileRegressor(quantile=tau, alpha=.01, solver="highs").fit(reference_x, target) for tau in LEVELS]
    raw_reference = np.column_stack([model.predict(reference_x) for model in models])
    raw_query = np.column_stack([model.predict(query_x) for model in models])
    reference = np.sort(np.maximum(0., raw_reference), axis=1)
    query = np.sort(np.maximum(0., raw_query), axis=1)
    parameters = {"levels": list(LEVELS), "coefficient": [model.coef_.tolist() for model in models],
                  "intercept": [float(model.intercept_) for model in models],
                  "reference_crossing_rows_before_sort": int(np.any(np.diff(raw_reference, axis=1) < -1e-10, axis=1).sum()),
                  "query_crossing_rows_before_sort": int(np.any(np.diff(raw_query, axis=1) < -1e-10, axis=1).sum()),
                  "nonzero_coefficients": [int(np.sum(np.abs(model.coef_) > 1e-10)) for model in models],
                  "postprocessing": "clip nonnegative then sort each quantile vector; no upper clipping"}
    return reference, query, parameters


def profile_kernel(training_profile, reference_profile, query_profile):
    distances = pdist(training_profile)
    positive = distances[distances > 1e-10]
    h = max(float(np.median(positive))*.5, 1e-6) if len(positive) else 1.
    logw = -np.square(cdist(query_profile, reference_profile))/(2*h*h)
    weight = np.exp(logw-logw.max(axis=1, keepdims=True))
    return weight/weight.sum(axis=1, keepdims=True), h


def crossfit_profiles(p, target, hand, fold, query_p, query_hand):
    """For each reference block, train profiles without its own outcome.

    Every block uses the SAME fitted map for reference and query profiles.
    Block CDFs are mixed in proportion to reference-block size. This is not a
    single frozen-map kernel, and no CP guarantee is claimed for this screen.
    """
    nr, nq = len(p), len(query_p)
    weights = {name: np.zeros((nq, nr)) for name in ("profile_life_3q", "profile_hand_q90", "profile_hand_3q")}
    reference_profiles = {name: np.full((nr, 3), np.nan) for name in ("life", "hand")}
    query_profiles, parameters, inner_audit = {"life": [], "hand": []}, {}, []
    for block in sorted(set(fold)):
        fitting, reference = np.flatnonzero(fold != block), np.flatnonzero(fold == block)
        assert not set(fitting) & set(reference)
        combined_p = np.r_[p[reference], query_p]
        combined_hand = np.concatenate((hand[reference], query_hand))
        train_x, evaluation_x, transform = design(p[fitting], combined_p, hand[fitting], combined_hand)
        parameters[str(int(block))] = {"transform": transform}
        for family in ("life", "hand"):
            train_profile, combined_profile, pars = profile_fit(train_x[family], target[fitting], evaluation_x[family])
            ref_profile, query_profile = combined_profile[:len(reference)], combined_profile[len(reference):]
            reference_profiles[family][reference] = ref_profile
            query_profiles[family].append(query_profile)
            modes = ("profile_life_3q",) if family == "life" else ("profile_hand_q90", "profile_hand_3q")
            for mode in modes:
                dims = slice(2, 3) if mode.endswith("q90") else slice(None)
                w, h = profile_kernel(train_profile[:, dims], ref_profile[:, dims], query_profile[:, dims])
                weights[mode][:, reference] = w * len(reference)/nr
                parameters[str(int(block))][f"{mode}_h"] = h
            parameters[str(int(block))][family] = pars
        inner_audit.append({"inner_fold": int(block), "fit_indices": fitting.tolist(), "reference_indices": reference.tolist(),
                            "block_mass": len(reference)/nr})
    assert all(np.isfinite(values).all() for values in reference_profiles.values())
    for weight in weights.values():
        np.testing.assert_allclose(weight.sum(axis=1), 1.)
    return weights, reference_profiles, {k: np.asarray(v) for k, v in query_profiles.items()}, parameters, inner_audit


def estimate(reference_p, reference_y, reference_hand, reference_fold, query_p, query_hand):
    """Query labels are absent; all methods share the same reference scores."""
    target = np.abs(reference_y-reference_p)/np.maximum(1., reference_p)
    r, q, transform = design(reference_p, query_p, reference_hand, query_hand)
    weights = {"kernel_global": np.full((len(query_p), len(reference_p)), 1/len(reference_p))}
    weights["kernel_life"], life_h = base.kernel_weights(r["life_geometry"], q["life_geometry"])
    weights["kernel_hand"], hand_h = base.kernel_weights(r["hand_geometry"], q["hand_geometry"])
    _, direct, direct_parameters = profile_fit(r["hand"], target, q["hand"])
    # Replay the original q90-only regressor without sorting for a fair direct-head baseline.
    qr_model = QuantileRegressor(quantile=.9, alpha=.01, solver="highs").fit(r["hand"], target)
    direct_q = np.maximum(0., qr_model.predict(q["hand"]))
    learned, reference_profiles, query_profiles, profile_parameters, inner_audit = crossfit_profiles(
        reference_p, target, reference_hand, reference_fold, query_p, query_hand)
    weights.update(learned)
    estimates = {name: base.quantiles(target, w) for name, w in weights.items()}
    estimates["qr_hand_q90"] = direct_q
    return estimates, weights, target, reference_profiles, query_profiles, {
        "full_reference_transform": transform, "life_h": life_h, "hand_h": hand_h,
        "full_reference_direct_profile": direct_parameters,
        "direct_q90_coefficient": qr_model.coef_.tolist(), "direct_q90_intercept": float(qr_model.intercept_),
        "inner_profile_models": profile_parameters}, inner_audit


def summarize(out, predictions, folds):
    rows, decisions = [], []
    for (model, phase, method), part in predictions.groupby(["model", "phase", "method"]):
        row = {"model": model, "phase": phase, "method": method, "pinball": float(part.pinball.mean()),
               "fraction_below_q": float(part.below_q.mean()), "mean_q": float(part.predicted_q90.mean()),
               "unique_cells": part.cell_file.nunique(), "presentations": len(part)}
        fm = folds[(folds.model == model) & (folds.phase == phase)].pivot(index="fold", columns="method", values="pinball")
        for control in ("kernel_global", "kernel_hand", "profile_hand_q90", "profile_life_3q"):
            ref = predictions[(predictions.model == model) & (predictions.phase == phase) & (predictions.method == control)]
            row[f"gain_pct_vs_{control}"] = float(100*(1-part.pinball.mean()/ref.pinball.mean()))
            row[f"wins_vs_{control}"] = int((fm[method] < fm[control]).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    for model in ("CPMLP", "CPLSTM"):
        candidate = summary[(summary.model == model) & (summary.method == "profile_hand_3q")].set_index("phase")
        cv, dev = candidate.loc["oof_cv"], candidate.loc["dev20"]
        passed = all(cv[f"gain_pct_vs_{c}"] >= 2 and cv[f"wins_vs_{c}"] >= 3
                     and dev[f"gain_pct_vs_{c}"] > 0 and dev[f"wins_vs_{c}"] >= 3
                     for c in ("kernel_global", "kernel_hand"))
        multi_gain = bool(cv.gain_pct_vs_profile_hand_q90 > 0 and cv.wins_vs_profile_hand_q90 >= 3
                          and dev.gain_pct_vs_profile_hand_q90 > 0)
        decisions.append({"model": model, "learned_geometry_gate": bool(passed), "multi_quantile_increment": multi_gain})
    base.cp.write(out / "decisions.json", {"rows": decisions, "action": "No final CP evaluation in this batch; only a passing screen warrants a separate frozen method study."})
    lines = ["# 多分位误差画像：第一轮", "", "固定原预测器，只用原train97 OOF与dev20；原cal/test未评价。三个线性分位头估计归一化绝对误差的0.1/0.5/0.9分位。",
        "原手工9维摘要经过参考侧寿命关联扣除、PCA3，加入log预测寿命二次基函数。每个hand头5个输入+截距，三头共18个回归系数，不训练深序列网络。",
        "", "## 主要对照", "", "kernel_global=全局经验误差分布；kernel_hand=此前固定手工邻域；profile_hand_3q=主要候选；profile_hand_q90=同一画像只使用最后一维；profile_life_3q=不加入手工信息；qr_hand_q90=直接用分位回归输出。",
        "画像参考点采用内层留辅助折拟合：一块参考电芯自己的误差不参与计算它的画像。每个内折对参考/查询用同一个映射，内折CDF按参考块大小混合；不是把不同坐标随意拼接。",
        "独立分位头输出先截断为非负、再逐行排序；单分位画像对照也使用同一排序后的最后一维。核带宽固定为内层拟合点画像距离中位数的一半，原手工对照规则不变，不调参。",
        "", "## 结果：主指标是90%归一化误差分位的pinball损失，越小越好", "", "|预测器|阶段|方法|pinball|对全局改善|对手工改善|", "|---|---|---|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"|{row.model}|{row.phase}|{row.method}|{row.pinball:.4f}|{row.gain_pct_vs_kernel_global:+.1f}%|{row.gain_pct_vs_kernel_hand:+.1f}%|")
    lines += ["", "## 运行前固定的判断标准", "", "学习几何门槛：OOF较全局与手工均改善至少2%，且各至少3/5折胜；dev20对两者均方向为正、各至少3/5模型视角胜。多分位数额外贡献另比较同模型的q90-only画像。"]
    for decision in decisions:
        lines.append(f"- {decision['model']}：学习几何门槛{'通过' if decision['learned_geometry_gate'] else '未通过'}；多个分位数增量检查{'通过' if decision['multi_quantile_increment'] else '未通过'}。")
    lines += ["", "## 边界", "", "本批CDF是经验尾部估计，没有CP有限样本修正或测试∞原子，fraction_below_q不是最终CP覆盖保证。所有查询和负结果保留。",
        "内层画像留出解决的是画像直接拟合自己的参考误差；底层OOF点预测器训练仍重叠，参考预测器可能见过外层查询标签，所以OOF-CV仍不是端到端独立嵌套验证。",
        "dev20未进入辅助模型训练，但历史已用于探索；五个模型视角仍只有20块电芯。参考OOF为多个辅助模型的误差混合，与最终主模型可能不同。",
        "此方法是借鉴Lee等的多分位残差监督思想、使用输出分位画像的低容量适配，不是原文深encoder或其理论保证的直接复现。",
        "", "图入口：visuals/PROFILE_CDF_all97.pdf；逐电芯画像在profile_rows.csv，全部权重/参考画像/查询画像及模型参数保存在neighbourhoods与models。"]
    (out / "BRIEF.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise FileExistsError("Use a new output directory; preserve frozen plans")
    base.cp.write(out / "plan.json", {"created_at": base.cp.stamp(), "code_sha256": base.cp.digest(__file__),
        "hypotheses": ["Hgeometry: learned low-capacity residual profile improves over BOTH global and fixed hand neighbourhood", "Hmulti: three quantile coordinates add value over q90-only and lifetime-only profiles"],
        "levels": list(LEVELS), "alpha": .01, "methods": list(METHODS), "inputs": "same life2 + hand-PCA3 design as previous frozen curve audit",
        "postprocessing": "clip nonnegative then sort; q90-only uses same last coordinate",
        "profile_reference": "inner leave original auxiliary fold out; same mapping for inner reference and query; mix block CDFs by n_block/n_reference",
        "bandwidth": "half positive median fitting-profile pair distance; constant profile -> uniform within block; no search",
        "gate": "OOF >=2% gain and >=3/5 wins vs BOTH global/hand; dev positive gain and >=3/5 view wins vs BOTH. Multi contribution: positive CV and dev vs q90-only, >=3/5 CV wins.",
        "data": "train97 OOF + dev20 only", "cal_test_evaluation": False, "point_training": False,
        "residual_score": "abs(y-p)/max(1,p)", "quantile_level": .9,
        "scope": "empirical-tail diagnostic, not CP; no finite-sample correction or infinity test atom",
        "plots": "all97 primary CPMLP OOF queries; first query example fixed by membership order"})
    for folder in ("neighbourhoods", "models"):
        (out / folder).mkdir()
    base.cp.write(out / "state.json", {"state": "running", "pid": os.getpid(), "updated_at": base.cp.stamp()})
    try:
        inputs = []
        def track(path):
            path = Path(path)
            inputs.append({"path": str(path), "sha256": base.cp.digest(path)})
        for path in (__file__, base.__file__, base.cp.__file__, SOURCE / "audit.json", SOURCE / "input_descriptors.npz", base.OLD / "oof_predictions.csv", base.OLD / "plan.json"):
            track(path)
        source_audit = base.cp.read(SOURCE / "audit.json")
        assert all(base.cp.digest(item["path"]) == item["sha256"] for item in source_audit["inputs"])
        raw = dict(np.load(SOURCE / "input_descriptors.npz", allow_pickle=False))
        names, hand, y = raw["names"], raw["hand"], raw["y"]
        oof = pd.read_csv(base.OLD / "oof_predictions.csv")
        old_plan = base.cp.read(base.OLD / "plan.json")
        manifest = base.cp.read(base.OLD / "dataset_manifest.json")["split"]["phases"]
        assert names[:97].tolist() == manifest["train"] and names[97:].tolist() == manifest["dev"]
        records, metrics, profile_rows, all_inner_audit = [], [], [], []
        for architecture in ("CPMLP", "CPLSTM"):
            table = oof[oof.model == architecture].set_index("cell_file").loc[names[:97]]
            p, fold = np.maximum(0., table.prediction.to_numpy()), table.fold.to_numpy()
            np.testing.assert_allclose(y[:97], table.eol)
            for job in [j for j in old_plan["jobs"] if j["model"] == architecture]:
                f = job["fold"]
                frozen_path = SOURCE / "frozen_features" / f"{job['key']}.npz"
                track(frozen_path)
                frozen = dict(np.load(frozen_path, allow_pickle=False))
                np.testing.assert_array_equal(frozen["names"], names)
                held, reference = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
                for phase, ri, qi in (("oof_cv", reference, held), ("dev20", np.arange(97), np.arange(97, 117))):
                    query_p = p[qi] if phase == "oof_cv" else np.maximum(0., frozen["prediction"][qi])
                    result, weights, target, ref_profiles, query_profiles, parameters, inner_audit = estimate(
                        p[ri], y[ri], hand[ri], fold[ri], query_p, hand[qi])
                    label = f"{job['key']}_{phase}"
                    base.cp.write(out / "models" / f"{label}.json", parameters)
                    for audit in inner_audit:
                        fitting = names[ri[np.asarray(audit["fit_indices"])]]
                        ref = names[ri[np.asarray(audit["reference_indices"])]]
                        assert not set(fitting) & set(ref) and not set(fitting) & set(names[qi])
                        all_inner_audit.append({"model": architecture, "outer_fold": f, "phase": phase, "inner_fold": audit["inner_fold"],
                            "fitting": fitting.tolist(), "reference": ref.tolist(), "query": names[qi].tolist(), "block_mass": audit["block_mass"]})
                    actual = np.abs(y[qi]-query_p)/np.maximum(1., query_p)
                    np.savez_compressed(out / "neighbourhoods" / f"{label}.npz", reference_names=names[ri], query_names=names[qi],
                        reference_scores=target, query_scores=actual, reference_fold=fold[ri], query_prediction=query_p, query_eol=y[qi],
                        reference_profile_hand=ref_profiles["hand"], reference_profile_life=ref_profiles["life"],
                        query_profile_hand=query_profiles["hand"], query_profile_life=query_profiles["life"], **weights)
                    block_ids = sorted(set(fold[ri]))
                    block_mass = np.array([np.mean(fold[ri] == block) for block in block_ids])
                    for family in ("hand", "life"):
                        average = np.einsum("b,bqk->qk", block_mass, query_profiles[family])
                        for j, i in enumerate(qi):
                            profile_rows.append({"model": architecture, "fold": f, "phase": phase, "family": family, "cell_file": names[i],
                                "profile_q10_mean": average[j, 0], "profile_q50_mean": average[j, 1], "profile_q90_mean": average[j, 2],
                                "actual_relative_error": actual[j], "note": "display average only; weights use each matched inner map"})
                    for method in METHODS:
                        q90 = result[method]
                        loss = .9*np.maximum(actual-q90, 0)+.1*np.maximum(q90-actual, 0)
                        metrics.append({"model": architecture, "fold": f, "phase": phase, "method": method, "n": len(qi),
                            "pinball": float(loss.mean()), "fraction_below_q": float(np.mean(actual <= q90)),
                            "mean_q": float(q90.mean()), "mean_ess": float(np.mean(1/np.square(weights[method]).sum(axis=1))) if method in weights else np.nan})
                        for j, i in enumerate(qi):
                            records.append({"model": architecture, "fold": f, "phase": phase, "method": method,
                                "cell_file": names[i], "point_prediction": query_p[j], "eol": y[i], "relative_error": actual[j],
                                "predicted_q90": q90[j], "pinball": loss[j], "below_q": bool(actual[j] <= q90[j])})
                    print(f"DONE {label}", flush=True)
        predictions, folds = pd.DataFrame(records), pd.DataFrame(metrics)
        predictions.to_csv(out / "predictions.csv", index=False)
        folds.to_csv(out / "fold_metrics.csv", index=False)
        pd.DataFrame(profile_rows).to_csv(out / "profile_rows.csv", index=False)
        base.cp.write(out / "inner_fit_audit.json", all_inner_audit)
        summarize(out, predictions, folds)
        assert all(base.cp.digest(row["path"]) == row["sha256"] for row in inputs)
        base.cp.write(out / "audit.json", {"inputs": inputs, "inputs_unchanged": True, "no_point_training": True,
            "original_cal_test_evaluated": False, "inner_fit_reference_disjoint_checks": len(all_inner_audit),
            "prediction_rows": len(predictions)})
        base.cp.write(out / "state.json", {"state": "complete", "updated_at": base.cp.stamp(), "plots": "pending", "review": "pending"})
        print(f"COMPLETE {out / 'BRIEF.md'}", flush=True)
    except Exception as error:
        base.cp.write(out / "state.json", {"state": "failed", "updated_at": base.cp.stamp(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
