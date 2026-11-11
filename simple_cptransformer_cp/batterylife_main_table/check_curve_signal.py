"""Bounded train/dev-only feature and neighbourhood audit on frozen auxiliaries.

No point-model training; no original calibration or test data is evaluated.
OOF-CV is a screening diagnostic, not end-to-end nested validation: reference
OOF auxiliary models overlap. Independent-of-aux-training dev20 is also read.
"""
import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler

import run_isu_disagreement_cp as cp

HERE = Path(__file__).resolve().parent
OLD = HERE / "isu_crossfit_risk_20261028"
OUT = HERE / "curve_signal_20261104_v2"
FAMILIES = ("hand", "embedding", "curve")
METHODS = ["qr_life"] + [f"qr_{f}" for f in FAMILIES] + ["kernel_global", "kernel_life"] + [f"kernel_{f}" for f in FAMILIES]


def features(curves):
    """Model-input descriptors; NOT physical SoH or measured Q(V)."""
    x = np.asarray(curves, dtype=float)
    assert x.ndim == 4 and x.shape[1:] == (100, 3, 300) and np.isfinite(x).all()
    early = x[:, :10].mean(axis=(1, 3))
    change = x[:, -10:].mean(axis=(1, 3)) - early
    rms = np.sqrt(np.square(np.diff(x, axis=1)).mean(axis=(1, 3)))
    hand = np.column_stack((early, change, rms))
    trajectories = x.mean(axis=3)
    grid = np.linspace(0, 99, 16)
    trend = np.stack([np.column_stack([np.interp(grid, np.arange(100), a[:, c]) for c in range(3)]).T.reshape(-1)
                      for a in trajectories])
    std = trajectories.std(axis=1)
    difference = np.diff(trajectories, axis=1).std(axis=1)
    t = np.arange(100) - 49.5
    slope = np.einsum("ntc,t->nc", trajectories, t) / np.sum(t*t)
    curve = np.column_stack((trend, std, difference, slope))
    assert hand.shape[1] == 9 and curve.shape[1] == 57
    return {"hand": hand, "curve": curve}


def life_design(reference_prediction, query_prediction):
    scaler = StandardScaler().fit(np.log(np.maximum(1., reference_prediction))[:, None])
    r = scaler.transform(np.log(np.maximum(1., reference_prediction))[:, None])
    q = scaler.transform(np.log(np.maximum(1., query_prediction))[:, None])
    return np.column_stack((r, r*r)), np.column_stack((q, q*q)), scaler


def transform_extra(reference, query, life_r, life_q, family):
    # Remove reference-side life association before inspecting extra structure.
    scaler = StandardScaler().fit(reference)
    r, q = scaler.transform(reference), scaler.transform(query)
    nuisance = Ridge(alpha=1.).fit(life_r, r)
    r, q = r-nuisance.predict(life_r), q-nuisance.predict(life_q)
    residual_scaler = StandardScaler().fit(r)
    r, q = np.clip(residual_scaler.transform(r), -5., 5.), np.clip(residual_scaler.transform(q), -5., 5.)
    pca = PCA(n_components=3, svd_solver="full").fit(r)
    pr, pq = pca.transform(r), pca.transform(q)
    pc_scale = StandardScaler().fit(pr)
    pr, pq = pc_scale.transform(pr), pc_scale.transform(pq)
    if family == "curve":
        # Six equally weighted feature blocks, not six groups of cells.
        block = np.r_[np.full(48, 1/np.sqrt(16)), np.full(9, 1/np.sqrt(3))] / np.sqrt(6)
        distance_r, distance_q = r*block, q*block
    else:
        distance_r, distance_q = pr/np.sqrt(3), pq/np.sqrt(3)
    return pr, pq, distance_r, distance_q, {
        "raw_mean": scaler.mean_.tolist(), "raw_scale": scaler.scale_.tolist(),
        "nuisance_coefficient": nuisance.coef_.tolist(), "nuisance_intercept": nuisance.intercept_.tolist(),
        "residual_mean": residual_scaler.mean_.tolist(), "residual_scale": residual_scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(), "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc_mean": pc_scale.mean_.tolist(), "pc_scale": pc_scale.scale_.tolist()}


def kernel_weights(reference, query):
    distances = pdist(reference)
    positive = distances[distances > 1e-12]
    # Fixed rule, no residuals, no validation bandwidth search.
    h = max(float(np.median(positive))*.5, 1e-6) if len(positive) else 1.
    logw = -np.square(cdist(query, reference))/(2*h*h)
    weights = np.exp(logw-logw.max(axis=1, keepdims=True))
    return weights/weights.sum(axis=1, keepdims=True), h


def quantiles(scores, weights):
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[:, order], axis=1)
    index = np.minimum(np.sum(cumulative < .9, axis=1), len(scores)-1)
    return scores[order[index]]


def estimate(reference_p, reference_y, query_p, reference_features, query_features):
    """Query outcomes are deliberately absent from the interface."""
    target = np.abs(reference_y-reference_p)/np.maximum(1., reference_p)
    life_r, life_q, life_scaler = life_design(reference_p, query_p)
    estimate, weights, parameters = {}, {}, {"life_mean": life_scaler.mean_.tolist(), "life_scale": life_scaler.scale_.tolist()}
    reg_scaler = StandardScaler().fit(life_r)
    lr, lq = reg_scaler.transform(life_r), reg_scaler.transform(life_q)
    def regress(name, r, q):
        model = QuantileRegressor(quantile=.9, alpha=.01, solver="highs").fit(r, target)
        estimate[name] = np.maximum(0., model.predict(q))
        parameters[name] = {"coefficient": model.coef_.tolist(), "intercept": float(model.intercept_)}
    regress("qr_life", lr, lq)
    weights["kernel_global"] = np.full((len(query_p), len(reference_p)), 1/len(reference_p))
    weights["kernel_life"], h = kernel_weights(life_r[:, :1], life_q[:, :1])
    parameters["kernel_life_h"] = h
    for family in FAMILIES:
        pr, pq, dr, dq, pars = transform_extra(reference_features[family], query_features[family], life_r, life_q, family)
        regress(f"qr_{family}", np.column_stack((lr, pr)), np.column_stack((lq, pq)))
        weights[f"kernel_{family}"], h = kernel_weights(np.column_stack((life_r[:, :1], dr)), np.column_stack((life_q[:, :1], dq)))
        parameters[family] = {**pars, "h": h}
    for name, w in weights.items():
        estimate[name] = quantiles(target, w)
    return estimate, weights, target, parameters


def load_bank(manifest, inputs):
    identity = cp.read(OLD / "bank_identity.json")
    assert cp.digest(OLD / "train_bank.npz") == identity["bank_sha256"]
    bank = dict(np.load(OLD / "train_bank.npz", allow_pickle=False))
    phases = manifest["split"]["phases"]
    assert bank["names"].tolist() == phases["train"]
    directory = cp.ROOT / "shared_cache/split_20261026/processed_cache"
    for path in directory.glob("dev_*.pkl"):
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload["identity"]["files"] != phases["dev"]:
            continue
        fields = payload["fields"]
        assert fields["files"] == phases["dev"]
        assert [fields["cell_prefix_counts"][name] for name in phases["dev"]] == [1]*20
        dev_curves = np.stack(fields["total_charge_discharge_curves"]).astype(np.float32)
        inputs.append({"path": str(path), "sha256": cp.digest(path)})
        break
    else:
        raise FileNotFoundError("Matching dev20 cache missing")
    label_path = Path(identity["labels"])
    assert cp.digest(label_path) == identity["labels_sha256"]
    labels = cp.read(label_path)
    names = np.r_[bank["names"], phases["dev"]]
    y = np.r_[bank["labels"], [labels[name] for name in phases["dev"]]]
    assert len(set(names)) == 117
    inputs.extend({"path": str(p), "sha256": cp.digest(p)} for p in (OLD / "train_bank.npz", label_path))
    return names, np.concatenate((bank["curves"], dev_curves)), y


def extract_all(out, old_plan, names, curves, oof, inputs, device):
    import torch
    upstream = HERE.parents[1] / "third_party/BatteryLife"
    sys.path.insert(0, str(upstream))
    torch.set_num_threads(4)
    target_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    result = {}
    lookup = {name: i for i, name in enumerate(names)}
    max_error = 0.
    for job in old_plan["jobs"]:
        folder = OLD / "folds" / job["key"]
        completion = cp.read(folder / "complete.json")
        checkpoint_path = folder / "auxiliary_model.pt"
        assert not completion["smoke_only"] and completion["job"] == job
        assert cp.digest(checkpoint_path) == completion["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["trained_on"] == job["outer_fit"]
        assert not set(names[97:]) & set(checkpoint["trained_on"])
        assert not set(job["outer_holdout"]) & set(checkpoint["trained_on"])
        config_path = Path(old_plan["source"]) / job["model"] / "config.json"
        cfg = SimpleNamespace(**cp.read(config_path)["model_config"])
        model = importlib.import_module(f"models.{job['model']}").Model(cfg).to(target_device).eval()
        model.load_state_dict(checkpoint["state_dict"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        predictions = np.full(len(names), np.nan)
        embeddings = [None]*len(names)
        with torch.inference_mode():
            # Packed LSTM/linear CUDA arithmetic can vary slightly with batch
            # composition. Replay the heldout group in its original order and
            # original batch size, rather than silently relaxing parity checks.
            for group in (job["outer_fit"], job["outer_holdout"], names[97:].tolist()):
                for start in range(0, len(group), cfg.batch_size):
                    indices = [lookup[name] for name in group[start:start+cfg.batch_size]]
                    x = torch.from_numpy(curves[indices]).to(target_device)
                    mask = torch.ones((len(x), 100), device=target_device)
                    p, z = model(x, mask, return_embedding=True)
                    predictions[indices] = p.reshape(-1).cpu().numpy()*checkpoint["label_std"]+checkpoint["label_mean"]
                    for index, value in zip(indices, z.cpu().numpy()):
                        embeddings[index] = value
        pred, embedding = np.asarray(predictions), np.asarray(embeddings)
        saved = oof[(oof.model == job["model"]) & (oof.fold == job["fold"])].set_index("cell_file").loc[job["outer_holdout"]]
        recovered = pred[[lookup[name] for name in job["outer_holdout"]]]
        np.testing.assert_allclose(recovered, saved.prediction, rtol=2e-6, atol=.003)
        max_error = max(max_error, float(np.max(np.abs(recovered-saved.prediction.to_numpy()))))
        assert np.isfinite(embedding).all()
        np.savez_compressed(out / "frozen_features" / f"{job['key']}.npz", names=names, prediction=pred, embedding=embedding)
        result[job["key"]] = {"prediction": np.maximum(0., pred), "embedding": embedding}
        for path in (checkpoint_path, folder / "complete.json", config_path, upstream / "models" / f"{job['model']}.py"):
            inputs.append({"path": str(path), "sha256": cp.digest(path)})
        print(f"INFERENCE {job['key']} n={len(names)} dim={embedding.shape[1]} (no training)", flush=True)
        del model
    return result, {"device": target_device, "oof_prediction_replay_max_cycles": max_error,
                    "validation_encoder_excludes_heldout": True, "dev20_excluded_from_all_auxiliary_training": True}


def evaluate(out, old_plan, names, y, oof, raw, extracted):
    records, fold_rows, parameters = [], [], {}
    lookup = {name: i for i, name in enumerate(names)}
    for architecture in ("CPMLP", "CPLSTM"):
        table = oof[oof.model == architecture].set_index("cell_file").loc[names[:97]]
        reference_p = np.maximum(0., table.prediction.to_numpy())
        np.testing.assert_allclose(table.eol, y[:97])
        for job in [j for j in old_plan["jobs"] if j["model"] == architecture]:
            feature = extracted[job["key"]]
            held = np.array([lookup[name] for name in job["outer_holdout"]])
            reference = np.array([i for i in range(97) if i not in set(held)])
            all_features = {**raw, "embedding": feature["embedding"]}
            for phase, ri, qi in (("oof_cv", reference, held), ("dev20", np.arange(97), np.arange(97, 117))):
                # CV targets use exact archived OOF predictions; dev uses this fold's frozen model.
                query_p = reference_p[qi] if phase == "oof_cv" else feature["prediction"][qi]
                estimates, weights, target, pars = estimate(reference_p[ri], y[ri], query_p,
                    {f: a[ri] for f, a in all_features.items()}, {f: a[qi] for f, a in all_features.items()})
                parameters[f"{job['key']}_{phase}"] = pars
                truth_score = np.abs(y[qi]-query_p)/np.maximum(1., query_p)
                np.savez_compressed(out / "neighbourhoods" / f"{job['key']}_{phase}.npz",
                    reference_names=names[ri], query_names=names[qi], reference_scores=target,
                    query_scores=truth_score, query_prediction=query_p, query_eol=y[qi],
                    **{name: w for name, w in weights.items()})
                for method, pred_q in estimates.items():
                    loss = .9*np.maximum(truth_score-pred_q, 0)+.1*np.maximum(pred_q-truth_score, 0)
                    fold_rows.append({"model": architecture, "fold": job["fold"], "phase": phase, "method": method,
                        "n": len(qi), "pinball": float(loss.mean()), "fraction_below_q": float(np.mean(truth_score <= pred_q)),
                        "mean_q": float(pred_q.mean()), "mean_ess": float(np.mean(1/np.square(weights[method]).sum(axis=1))) if method in weights else np.nan})
                    for j, i in enumerate(qi):
                        records.append({"model": architecture, "fold": job["fold"], "phase": phase, "method": method,
                            "cell_file": names[i], "point_prediction": query_p[j], "eol": y[i], "relative_error": truth_score[j],
                            "predicted_q90": pred_q[j], "pinball": loss[j], "below_q": bool(truth_score[j] <= pred_q[j])})
    pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    cp.write(out / "transform_and_models.json", parameters)
    return pd.DataFrame(records), pd.DataFrame(fold_rows)


def review(out, predictions, folds):
    summary, decisions = [], []
    for (model, phase, method), part in predictions.groupby(["model", "phase", "method"]):
        control = "qr_life" if method.startswith("qr_") else "kernel_life"
        ref = predictions[(predictions.model == model) & (predictions.phase == phase) & (predictions.method == control)].set_index(["fold", "cell_file"])
        paired = ref.loc[pd.MultiIndex.from_frame(part[["fold", "cell_file"]])]
        base = float(paired.pinball.mean())
        fm = folds[(folds.model == model) & (folds.phase == phase)].pivot(index="fold", columns="method", values="pinball")
        summary.append({"model": model, "phase": phase, "method": method, "control": control,
            "cell_presentations": len(part), "unique_cells": part.cell_file.nunique(), "pinball": float(part.pinball.mean()),
            "gain_pct_vs_life": float(100*(1-part.pinball.mean()/base)), "fold_wins_vs_life": int((fm[method] < fm[control]).sum()),
            "fraction_below_q": float(part.below_q.mean()), "mean_q": float(part.predicted_q90.mean())})
    summary = pd.DataFrame(summary)
    for model in ("CPMLP", "CPLSTM"):
        for method in [f"{route}_{family}" for route in ("qr", "kernel") for family in FAMILIES]:
            rows = summary[(summary.model == model) & (summary.method == method)].set_index("phase")
            cv, dev = rows.loc["oof_cv"], rows.loc["dev20"]
            passed = cv.gain_pct_vs_life >= 2 and cv.fold_wins_vs_life >= 3 and dev.gain_pct_vs_life > 0 and dev.fold_wins_vs_life >= 3
            decisions.append({"model": model, "method": method, "advance_signal": bool(passed)})
    summary.to_csv(out / "summary.csv", index=False)
    cp.write(out / "decisions.json", {"gate": "OOF-CV pinball improves >=2% vs matched life-only route with >=3/5 fold wins; dev20 gain positive and >=3/5 views better. Exploratory screen only, not an automatic CP success.", "rows": decisions})
    lines = ["# 曲线信息检查：先看这里", "", "固定前100周期，只使用原train97和dev20。未训练任何预测器，未评价原cal/test。",
        "分折冻结模型各自提取表示，不拼接不同模型的embedding坐标。归一化误差=绝对误差/max(1,预测寿命)。", "",
        "## 检查什么", "", "hand：归一化输入三通道的前10周期均值、后10与前10差、逐周期变化RMS，共9维；不是物理SoH。",
        "embedding：对应分折冻结预测器的内部表示，验证电芯未参与该表示模型训练。",
        "curve：复用旧57维时间曲线特征；直接距离按电压/电流/容量趋势、标准差、差分标准差、斜率六个特征块等权。六组是特征组，不是电芯类别。",
        "各附加信息先在参考侧扣除log寿命二次项能拟合的关联；PCA3和标准化也只拟合参考侧。这是建模控制，不是条件独立证明。", "",
        "## 两种检查", "", "qr：固定alpha=.01的90%分位数回归，life-only与增加三维特征对照；预测目标直接为归一化误差，不是log误差。",
        "kernel：直接用参考误差构造CDF，全局、仅寿命及寿命+特征分别比较；RBF带宽固定为参考两两距离中位数的一半，无监督选择。",
        "这里是经验90%尾部估计诊断，没有加入CP有限样本修正或测试∞原子；fraction_below_q不是最终CP覆盖率，不能把这些CDF当成有保证的预测区间。", "",
        "## 结果：pinball越低越好，增益与同路线life-only比较", "", "|模型|证据|方法|pinball|相对寿命增益|胜出折/视角|", "|---|---|---|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"|{r.model}|{r.phase}|{r.method}|{r.pinball:.4f}|{r.gain_pct_vs_life:+.1f}%|{r.fold_wins_vs_life}/5|")
    lines += ["", "## 预定门槛", "", "OOF-CV较匹配寿命对照改善至少2%且至少3/5折胜；dev20方向为正且至少3/5模型视角胜，才列为下一步候选。所有方法报告，不在dev结果后改特征或带宽。"]
    for model in ("CPMLP", "CPLSTM"):
        passed = [d["method"] for d in decisions if d["model"] == model and d["advance_signal"]]
        lines.append(f"- {model}：{'、'.join(passed) if passed else '没有候选'}通过初筛门槛。")
    lines += ["", "## 证据边界", "", "OOF-CV每个验证折使用未训练过该折电芯的冻结encoder，避免直接使用主模型训练集表示；但参考OOF残差来自其他辅助模型，这些模型训练集有重叠、可能见过当前验证折标签，所以不是端到端嵌套独立验证。",
        "dev20没有进入任何辅助模型训练/epoch选择，但历史上已看过；同20块由5个模型重复预测，不是100块新电芯。参考误差来自5个辅助预测器的混合，最终主预测器的条件误差分布仍可能不同。",
        "如果出现候选，只说明值得更严格复核及后续同分数CP消融，不证明已经提升最终LCP或具有条件覆盖保证。", "",
        "## 可视化", "", "visuals/OOF_CDF_all97.pdf：主CPMLP所有97块训练侧OOF查询的CDF，比较全局、寿命邻域、曲线/手工/embedding邻域。",
        "邻域逐点权重和CDF、选中分位点、ESS、逐电芯误差与q均保留；标题明确这不是最终CP。没有按好坏筛选电芯。"]
    (out / "BRIEF.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise FileExistsError("Use a new directory to preserve the frozen run")
    cp.write(out / "plan.json", {"created_at": cp.stamp(), "code_sha256": cp.digest(__file__),
        "hypotheses": ["Hfeature: low-dimensional input summaries or frozen embeddings improve relative-error q90 beyond lifetime", "Hneighbour: a fixed curve neighbourhood estimates the error tail better than lifetime-only without a learned risk scalar"],
        "data": "original train97 OOF and original dev20 only; original cal/test excluded", "point_training": False,
        "models": ["CPMLP", "CPLSTM"], "methods": METHODS, "quantile": .9, "regression_alpha": .01,
        "pca_components": 3, "kernel_bandwidth_rule": "0.5*median positive reference pairwise distance; no tuning",
        "feature_families": {"hand": "9 model-input early/change/RMS descriptors", "embedding": "fold-specific frozen encoder; no pooling coordinate systems", "curve": "57 time features, six balanced blocks for distances"},
        "validation": "OOF leave auxiliary fold out (screen only; overlapping reference OOF fits); dev20 repeated five frozen-model views",
        "decision": "OOF gain>=2% and >=3/5 fold wins vs matched life-only; positive dev gain and >=3/5 view wins; exploratory candidates only",
        "plots": "all97 primary OOF queries; no outcome-based page selection", "new_cp_evaluation": False})
    for directory in ("frozen_features", "neighbourhoods"):
        (out / directory).mkdir()
    cp.write(out / "state.json", {"state": "running", "stage": "load", "pid": os.getpid(), "updated_at": cp.stamp()})
    try:
        inputs = [{"path": str(Path(__file__)), "sha256": cp.digest(__file__)}]
        old_plan = cp.read(OLD / "plan.json")
        for dependency in old_plan["dependencies"]:
            assert cp.digest(dependency["path"]) == dependency["sha256"]
        manifest = cp.read(OLD / "dataset_manifest.json")
        assert cp.digest(OLD / "dataset_manifest.json") == old_plan["manifest_sha256"]
        for path in (OLD / "plan.json", OLD / "dataset_manifest.json", OLD / "bank_identity.json", OLD / "oof_predictions.csv", Path(cp.__file__)):
            inputs.append({"path": str(path), "sha256": cp.digest(path)})
        oof = pd.read_csv(OLD / "oof_predictions.csv")
        names, curves, y = load_bank(manifest, inputs)
        assert set(oof.cell_file) == set(names[:97]) and len(oof) == 194
        raw = features(curves)
        np.savez_compressed(out / "input_descriptors.npz", names=names, hand=raw["hand"], curve=raw["curve"], y=y)
        print("Extracting frozen representations on train97+dev20 only", flush=True)
        extracted, verification = extract_all(out, old_plan, names, curves, oof, inputs, args.device)
        del curves
        cp.write(out / "state.json", {"state": "running", "stage": "tail_and_neighbourhood", "pid": os.getpid(), "updated_at": cp.stamp()})
        predictions, folds = evaluate(out, old_plan, names, y, oof, raw, extracted)
        review(out, predictions, folds)
        assert all(cp.digest(row["path"]) == row["sha256"] for row in inputs)
        cp.write(out / "audit.json", {**verification, "inputs_unchanged": True, "inputs": inputs,
            "train_cells": 97, "dev_cells": 20, "cal_test_evaluated": False, "new_point_training": False,
            "prediction_rows": len(predictions), "scientific_scope": "feature/tail screen, not final CP"})
        cp.write(out / "state.json", {"state": "complete", "stage": "numerical_complete", "updated_at": cp.stamp(), "plots": "pending"})
        print(f"COMPLETE {out / 'BRIEF.md'}", flush=True)
    except Exception as error:
        cp.write(out / "state.json", {"state": "failed", "updated_at": cp.stamp(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
