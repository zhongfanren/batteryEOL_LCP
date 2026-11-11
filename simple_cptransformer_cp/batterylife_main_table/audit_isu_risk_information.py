"""Exploratory information audit on frozen ISU predictors; no CP or GPU training.

Historical cal/test labels are used explicitly for diagnostic risk-model CV.
These outputs must never be presented as fresh conformal calibration/test data.
"""

import argparse
import hashlib
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import SplineTransformer, StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE / "isu_stability_20261028"
DEFAULT_OUT = ROOT / "risk_information_20261028"
PRIMARY = "s20261026_r2024_CPMLP"
PHASES = ("train", "dev", "calibration", "test")
FAMILIES = ("constant", "life", "protocol", "embedding", "curve_summary", "disagreement", "novelty")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            result.update(chunk)
    return result.hexdigest()


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def condition(name):
    return re.search(r"_(G\d+)C", str(name)).group(1)


def summarize_curve(curve):
    curve = np.asarray(curve, dtype=float)
    assert curve.shape == (100, 3, 300) and np.isfinite(curve).all()
    # These are normalized model-input summaries, not physical capacity/SoH.
    early = curve[:10].mean(axis=(0, 2))
    change = curve[-10:].mean(axis=(0, 2)) - early
    variability = np.sqrt(np.mean(np.diff(curve, axis=0) ** 2, axis=(0, 2)))
    return np.r_[early, change, variability]


def curve_inventory(out, manifest):
    """Read one existing split's caches once, using exactly the T100 array."""
    directory = ROOT / "shared_cache/split_20261026/processed_cache"
    rows, inputs = [], []
    for phase in PHASES:
        names = manifest["split"]["phases"][phase]
        for path in sorted(directory.glob(f"{phase}_*.pkl")):
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload["identity"]["files"] != names:
                continue
            fields = payload["fields"]
            assert fields["files"] == names
            counts = np.asarray([fields["cell_prefix_counts"][n] for n in names])
            assert np.all(counts == (100 if phase == "train" else 1))
            indices = np.cumsum(counts) - 1
            for name, index in zip(names, indices):
                values = summarize_curve(fields["total_charge_discharge_curves"][index])
                rows.append(dict(cell_file=name, **{f"curve_{i}": v for i, v in enumerate(values)}))
            inputs.append({"path": str(path), "sha256": digest(path)})
            print(f"CURVES phase={phase} cells={len(names)}", flush=True)
            del payload, fields
            break
        else:
            raise FileNotFoundError(f"No matching cache: {phase}")
    table = pd.DataFrame(rows).set_index("cell_file")
    assert len(table) == table.index.nunique() == 239
    table.to_csv(out / "curve_inventory.csv")
    return table, inputs


def residual_features(train_base, bases, raw):
    """Remove training-side life association and standardize without any Y."""
    nuisance = Ridge(alpha=1.).fit(train_base, raw["train"])
    # sklearn may squeeze a one-column regression target in predict().
    residuals = {p: raw[p] - nuisance.predict(bases[p]).reshape(raw[p].shape) for p in PHASES}
    scaler = StandardScaler().fit(residuals["train"])
    dimension = residuals["train"].shape[1]
    return {p: scaler.transform(residuals[p]) / np.sqrt(dimension) for p in PHASES}


def feature_design(data, manifest, cohort, curves):
    pred = {p: np.maximum(1., data[f"{p}_prediction"]) for p in PHASES}
    spline = SplineTransformer(n_knots=4, degree=2, include_bias=False,
                               extrapolation="linear").fit(np.log(pred["train"])[:, None])
    raw_basis = {p: spline.transform(np.log(pred[p])[:, None]) for p in PHASES}
    scaler = StandardScaler().fit(raw_basis["train"])
    bases = {p: scaler.transform(raw_basis[p]) / np.sqrt(raw_basis[p].shape[1]) for p in PHASES}
    metadata = {r["cell_file"]: r["metadata"] for r in manifest["cells"]}
    rates = {p: np.asarray([[metadata[n][k][0]["rate_in_C"]
                            for k in ("charge_protocol", "discharge_protocol")]
                           for n in data[f"{p}_names"]]) for p in PHASES}
    pca = PCA(n_components=3, whiten=True, svd_solver="full").fit(data["train_embedding"])
    embedding = {p: pca.transform(data[f"{p}_embedding"]) for p in PHASES}
    raw_summary = {p: curves.loc[data[f"{p}_names"]].to_numpy() for p in PHASES}
    summary_scaler = StandardScaler().fit(raw_summary["train"])
    summary_pca = PCA(n_components=3, whiten=True, svd_solver="full").fit(
        summary_scaler.transform(raw_summary["train"]))
    summary = {p: summary_pca.transform(summary_scaler.transform(raw_summary[p])) for p in PHASES}
    same_family = [v for key, v in cohort.items() if key.endswith(manifest["audit_model"])]
    assert len(same_family) == 3 and len(cohort) == 6
    disagreement = {}
    for p in PHASES:
        predictions = {}
        for key, values in cohort.items():
            assert np.array_equal(values[f"{p}_names"], data[f"{p}_names"])
            predictions.setdefault(key.rsplit("_", 1)[-1], []).append(values[f"{p}_prediction"])
        spread = np.std([v[f"{p}_prediction"] for v in same_family], axis=0, ddof=1)
        difference = np.abs(np.mean(predictions["CPMLP"], axis=0) - np.mean(predictions["CPLSTM"], axis=0))
        disagreement[p] = np.log1p(np.column_stack((spread, difference)) / pred[p][:, None])
    neighbors = NearestNeighbors(n_neighbors=6).fit(embedding["train"])
    novelty = {}
    for p in PHASES:
        distances = neighbors.kneighbors(embedding[p], return_distance=True)[0]
        # Training queries omit their zero self-distance.
        novelty[p] = np.log1p(distances[:, 1:6].mean(axis=1) if p == "train"
                              else distances[:, :5].mean(axis=1))[:, None]
    groups = {"protocol": rates, "embedding": embedding, "curve_summary": summary,
              "disagreement": disagreement, "novelty": novelty}
    design = {"constant": {p: np.zeros((len(pred[p]), 1)) for p in PHASES}, "life": bases}
    for name, raw in groups.items():
        transformed = residual_features(bases["train"], bases, raw)
        design[name] = {p: np.column_stack((bases[p], transformed[p])) for p in PHASES}
    assert all(np.isfinite(x).all() for v in design.values() for x in v.values())
    return design


def fit_predict(x, relative_error, query):
    # Fixed low-capacity models; no parameter or feature choice from outcomes.
    model = Ridge(alpha=10.).fit(x, np.log1p(relative_error))
    risk = model.predict(query)
    tail = (relative_error > .5).astype(int)
    if len(np.unique(tail)) < 2:
        probability = np.full(len(query), tail.mean())
    else:
        classifier = LogisticRegression(C=.1, max_iter=2000, solver="lbfgs").fit(x, tail)
        probability = classifier.predict_proba(query)[:, 1]
    return risk, probability


def scores(relative_error, risk, probability):
    target = np.log1p(relative_error)
    event = relative_error > .5
    correlation = spearmanr(risk, relative_error).statistic if np.ptp(risk) > 1e-12 else np.nan
    return {"n": len(target), "tail_events": int(event.sum()), "log_relative_mse": np.mean((risk - target) ** 2),
            "spearman": correlation, "tail_brier": np.mean((probability - event) ** 2),
            "tail_auc": roc_auc_score(event, probability) if len(np.unique(event)) == 2 else np.nan}


def run_job(spec, loaded, curves):
    job = spec["key"]
    source = Path(spec["source"])
    manifest = read(source / "dataset_manifest.json")
    manifest["audit_model"] = spec["model"]
    data = loaded[job]
    cohort = {k: v for k, v in loaded.items() if k.startswith(f"s{spec['split_seed']}_")}
    design = feature_design(data, manifest, cohort, curves)
    names = np.r_[data["calibration_names"], data["test_names"]]
    prediction = np.r_[data["calibration_prediction"], data["test_prediction"]]
    truth = np.r_[data["calibration_eol"], data["test_eol"]]
    relative = np.abs(truth - prediction) / np.maximum(1., prediction)
    dev_relative = np.abs(data["dev_eol"] - data["dev_prediction"]) / np.maximum(1., data["dev_prediction"])
    groups = np.asarray([condition(n) for n in names])
    folds = list(GroupKFold(n_splits=5).split(names, groups=groups))
    predictions, metric_rows, audit = [], [], []
    for family in FAMILIES:
        pool_x = np.concatenate((design[family]["calibration"], design[family]["test"]))
        direct = fit_predict(design[family]["dev"], dev_relative, pool_x)
        outputs = {"dev20_to_pool": direct}
        for mode, size in (("group_cv20", 20), ("group_cv_all", None)):
            risk, probability = np.full(len(names), np.nan), np.full(len(names), np.nan)
            for fold, (train, test) in enumerate(folds):
                assert not set(groups[train]) & set(groups[test])
                subset = train if size is None else np.random.default_rng(20261028 + fold).choice(train, size, replace=False)
                risk[test], probability[test] = fit_predict(pool_x[subset], relative[subset], pool_x[test])
                if family == "life":
                    audit.append({"job": job, "mode": mode, "fold": fold, "train_n": len(subset), "test_n": len(test),
                                  "train_names": names[subset].tolist(), "test_names": names[test].tolist(),
                                  "condition_disjoint": True, "training_tail_events": int((relative[subset] > .5).sum())})
            assert np.isfinite(risk).all() and np.isfinite(probability).all()
            outputs[mode] = risk, probability
        for mode, (risk, probability) in outputs.items():
            for scope, keep in (("pool122", np.ones(len(names), bool)), ("original_test61", np.arange(len(names)) >= 61)):
                metric_rows.append({"job": job, "model": spec["model"], "split_seed": spec["split_seed"],
                                    "train_seed": spec["train_seed"], "family": family, "mode": mode,
                                    "scope": scope, **scores(relative[keep], risk[keep], probability[keep])})
            for i, name in enumerate(names):
                predictions.append({"job": job, "cell_file": name, "condition": groups[i], "phase": "calibration" if i < 61 else "test",
                                    "family": family, "mode": mode, "prediction": prediction[i], "eol": truth[i],
                                    "relative_error": relative[i], "risk_prediction": risk[i], "tail_probability": probability[i]})
    support = {"job": job, "dev_n": 20, "dev_tail_events": int((dev_relative > .5).sum()),
               "dev_q90_relative_error": float(np.quantile(dev_relative, .9)),
               "pool_tail_events": int((relative > .5).sum()), "pool_conditions": len(set(groups)),
               "dev_conditions": len(set(condition(n) for n in data["dev_names"]))}
    return metric_rows, predictions, audit, support


def tables(metrics):
    base = metrics[metrics.family == "life"].set_index(["job", "mode", "scope"])
    paired = metrics.copy()
    ref = base.loc[pd.MultiIndex.from_frame(paired[["job", "mode", "scope"]])]
    for loss in ("log_relative_mse", "tail_brier"):
        paired[f"{loss}_gain_pct"] = 100 * (1 - paired[loss].to_numpy() / ref[loss].to_numpy())
    paired["tail_auc_delta"] = paired.tail_auc.to_numpy() - ref.tail_auc.to_numpy()
    rows = []
    for keys, group in paired[paired.scope == "pool122"].groupby(["model", "mode", "family"]):
        row = dict(zip(("model", "mode", "family"), keys))
        for name in ("log_relative_mse_gain_pct", "tail_brier_gain_pct", "tail_auc_delta", "spearman"):
            row[name] = group[name].mean()
        row.update(fits=len(group), mse_wins=int((group.log_relative_mse_gain_pct > 0).sum()),
                   brier_wins=int((group.tail_brier_gain_pct > 0).sum()))
        rows.append(row)
    return paired, pd.DataFrame(rows)


def report(out, metrics, paired, aggregate, support):
    labels = {"constant": "常数", "life": "仅预测寿命", "protocol": "寿命＋倍率", "embedding": "寿命＋曲线表示",
              "curve_summary": "寿命＋曲线变化摘要", "disagreement": "寿命＋模型分歧", "novelty": "寿命＋训练距离"}
    lines = ["# 可利用风险信息审计", "", "这是冻结预测上的探索性诊断，不是新LCP方法的最终效果评价。未训练或更换点预测器。", "",
             "## 固定问题与信息来源", "", "在预测寿命相近时，哪些可提前获得的信息能进一步预测误差？",
             "相对误差定义为 |真实寿命−预测寿命| / max(1,预测寿命)，与MAPE分母不同。",
             "风险回归预测 log(1＋相对误差)，大误差事件预先固定为相对误差>50%；后者不是条件90%分位数。",
             "寿命对照采用训练侧拟合的log预测寿命二次样条；每类附加特征先在训练侧扣除其与寿命样条的可拟合关系。",
             "这只是控制寿命影响的建模检查，不证明严格条件独立或因果关系。所有正负结果均保留。", "",
             "倍率只用充/放电C-rate，不使用有已知口径问题的SOC字段；曲线表示为训练侧PCA3。",
             "曲线摘要取归一化输入三通道的前10周期均值、末10与前10均值差、相邻周期变化RMS，再训练侧PCA3；不是物理SoH。",
             "分歧为同架构3训练种子的标准差，以及CPMLP/CPLSTM各3种子均值差，均除以当前预测寿命后log1p。",
             "训练距离为训练侧曲线PCA空间的5近邻平均距离。分歧不改变点预测中心。", "",
             "## 三种证据强度", "", "- dev20_to_pool：仅用原20块dev残差拟合风险模型，评价122块历史cal+test；另存原test61结果。dev曾用于选epoch，dev拟合本身不视为独立验证。",
             "- group_cv20：历史122块中按G实验条件做5折，每折固定随机取20块训练风险模型，预测未见条件的剩余折。",
             "- group_cv_all：完全相同的折，使用全部约97块拟合风险模型。与group_cv20成对比较监督样本量，但不把差异直接归因于原dev不足。",
             "后两项明确使用历史校准/测试标签做风险模型开发诊断，不能再把这些标签称为未使用的最终校准/测试。",
             "点预测器始终未使用本设置122块标签训练；表示、标准化及寿命扣除只拟合原97块训练电芯。",
             "18设置相互重叠，9/9胜出只描述方向一致性，不是9个独立数据集或显著性检验。", "",
             "## 主设置：CPMLP seed2024、原划分", "", "下表误差损失/大误差Brier改善均相对仅寿命对照，正值更好；AUC差为正表示排序改善。",
             "", "| 诊断 | 附加信息 | 误差损失改善 | 大误差Brier改善 | AUC变化 |", "|---|---|---:|---:|---:|"]
    primary = paired[(paired.job == PRIMARY) & (paired.scope == "pool122")]
    for _, r in primary.iterrows():
        if r.family not in ("life", "constant"):
            lines.append(f"| {r['mode']} | {labels[r.family]} | {r.log_relative_mse_gain_pct:+.1f}% | {r.tail_brier_gain_pct:+.1f}% | {r.tail_auc_delta:+.3f} |")
    lines += ["", "## 所有冻结设置复核", "", "每模型9个设置，列出平均成对改善和改善的设置数。", "",
              "| 模型 | 诊断 | 信息 | 误差损失改善/胜数 | 大误差Brier改善/胜数 |", "|---|---|---|---:|---:|"]
    for _, r in aggregate.iterrows():
        if r.family not in ("life", "constant"):
            lines.append(f"| {r.model} | {r['mode']} | {labels[r.family]} | {r.log_relative_mse_gain_pct:+.1f}% / {r.mse_wins}/9 | {r.tail_brier_gain_pct:+.1f}% / {r.brier_wins}/9 |")
    selected = support[support.job == PRIMARY].iloc[0]
    lines += ["", "## 样本与解释限制", "", f"主设置dev20中大误差事件{int(selected.dev_tail_events)}个；122块历史池中{int(selected.pool_tail_events)}个。20块样本的90%尾部仅约2块，不能支撑复杂尾部分布学习。",
              "Brier与AUC分别评价概率误差和排序；只改善一个不能称为稳定风险信息。风险回归改善也不自动意味着90%区间更有效。",
              "group_cv_all优于group_cv20只能支持在当前审计设计中增加残差监督有帮助，不能保证点预测交叉拟合残差具有相同分布。",
              "本次不按结果挑选带宽/特征参数，不计算新方法的最终CP覆盖，不改变冻结清单、预测文件和checkpoint。", "",
              "完整文件：plan.json、metrics.csv、paired.csv、aggregate.csv、predictions.csv、fold_audit.json、support.csv、audit.json。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, model in zip(axes, ("CPMLP", "CPLSTM")):
        group = aggregate[(aggregate.model == model) & (aggregate['mode'] == "group_cv_all")]
        group = group[~group.family.isin(["constant", "life"])].set_index("family").loc[list(FAMILIES[2:])]
        y = np.arange(len(group))
        ax.barh(y - .17, group.log_relative_mse_gain_pct, height=.32, label="Relative-error loss gain")
        ax.barh(y + .17, group.tail_brier_gain_pct, height=.32, label="Large-error Brier gain")
        ax.set_yticks(y, group.index)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(f"{model}: grouped diagnostic CV, all risk-training cells")
        ax.set_xlabel("Mean improvement vs lifetime only (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "information_gain.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        raise FileExistsError("Audit already prepared; use a new output directory")
    write(plan_path, {"created_at": stamp(), "code_sha256": digest(__file__), "primary": PRIMARY,
                      "families": FAMILIES, "point_predictors": "all 18 frozen; no selection",
                      "ridge_alpha": 10, "logistic_C": .1, "large_error_threshold": .5,
                      "life_basis": "log prediction; training-only quantile knots4, degree2 spline",
                      "extras": "training-only residualization against life basis, Ridge alpha1, standardize and divide sqrt(dim)",
                      "folds": "5-fold GroupKFold by filename G condition, paired training20/all",
                      "seed": 20261028, "hyperparameter_search": False,
                      "status": "post-hoc exploratory; historical cal/test used for diagnostic risk CV; no final CP claim"})
    write(out / "state.json", {"state": "running", "pid": os.getpid(), "started_at": stamp()})
    plan = read(ROOT / "plan.json")
    original = read(ROOT / "candidate_20261026.json")
    curves, inputs = curve_inventory(out, original)
    loaded = {}
    for spec in plan["jobs"]:
        folder = ROOT / "analysis" / spec["key"]
        source = Path(spec["source"])
        frozen = read(source / spec["model"] / "frozen_predictor.json")
        assert digest(source / spec["model"] / "best_model.pt") == frozen["checkpoint_sha256"]
        assert digest(source / "dataset_manifest.json") == frozen["manifest_sha256"]
        assert read(folder / "feature_identity.json")["checkpoint_sha256"] == frozen["checkpoint_sha256"]
        loaded[spec["key"]] = dict(np.load(folder / "features.npz"))
        phases = read(source / "dataset_manifest.json")["split"]["phases"]
        assert len(set(sum(phases.values(), []))) == 239
        for p in PHASES:
            assert list(loaded[spec["key"]][f"{p}_names"]) == phases[p]
            if p != "train":
                csv = source / spec["model"] / f"{p}_predictions.csv"
                previous = pd.read_csv(csv)
                assert previous.cell_file.tolist() == phases[p]
                np.testing.assert_allclose(loaded[spec["key"]][f"{p}_prediction"], np.maximum(0, previous.prediction), rtol=0, atol=1e-9)
                np.testing.assert_allclose(loaded[spec["key"]][f"{p}_eol"], previous.eol, rtol=0, atol=1e-9)
                inputs.append({"path": str(csv), "sha256": digest(csv)})
        inputs.extend({"path": str(p), "sha256": digest(p)} for p in
                      (source / "dataset_manifest.json", source / spec["model"] / "best_model.pt", folder / "features.npz"))
    rows, predictions, audits, supports = [], [], [], []
    for index, spec in enumerate(plan["jobs"]):
        print(f"AUDIT {index + 1}/18 {spec['key']}", flush=True)
        a, b, c, d = run_job(spec, loaded, curves)
        rows.extend(a); predictions.extend(b); audits.extend(c); supports.append(d)
        write(out / "state.json", {"state": "running", "completed_jobs": index + 1, "total_jobs": 18, "updated_at": stamp()})
    metrics = pd.DataFrame(rows)
    paired, aggregate = tables(metrics)
    support = pd.DataFrame(supports)
    for name, table in (("metrics", metrics), ("paired", paired), ("aggregate", aggregate), ("predictions", pd.DataFrame(predictions)), ("support", support)):
        table.to_csv(out / f"{name}.csv", index=False)
    write(out / "fold_audit.json", audits)
    assert all(digest(r["path"]) == r["sha256"] for r in inputs)
    write(out / "audit.json", {"inputs_unchanged": True, "verified_inputs": inputs, "fold_condition_disjoint": True,
                               "feature_preprocessing": "original train97 only", "all_18_jobs_reported": True})
    report(out, metrics, paired, aggregate, support)
    write(out / "state.json", {"state": "complete", "completed_jobs": 18, "updated_at": stamp(), "report": str(out / "REPORT.md")})
    print(f"COMPLETE {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
