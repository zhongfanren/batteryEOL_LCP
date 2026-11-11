"""Explicitly post-hoc decomposition of the risk-information audit's signal.

No feature/parameter search: report both disagreement components and a
life-controlled absolute-error regression sensitivity check for every fit.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import audit_isu_risk_information as audit


OUT = audit.DEFAULT_OUT / "disagreement_followup"


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "plan.json").exists():
        raise FileExistsError("Choose a new output directory")
    audit.write(OUT / "plan.json", {
        "created_at": audit.stamp(), "code_sha256": audit.digest(__file__),
        "parent_plan_sha256": audit.digest(audit.DEFAULT_OUT / "plan.json"),
        "reason": "post-hoc follow-up: combined model disagreement was the clearest exploratory signal",
        "components": ["seed_spread", "architecture_gap"],
        "risk_modes": ["dev20_to_pool", "group_cv20", "group_cv_all"],
        "absolute_error_sensitivity": "same Ridge alpha10 and lifetime basis; target log1p(abs error in cycles)",
        "selection": "report both components, both models and all fits; no tuned parameter or fresh CP claim"})
    plan = audit.read(audit.ROOT / "plan.json")
    curves = pd.read_csv(audit.DEFAULT_OUT / "curve_inventory.csv").set_index("cell_file")
    loaded = {s["key"]: dict(np.load(audit.ROOT / "analysis" / s["key"] / "features.npz")) for s in plan["jobs"]}
    rows, raw_rows = [], []
    for spec in plan["jobs"]:
        job = spec["key"]
        manifest = audit.read(Path(spec["source"]) / "dataset_manifest.json")
        manifest["audit_model"] = spec["model"]
        data = loaded[job]
        cohort = {k: v for k, v in loaded.items() if k.startswith(f"s{spec['split_seed']}_")}
        design = audit.feature_design(data, manifest, cohort, curves)
        for component, column in (("seed_spread", -2), ("architecture_gap", -1)):
            # Parent divides a two-feature block by sqrt(2); each ablated block
            # has one feature, so undo that factor with no new scaling fit.
            design[component] = {p: np.column_stack((design["life"][p], design["disagreement"][p][:, column] * np.sqrt(2))) for p in audit.PHASES}
        names = np.r_[data["calibration_names"], data["test_names"]]
        pred = np.r_[data["calibration_prediction"], data["test_prediction"]]
        truth = np.r_[data["calibration_eol"], data["test_eol"]]
        absolute = np.abs(truth - pred)
        relative = absolute / np.maximum(1., pred)
        dev_absolute = np.abs(data["dev_eol"] - data["dev_prediction"])
        dev_relative = dev_absolute / np.maximum(1., data["dev_prediction"])
        groups = [audit.condition(n) for n in names]
        folds = list(GroupKFold(n_splits=5).split(names, groups=groups))
        for family in ("life", "disagreement", "seed_spread", "architecture_gap"):
            x = np.concatenate((design[family]["calibration"], design[family]["test"]))
            for mode in ("dev20_to_pool", "group_cv20", "group_cv_all"):
                if mode == "dev20_to_pool":
                    risk, probability = audit.fit_predict(design[family]["dev"], dev_relative, x)
                    absolute_risk = Ridge(alpha=10.).fit(design[family]["dev"], np.log1p(dev_absolute)).predict(x)
                else:
                    risk, probability, absolute_risk = (np.full(len(x), np.nan) for _ in range(3))
                    for fold, (train, test) in enumerate(folds):
                        subset = train if mode == "group_cv_all" else np.random.default_rng(20261028 + fold).choice(train, 20, replace=False)
                        risk[test], probability[test] = audit.fit_predict(x[subset], relative[subset], x[test])
                        absolute_risk[test] = Ridge(alpha=10.).fit(x[subset], np.log1p(absolute[subset])).predict(x[test])
                values = audit.scores(relative, risk, probability)
                values["log_absolute_mse"] = np.mean((absolute_risk - np.log1p(absolute)) ** 2)
                rows.append({"job": job, "model": spec["model"], "family": family, "mode": mode, **values})
                for i, name in enumerate(names):
                    raw_rows.append({"job": job, "cell_file": name, "family": family, "mode": mode,
                                     "relative_risk": risk[i], "tail_probability": probability[i], "absolute_risk": absolute_risk[i]})
    metrics = pd.DataFrame(rows)
    references = metrics[metrics.family == "life"].set_index(["job", "mode"])
    reference = references.loc[pd.MultiIndex.from_frame(metrics[["job", "mode"]])]
    for loss in ("log_relative_mse", "tail_brier", "log_absolute_mse"):
        metrics[f"{loss}_gain_pct"] = 100 * (1 - metrics[loss].to_numpy() / reference[loss].to_numpy())
    parent = pd.read_csv(audit.DEFAULT_OUT / "metrics.csv")
    matched = parent[(parent.scope == "pool122") & parent.family.isin(["life", "disagreement"])].set_index(["job", "family", "mode"])
    check = metrics[metrics.family.isin(["life", "disagreement"])].set_index(["job", "family", "mode"])
    for loss in ("log_relative_mse", "tail_brier", "tail_auc"):
        np.testing.assert_allclose(check[loss], matched.loc[check.index, loss], rtol=1e-10, atol=1e-12)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(OUT / "predictions.csv", index=False)
    summary = []
    for keys, group in metrics.groupby(["model", "mode", "family"]):
        row = dict(zip(("model", "mode", "family"), keys))
        for loss in ("log_relative_mse", "tail_brier", "log_absolute_mse"):
            row[f"{loss}_gain_pct"] = group[f"{loss}_gain_pct"].mean()
            row[f"{loss}_wins"] = int((group[f"{loss}_gain_pct"] > 0).sum())
        summary.append(row)
    aggregate = pd.DataFrame(summary)
    aggregate.to_csv(OUT / "aggregate.csv", index=False)
    lines = ["# 模型分歧的追加拆解", "", "明确属于看到主审计结果后的探索性检查；不是预先独立验证。所有参数沿用主审计。", "",
             "seed_spread：同架构三个训练种子的分歧，需要3模型；architecture_gap：两个架构各三种子均值的差，需要6模型。",
             "disagreement：两者组合，需要6模型。点预测中心保持原单模型，不采用ensemble均值替代。",
             "绝对误差检查依然控制预测寿命，但不使用相对误差分母，用于检查信号是否仅由共同分母驱动。", "",
             "下表为历史池按实验条件5折、每折全部风险训练样本的结果。正值表示相对仅寿命对照改善。", "",
             "| 模型 | 分歧信息 | 相对误差损失改善/胜数 | 大误差Brier改善/胜数 | 绝对误差损失改善/胜数 |",
             "|---|---|---:|---:|---:|"]
    for _, r in aggregate[(aggregate['mode'] == "group_cv_all") & (aggregate.family != "life")].iterrows():
        lines.append(f"| {r.model} | {r.family} | {r.log_relative_mse_gain_pct:+.1f}% / {r.log_relative_mse_wins}/9 | {r.tail_brier_gain_pct:+.1f}% / {r.tail_brier_wins}/9 | {r.log_absolute_mse_gain_pct:+.1f}% / {r.log_absolute_mse_wins}/9 |")
    lines += ["", "主审计中life和disagreement的MSE、Brier、AUC已逐项重算一致。所有模式与主设置结果见metrics.csv，逐电芯预测见predictions.csv。",
              "这些结果不证明最佳距离已确定，不证明单种子或单个跨架构差值也能复现，不证明90%条件分位数可准确估计。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit.write(OUT / "audit.json", {"parent_metrics_reproduced": True, "jobs": 18, "no_point_training": True})
    print(aggregate[(aggregate['mode'] == "group_cv_all") & (aggregate.family != "life")].to_string(index=False))
    print(f"COMPLETE {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
