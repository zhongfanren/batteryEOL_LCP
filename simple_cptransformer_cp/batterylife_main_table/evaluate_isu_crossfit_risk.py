"""Automatic frozen-risk/CP evaluation after all nested OOF jobs complete.

All learned scales/distances are fitted on train-only OOF97 or original dev20.
Calibration labels are only used for quantiles, never for risk-model fitting.
Historical evaluation remains exploratory.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.preprocessing import StandardScaler

from curated_cp_evaluation import write_json
import run_isu_disagreement_cp as cp


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "isu_crossfit_risk_20261028"
PHASES = ("dev", "calibration", "test")


class RiskScale:
    def __init__(self, prediction, gap, truth, use_gap, alpha=10., quantile=False):
        self.use_gap = use_gap
        self.quantile = quantile
        self.scaler = StandardScaler().fit(self.raw(prediction, gap))
        target = np.log(np.maximum(.02, np.abs(truth - prediction) / np.maximum(1., prediction)))
        estimator = QuantileRegressor(quantile=.9, alpha=.05, solver="highs") if quantile else Ridge(alpha=alpha)
        self.model = estimator.fit(self.design(prediction, gap), target)

    @staticmethod
    def raw(prediction, gap):
        life = np.maximum(1., prediction)
        return np.column_stack((np.log(life), np.log1p(gap / life)))

    def design(self, prediction, gap):
        z = self.scaler.transform(self.raw(prediction, gap))
        return np.column_stack((z[:, 0], z[:, 0]**2, z[:, 1])) if self.use_gap else np.column_stack((z[:, 0], z[:, 0]**2))

    def log_risk(self, prediction, gap):
        return np.clip(self.model.predict(self.design(prediction, gap)), -4., 2.)

    def scale(self, prediction, gap):
        return np.maximum(1., prediction) * np.exp(self.log_risk(prediction, gap))

    def description(self):
        return {"use_gap": self.use_gap, "estimator": "quantile90_L1_alpha.05" if self.quantile else "ridge_alpha10",
                "feature_mean": self.scaler.mean_.tolist(),
                "feature_std": self.scaler.scale_.tolist(), "coefficient": self.model.coef_.tolist(),
                "intercept": self.model.intercept_, "target": "log(max(.02,abs_error/max(1,prediction)))",
                "predicted_log_risk_clip": [-4., 2.]}


def build_bundle(oof, fixed, cohort, prior_scales, model):
    names = oof[oof.model == model].cell_file.tolist()
    tables = {m: oof[oof.model == m].set_index("cell_file").loc[names] for m in ("CPMLP", "CPLSTM")}
    ref_prediction = np.maximum(0., tables[model].prediction.to_numpy())
    ref_gap = np.abs(np.maximum(0., tables["CPMLP"].prediction.to_numpy()) - np.maximum(0., tables["CPLSTM"].prediction.to_numpy()))
    ref_truth = tables[model].eol.to_numpy()
    pair_gap, six_gap = {}, {}
    for p in PHASES:
        pair = {}
        by_arch = {}
        for key, values in cohort.items():
            assert np.array_equal(values[f"{p}_names"], fixed[f"{p}_names"])
            arch = key.rsplit("_", 1)[-1]
            by_arch.setdefault(arch, []).append(values[f"{p}_prediction"])
            if "_r2024_" in key:
                pair[arch] = values[f"{p}_prediction"]
        pair_gap[p] = np.abs(pair["CPMLP"] - pair["CPLSTM"])
        six_gap[p] = np.abs(np.mean(by_arch["CPMLP"], axis=0) - np.mean(by_arch["CPLSTM"], axis=0))
    fit_models = {"oof_life": RiskScale(ref_prediction, ref_gap, ref_truth, False),
                  "oof_pair": RiskScale(ref_prediction, ref_gap, ref_truth, True),
                  "dev20_pair": RiskScale(fixed["dev_prediction"], pair_gap["dev"], fixed["dev_eol"], True),
                  "oof_pair_tail": RiskScale(ref_prediction, ref_gap, ref_truth, True, quantile=True),
                  "dev20_pair_tail": RiskScale(fixed["dev_prediction"], pair_gap["dev"], fixed["dev_eol"], True, quantile=True)}
    life_mean, life_std = np.log(np.maximum(1., ref_prediction)).mean(), max(.1, np.log(np.maximum(1., ref_prediction)).std())
    ref_risk = fit_models["oof_pair_tail"].log_risk(ref_prediction, ref_gap)
    risk_mean, risk_std = ref_risk.mean(), max(.1, ref_risk.std())
    scale, coordinates, risk = {}, {}, {}
    dev_errors = np.abs(fixed["dev_eol"] - fixed["dev_prediction"])
    dev_z = (fixed["dev_prediction"] - prior_scales["prediction_mean"]) / prior_scales["prediction_std"]
    for p in PHASES:
        pred = np.maximum(0., fixed[f"{p}_prediction"])
        life = np.maximum(1., pred)
        scale[p] = {"split_cp": np.ones(len(pred)), "relative_cp": life,
                    "gap6_scale_cp": life + six_gap[p], "pair_gap_scale_cp": life + pair_gap[p]}
        for name, estimator in fit_models.items():
            scale[p][f"{name}_scale_cp"] = estimator.scale(pred, pair_gap[p])
        query_z = (pred - prior_scales["prediction_mean"]) / prior_scales["prediction_std"]
        logs = -(query_z[:, None] - dev_z[None, :])**2 / 4.5
        weights = np.exp(logs - logs.max(axis=1, keepdims=True))
        scale[p]["prediction_scale_cp"] = np.maximum(max(1., .1*np.median(dev_errors)), weights @ dev_errors / weights.sum(axis=1))
        risk[p] = fit_models["oof_pair_tail"].log_risk(pred, pair_gap[p])
        coordinates[p] = {"life": ((np.log(life) - life_mean) / life_std)[:, None],
                          "learned_risk": ((risk[p] - risk_mean) / risk_std)[:, None]}
    gap_cuts = np.quantile(pair_gap["dev"] / np.maximum(1., fixed["dev_prediction"]), [1/3, 2/3])
    risk_cuts = np.quantile(risk["dev"], [1/3, 2/3])
    groups = {p: {"pair_gap": np.searchsorted(gap_cuts, pair_gap[p] / np.maximum(1., fixed[f"{p}_prediction"]), side="right"),
                  "learned_risk": np.searchsorted(risk_cuts, risk[p], side="right")} for p in PHASES}
    return {"scales": scale, "coordinates": coordinates, "groups": groups, "risk": risk, "pair_gap": pair_gap,
            "descriptions": {k: v.description() for k, v in fit_models.items()},
            "normalization": {"life_mean": life_mean, "life_std": life_std, "risk_mean": risk_mean, "risk_std": risk_std,
                              "pair_gap_cutpoints": gap_cuts.tolist(), "learned_risk_cutpoints": risk_cuts.tolist()},
            "reference": {"prediction": ref_prediction, "gap": ref_gap, "truth": ref_truth,
                          "n": len(names), "large_errors": int(np.sum(np.abs(ref_truth-ref_prediction)/np.maximum(1.,ref_prediction)>.5))}}


def radii_from_calibration(scales, coordinates, prediction, truth, ci, ti, repetition):
    error = np.abs(truth[ci] - prediction[ci])
    radii = {name: cp.finite_quantile(error / values[ci]) * values[ti] for name, values in scales.items()}
    diagnostics = {}
    normals = np.random.default_rng(20261028 + repetition).normal(size=(len(ti), 1))
    for location, z in coordinates.items():
        for bandwidth in (3., 6.):
            weights, ess, mass = cp.random_weights(z[ci], z[ti], bandwidth, normals)
            for score, scale in (("relative", scales["relative_cp"]), ("oof_tail", scales["oof_pair_tail_scale_cp"])):
                name = f"{score}_rlcp_{location}_h{bandwidth:g}"
                radii[name] = cp.weighted_quantiles(error / scale[ci], weights) * scale[ti]
                diagnostics[name] = {"mean_ess": ess.mean(), "mean_self_mass": mass.mean()}
    return radii, diagnostics


def evaluate_model(fixed, bundle, model, repetitions):
    names = np.r_[fixed["calibration_names"], fixed["test_names"]]
    prediction = np.r_[fixed["calibration_prediction"], fixed["test_prediction"]]
    truth = np.r_[fixed["calibration_eol"], fixed["test_eol"]]
    scales = {name: np.r_[bundle["scales"]["calibration"][name], bundle["scales"]["test"][name]] for name in bundle["scales"]["calibration"]}
    coordinates = {name: np.r_[bundle["coordinates"]["calibration"][name], bundle["coordinates"]["test"][name]] for name in ("life", "learned_risk")}
    group_values = {axis: np.r_[bundle["groups"]["calibration"][axis], bundle["groups"]["test"][axis]] for axis in ("pair_gap", "learned_risk")}
    allocation, metrics, original, support = {}, [], [], []
    rng = np.random.default_rng(20261028)
    relative_error = np.abs(truth - prediction) / np.maximum(1., prediction)
    for axis, values in group_values.items():
        for group in range(3):
            keep = values == group
            support.append({"model": model, "axis": axis, "group": group, "pool_n": int(keep.sum()),
                            "mean_relative_error": relative_error[keep].mean() if keep.any() else np.nan,
                            "mean_absolute_error": np.abs(truth[keep]-prediction[keep]).mean() if keep.any() else np.nan,
                            "large_error_fraction": np.mean(relative_error[keep]>.5) if keep.any() else np.nan})
    for repetition in range(repetitions + 1):
        permutation = np.arange(122) if repetition == 0 else rng.permutation(122)
        ci, ti = permutation[:61], permutation[61:]
        radii, diagnostics = radii_from_calibration(scales, coordinates, prediction, truth, ci, ti, repetition)
        for method, radius in radii.items():
            lower, upper, width, covered, score = cp.cell_metrics(truth[ti], prediction[ti], radius)
            metrics.append({"model": model, "method": method, "repetition": repetition, "coverage": covered.mean(),
                            "mean_width": width.mean(), "interval_score": score.mean(), "infinite": int(np.isinf(radius).sum()),
                            "relative_width": np.mean(width/np.maximum(1.,prediction[ti])), **diagnostics.get(method, {})})
            phase = "original" if repetition == 0 else "resplits"
            for axis, values in group_values.items():
                for group in range(3):
                    cp.add_group(allocation, (phase, method, axis, group), values[ti] == group, width, covered, score, prediction[ti])
            if repetition == 0:
                for j, i in enumerate(ti):
                    original.append({"model": model, "method": method, "cell_file": names[i], "prediction": prediction[i], "eol": truth[i],
                                     "radius": radius[j], "lower": lower[j], "upper": upper[j], "covered": bool(covered[j]),
                                     "width": width[j], "interval_score": score[j], "pair_gap_group": group_values["pair_gap"][i],
                                     "learned_risk_group": group_values["learned_risk"][i]})
    groups = []
    for (phase, method, axis, group), values in allocation.items():
        count, covered, width, score, relative_width = values
        groups.append({"model": model, "phase": phase, "method": method, "axis": axis, "group": group,
                       "presentations": int(count), "coverage": covered/count, "mean_width": width/count,
                       "interval_score": score/count, "relative_width": relative_width/count})
    return metrics, original, groups, support


def verify_and_report(out, metrics, original, groups, support, diagnostics, loaded):
    for model in ("CPMLP", "CPLSTM"):
        job = f"s20261026_r2024_{model}"
        prior = pd.read_csv(cp.ROOT / "disagreement_cp_20261028/metrics.csv")
        prior = prior[prior.job == job].set_index(["method", "repetition"])
        for method in ("split_cp", "relative_cp", "prediction_scale_cp", "gap6_scale_cp"):
            previous_name = "gap_scale_cp" if method == "gap6_scale_cp" else method
            rows = metrics[(metrics.model == model) & (metrics.method == method)]
            ref = prior.loc[pd.MultiIndex.from_arrays([[previous_name]*len(rows), rows.repetition])]
            for column in ("coverage", "mean_width", "interval_score"):
                np.testing.assert_allclose(rows[column], ref[column], rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(original.width, original.upper-original.lower)
    np.testing.assert_array_equal(original.covered, (original.lower<=original.eol)&(original.eol<=original.upper))
    recomputed = original.width + 20*(np.maximum(original.lower-original.eol,0)+np.maximum(original.eol-original.upper,0))
    np.testing.assert_allclose(original.interval_score, recomputed)
    summary = metrics[metrics.repetition>0].groupby(["model","method"])[["coverage","mean_width","interval_score","relative_width"]].mean()
    # pandas groupby sums can turn repeated infinities into NaNs; retain them explicitly.
    for key, rows in metrics[metrics.repetition>0].groupby(["model","method"]):
        for column in ("mean_width","interval_score","relative_width"):
            summary.loc[key,column] = np.mean(rows[column].to_numpy())
        summary.loc[key,"infinite_cell_fraction"] = rows.infinite.mean()/61
        summary.loc[key,"infinite_set_fraction"] = np.mean(rows.infinite>0)
    summary.to_csv(out/"summary.csv")
    lines = ["# 训练侧交叉拟合风险学习：自动结果简报", "", "所有辅助训练和风险拟合均在原train97或原dev20侧；OOF误差没有用于最终CP分位数。",
             "主预测器仍为原CPMLP seed2024；CPLSTM seed2024作固定复核。原cal61/test61保持原成员，另外200次固定池重分仅作探索。", "",
             "## 独立误差样本", "", "每块训练电芯由没有见过它的辅助模型预测；内部验证选epoch后，在外折训练侧从头重训。",
             "得到97块不同电芯的外折误差，并不意味着97个统计独立的拟合残差：不同辅助模型训练集重叠。也不是新获得97块数据。",
             "辅助模型每折约77块训练、最终主模型97块，条件误差分布可能不同；分数最后仍由原61块校准电芯校准。", "",
             "## 风险诊断", "", "| 预测器 | OOF大误差数/97 | dev大误差数/20 | OOF风险与历史池相对误差rho |", "|---|---:|---:|---:|"]
    for row in diagnostics:
        lines.append(f"| {row['model']} | {row['oof_large_errors']} | {row['dev_large_errors']} | {row['pool_relative_risk_rho']:.3f} |")
    lines += ["", "大误差定义仍为误差>预测寿命50%；rho是风险排序诊断，不是90%条件分位数保证。", "",
              "风险尺度同时保留Ridge的log误差平均趋势和固定正则化的90%分位数回归（tail）。LCP的learned_risk距离使用后者；估计尾部仍不自动保证条件90%覆盖。",
              "## 冻结分数后的平均重分表现", "", "| 预测器 | 方法 | 覆盖 | 宽度 | Interval score | 无穷比例 |", "|---|---|---:|---:|---:|---:|"]
    for (model,method),r in summary.iterrows():
        lines.append(f"| {model} | {method} | {r.coverage:.2%} | {r.mean_width:.1f} | {r.interval_score:.1f} | {r.infinite_cell_fraction:.3%} |")
    lines += ["", "## 原61块test（单独保留，不能被重分平均掩盖）", "", "| 预测器 | 方法 | 覆盖 | 宽度 | Interval score |", "|---|---|---:|---:|---:|"]
    main_methods = ("relative_cp","gap6_scale_cp","pair_gap_scale_cp","dev20_pair_scale_cp","oof_life_scale_cp","oof_pair_scale_cp",
                    "dev20_pair_tail_scale_cp","oof_pair_tail_scale_cp")
    for r in metrics[(metrics.repetition==0)&metrics.method.isin(main_methods)].itertuples():
        lines.append(f"| {r.model} | {r.method} | {r.coverage:.2%} | {r.mean_width:.1f} | {r.interval_score:.1f} |")
    lines += ["", "## 如何归因", "", "- dev20_pair与oof_pair比较：同一低容量风险算法，残差参考样本及预测器训练规模不同，不能把差异全部归因于样本数。",
              "- oof_life与oof_pair比较：相同97块OOF残差、相同寿命基函数，仅增加跨架构分歧。",
              "- relative/gap6/pair_gap是无需残差拟合的强对照；旧prediction_scale保留。",
              "- dev20_pair_tail与oof_pair_tail对比直接90%尾部尺度学习；模型正则强度、特征和目标保持相同。",
              "- RLCP的relative分数对照relative_cp；oof_tail分数对照oof_pair_tail_scale_cp。每个分数下比较life与learned_risk距离，h3/h6全部报告。",
              "- pair_gap分组使用dev可观察单种子模型对分歧；learned_risk分组用dev预测风险阈值，均未用cal/test误差定组。它们与此前六模型分歧组不同。",
              "分组真实误差见group_support.csv，覆盖/宽度/score见group_metrics.csv；高预测风险组更宽本身不是有效证据。",
              "全体无穷保留；历史标签早已查看，跨设置和重分有依赖，均不是新的确认性覆盖或每组90%保证。", "",
              "已检查四个冻结基线与上一批精确一致、原test逐电芯指标算术一致。完整模型系数、分折身份和来源指纹均已保存。"]
    (out/"BRIEF.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args(); out=args.out.resolve()
    plan=cp.read(out/"plan.json")
    assert cp.digest(__file__)==plan["evaluator_sha256"]
    assert cp.digest(out/"dataset_manifest.json")==plan["manifest_sha256"]
    assert all(cp.digest(row["path"])==row["sha256"] for row in plan["dependencies"])
    bank_identity=cp.read(out/"bank_identity.json")
    assert cp.digest(out/"train_bank.npz")==bank_identity["bank_sha256"]
    oof=pd.read_csv(out/"oof_predictions.csv")
    assert len(oof)==194
    phases=cp.read(out/"dataset_manifest.json")["split"]["phases"]
    for model in ("CPMLP","CPLSTM"):
        rows=oof[oof.model==model]
        assert rows.cell_file.nunique()==97 and set(rows.cell_file)==set(phases["train"])
    for job in plan["jobs"]:
        completion=cp.read(out/"folds"/job["key"]/"complete.json")
        assert completion["job"]==job and not completion["smoke_only"]
        assert not set(job["outer_fit"]) & set(job["outer_holdout"])
        assert cp.digest(out/"folds"/job["key"]/"auxiliary_model.pt")==completion["checkpoint_sha256"]
    parent=cp.read(cp.ROOT/"plan.json")
    cohort={s["key"]:dict(np.load(cp.ROOT/"analysis"/s["key"]/"features.npz")) for s in parent["jobs"] if s["split_seed"]==20261026}
    rows,originals,group_rows,support_rows,diagnostics,query_rows=[],[],[],[],[],[]
    inputs=[]
    for model in ("CPMLP","CPLSTM"):
        spec=next(s for s in parent["jobs"] if s["key"]==f"s20261026_r2024_{model}")
        fixed=cohort[spec["key"]]
        source=Path(spec["source"])/model
        frozen=cp.read(source/"frozen_predictor.json")
        assert cp.digest(source/"best_model.pt")==frozen["checkpoint_sha256"]
        inputs.append({"path":str(source/"best_model.pt"),"sha256":frozen["checkpoint_sha256"]})
        bundle=build_bundle(oof,fixed,cohort,cp.read(source/"results.json")["scales"],model)
        # Persist all risk parameters before evaluating any calibration/test errors.
        write_json(out/f"{model}_frozen_risk_models.json", {"models":bundle["descriptions"],"normalization":bundle["normalization"],
                   "fit_sources":{"oof":phases["train"],"dev20":phases["dev"]},"oof_sha256":cp.digest(out/"oof_predictions.csv")})
        for phase in PHASES:
            for i,name in enumerate(fixed[f"{phase}_names"]):
                query_rows.append({"model":model,"phase":phase,"cell_file":name,"prediction":fixed[f"{phase}_prediction"][i],
                                   "pair_gap":bundle["pair_gap"][phase][i],"log_risk":bundle["risk"][phase][i],
                                   **{k:v[i] for k,v in bundle["scales"][phase].items()}})
        a,b,c,d=evaluate_model(fixed,bundle,model,plan["cp"]["holdout_resplits"])
        rows.extend(a); originals.extend(b); group_rows.extend(c); support_rows.extend(d)
        truth=np.r_[fixed["calibration_eol"],fixed["test_eol"]]
        pred=np.r_[fixed["calibration_prediction"],fixed["test_prediction"]]
        risk=np.r_[bundle["risk"]["calibration"],bundle["risk"]["test"]]
        diagnostics.append({"model":model,"oof_large_errors":bundle["reference"]["large_errors"],
                            "dev_large_errors":int(np.sum(np.abs(fixed["dev_eol"]-fixed["dev_prediction"])/np.maximum(1.,fixed["dev_prediction"])>.5)),
                            "pool_relative_risk_rho":float(spearmanr(risk,np.abs(truth-pred)/np.maximum(1.,pred)).statistic),
                            "oof_mean_relative_error":float(np.mean(np.abs(bundle["reference"]["truth"]-bundle["reference"]["prediction"])/np.maximum(1.,bundle["reference"]["prediction"]))),
                            "pool_mean_relative_error":float(np.mean(np.abs(truth-pred)/np.maximum(1.,pred)))})
    metrics,original,groups,support=map(pd.DataFrame,(rows,originals,group_rows,support_rows))
    for name,table in (("metrics",metrics),("original_intervals",original),("group_metrics",groups),("group_support",support),
                       ("query_scales",pd.DataFrame(query_rows)),("risk_diagnostics",pd.DataFrame(diagnostics))):
        table.to_csv(out/f"{name}.csv",index=False)
    verify_and_report(out,metrics,original,groups,support,diagnostics,cohort)
    assert all(cp.digest(i["path"])==i["sha256"] for i in inputs)
    write_json(out/"evaluation_complete.json",{"completed_at":cp.stamp(),"rows":len(metrics),"frozen_sources_unchanged":True,
               "risk_fit_only_train_oof_and_dev":True,"oof_not_used_as_final_calibration":True,"sources":inputs})
    print(f"EVALUATION COMPLETE {out/'BRIEF.md'}",flush=True)


if __name__=="__main__":
    main()
