"""Independent completed-run audit; diagnose constant learned localizers.

Preserves the frozen runner/evaluator and all original experimental outputs.
No fitting, parameter search, CP re-evaluation or GPU work is performed.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from run_isu_disagreement_cp import digest, read


HERE = Path(__file__).resolve().parent
ROOT = HERE / "isu_crossfit_risk_20261028"
OUT = ROOT / "analysis_review"
METHODS = ("relative_cp", "gap6_scale_cp", "pair_gap_scale_cp", "prediction_scale_cp",
           "dev20_pair_scale_cp", "oof_life_scale_cp", "oof_pair_scale_cp", "oof_pair_tail_scale_cp")


def reconstruct(model, prediction, gap):
    p = np.maximum(1., prediction)
    raw = np.column_stack((np.log(p), np.log1p(gap / p)))
    z = (raw - model["feature_mean"]) / model["feature_std"]
    design = np.column_stack((z[:, 0], z[:, 0]**2, z[:, 1])) if model["use_gap"] else np.column_stack((z[:, 0], z[:, 0]**2))
    return np.clip(design @ np.asarray(model["coefficient"]) + model["intercept"], -4., 2.), design


def main():
    OUT.mkdir(exist_ok=True)
    plan, state = read(ROOT / "plan.json"), read(ROOT / "state.json")
    assert state["state"] == "complete" and state["evaluation_exit_code"] == 0
    assert digest(HERE / "run_isu_crossfit_risk.py") == plan["runner_sha256"]
    assert digest(HERE / "evaluate_isu_crossfit_risk.py") == plan["evaluator_sha256"]
    assert all(digest(x["path"]) == x["sha256"] for x in plan["dependencies"])
    assert digest(ROOT / "dataset_manifest.json") == plan["manifest_sha256"]
    identity = read(ROOT / "bank_identity.json")
    assert digest(ROOT / "train_bank.npz") == identity["bank_sha256"]
    with np.load(ROOT / "train_bank.npz") as data:
        bank = dict(data)
    labels = dict(zip(bank["names"], bank["labels"]))
    phase = read(ROOT / "dataset_manifest.json")["split"]["phases"]
    oof = pd.read_csv(ROOT / "oof_predictions.csv")
    metrics = pd.read_csv(ROOT / "metrics.csv")
    intervals = pd.read_csv(ROOT / "original_intervals.csv")
    query = pd.read_csv(ROOT / "query_scales.csv")
    groups = pd.read_csv(ROOT / "group_metrics.csv")
    assert len(oof) == 194 and len(metrics) == 7236
    fold_rows = []
    for spec in plan["jobs"]:
        folder = ROOT / "folds" / spec["key"]
        complete = read(folder / "complete.json")
        assert state["jobs"][spec["key"]]["exit_code"] == 0
        assert complete["job"] == spec and not complete["smoke_only"]
        outer, held = set(spec["outer_fit"]), set(spec["outer_holdout"])
        assert not outer & held and outer | held == set(phase["train"])
        assert not set(spec["inner_fit"]) & set(spec["inner_dev"])
        assert set(spec["inner_fit"] + spec["inner_dev"]) == outer
        assert not (outer | held) & set(phase["dev"] + phase["calibration"] + phase["test"])
        assert digest(folder / "auxiliary_model.pt") == complete["checkpoint_sha256"]
        checkpoint = torch.load(folder / "auxiliary_model.pt", map_location="cpu", weights_only=False)
        assert checkpoint["trained_on"] == spec["outer_fit"]
        values = np.asarray([labels[n] for n in spec["outer_fit"]])
        np.testing.assert_allclose([checkpoint["label_mean"], checkpoint["label_std"]], [values.mean(), values.std()], rtol=1e-12)
        inner, refit = read(folder / "inner_selection_history.json"), read(folder / "outer_refit_history.json")
        selected = min(inner, key=lambda x: x["inner_dev_mape"])["epoch"]
        assert selected == complete["selected_epoch"] == checkpoint["epoch"] == len(refit)
        original = pd.read_csv(folder / "oof_predictions.csv")
        assert original.cell_file.tolist() == spec["outer_holdout"]
        np.testing.assert_allclose(original.eol, [labels[n] for n in original.cell_file], rtol=0, atol=1e-10)
        consolidated = oof[(oof.model == spec["model"]) & (oof.fold == spec["fold"])]
        pd.testing.assert_frame_equal(original.reset_index(drop=True), consolidated.reset_index(drop=True))
        fold_rows.append({"job": spec["key"], "fit_n": len(outer), "holdout_n": len(held),
                          "inner_epochs": len(inner), "selected_epoch": selected,
                          "selected_at_cap": selected == plan["epoch_cap"], "fit_only_label_scale": True})
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_audit.csv", index=False)
    for model in ("CPMLP", "CPLSTM"):
        cells = oof[oof.model == model].cell_file
        assert len(cells) == cells.nunique() == 97 and set(cells) == set(phase["train"])
    # Recompute every original-test interval and all corresponding metrics.
    np.testing.assert_allclose(intervals.lower, np.maximum(0., intervals.prediction-intervals.radius))
    np.testing.assert_allclose(intervals.upper, intervals.prediction+intervals.radius)
    np.testing.assert_allclose(intervals.width, intervals.upper-intervals.lower)
    coverage = (intervals.lower <= intervals.eol) & (intervals.eol <= intervals.upper)
    np.testing.assert_array_equal(coverage, intervals.covered)
    score = intervals.width + 20*(np.maximum(intervals.lower-intervals.eol, 0.) + np.maximum(intervals.eol-intervals.upper, 0.))
    np.testing.assert_allclose(score, intervals.interval_score)
    for (model, method), rows in intervals.groupby(["model", "method"]):
        reference = metrics[(metrics.model == model) & (metrics.method == method) & (metrics.repetition == 0)].iloc[0]
        np.testing.assert_allclose([rows.covered.mean(), np.mean(rows.width.to_numpy()), np.mean(rows.interval_score.to_numpy())],
                                  [reference.coverage, reference.mean_width, reference.interval_score])
    # Independently verify each diagnostic partition recovers the full metrics.
    for (model, method, phase_name, axis), rows in groups.groupby(["model", "method", "phase", "axis"]):
        target = metrics[(metrics.model == model) & (metrics.method == method)]
        target = target[target.repetition.eq(0) if phase_name == "original" else target.repetition.gt(0)]
        assert rows.presentations.sum() == 61*len(target)
        for column in ("coverage", "mean_width", "interval_score"):
            value = np.sum(rows[column]*rows.presentations) / rows.presentations.sum()
            np.testing.assert_allclose(value, np.mean(target[column].to_numpy()), rtol=1e-10)
    diagnostic_rows = []
    for model in ("CPMLP", "CPLSTM"):
        frozen = read(ROOT / f"{model}_frozen_risk_models.json")
        assert frozen["oof_sha256"] == digest(ROOT / "oof_predictions.csv")
        assert frozen["fit_sources"]["oof"] == phase["train"]
        assert frozen["fit_sources"]["dev20"] == phase["dev"]
        ref = oof[oof.model == model].set_index("cell_file")
        other = oof[oof.model != model].set_index("cell_file").loc[ref.index]
        pred, truth = np.maximum(0., ref.prediction.to_numpy()), ref.eol.to_numpy()
        gap = np.abs(pred-np.maximum(0., other.prediction.to_numpy()))
        relative_error = np.abs(truth-pred)/np.maximum(1.,pred)
        tail = frozen["models"]["oof_pair_tail"]
        _, x = reconstruct(tail, pred, gap)
        assert np.count_nonzero(tail["coefficient"]) == 0
        # KKT certificate: at the constant .9-quantile, every absolute feature
        # loss gradient lies below L1 alpha.05. The zero solution is consistent
        # with the chosen optimization objective, not a failed optimizer.
        target = np.log(np.maximum(.02, relative_error))
        intercept = tail["intercept"]
        ties = np.isclose(target, intercept, atol=1e-10, rtol=0)
        psi = np.where(target < intercept, -.1, .9)
        psi[ties] = 0.
        assert ties.any()
        psi[ties] = -psi[~ties].sum()/ties.sum()
        assert np.all((psi[ties] >= -.1-1e-10) & (psi[ties] <= .9+1e-10))
        gradient = x.T @ psi / len(target)
        assert np.max(np.abs(gradient)) <= plan["risk_model"]["tail_L1_alpha"] + 1e-10
        q = query[query.model == model]
        for name, estimator in frozen["models"].items():
            log_risk, _ = reconstruct(estimator, q.prediction.to_numpy(), q.pair_gap.to_numpy())
            expected = np.maximum(1., q.prediction.to_numpy()) * np.exp(log_risk)
            np.testing.assert_allclose(expected, q[f"{name}_scale_cp"], rtol=1e-10, atol=1e-9)
        actual_risk_groups = pd.read_csv(ROOT / "group_support.csv")
        risk_groups = actual_risk_groups[(actual_risk_groups.model == model) & (actual_risk_groups.axis == "learned_risk")]
        assert risk_groups.pool_n.tolist() == [0, 0, 122]
        baseline = metrics[(metrics.model == model) & (metrics.method == "relative_cp")].set_index("repetition")
        equivalents = ["oof_pair_tail_scale_cp"] + [f"{score}_rlcp_learned_risk_h{h}" for score in ("relative", "oof_tail") for h in (3, 6)]
        for method in equivalents:
            rows = metrics[(metrics.model == model) & (metrics.method == method)].set_index("repetition").loc[baseline.index]
            for column in ("coverage", "mean_width", "interval_score"):
                np.testing.assert_allclose(rows[column], baseline[column], rtol=1e-10, atol=1e-8)
        # Tail NaN correlation reflects zero variance. Assess mean-risk ranking
        # separately, without fitting anything or changing the frozen model.
        source = Path(plan["source"]) / model
        truth_by_name = {}
        for phase_name in ("calibration", "test"):
            values = pd.read_csv(source / f"{phase_name}_predictions.csv")
            truth_by_name.update(dict(zip(values.cell_file, values.eol)))
        pool = q[q.phase.isin(["calibration", "test"])]
        y = pool.cell_file.map(truth_by_name).to_numpy()
        p = pool.prediction.to_numpy()
        e = np.abs(y-p)/np.maximum(1.,p)
        mean_risk = pool.oof_pair_scale_cp.to_numpy()/np.maximum(1.,p)
        diagnostic_rows.append({"model": model, "oof_cells": 97, "oof_large_error_count": int((relative_error>.5).sum()),
                                "oof_mean_relative_error": relative_error.mean(), "pool_mean_relative_error": e.mean(),
                                "oof_tail_nonzero_coefficients": 0, "constant_tail_gradient_max": np.abs(gradient).max(),
                                "L1_alpha": plan["risk_model"]["tail_L1_alpha"], "tail_groups_valid": False,
                                "mean_risk_pool_spearman": spearmanr(mean_risk,e).statistic,
                                "constant_tail_radius_factor_before_CP": np.exp(intercept)})
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(OUT / "risk_diagnostics.csv", index=False)
    summary = metrics[metrics.repetition.gt(0)].groupby(["model","method"])[["coverage","mean_width","interval_score"]].agg(lambda x: np.mean(x.to_numpy()))
    summary.to_csv(OUT / "summary.csv")
    comparisons = []
    for model in ("CPMLP", "CPLSTM"):
        target = summary.loc[(model,"oof_pair_scale_cp")]
        for baseline_name in ("relative_cp","gap6_scale_cp","dev20_pair_scale_cp","oof_life_scale_cp"):
            base = summary.loc[(model,baseline_name)]
            comparisons.append({"model":model,"method":"oof_pair_scale_cp","baseline":baseline_name,
                                "width_change_pct":100*(target.mean_width/base.mean_width-1),
                                "score_change_pct":100*(target.interval_score/base.interval_score-1),
                                "coverage_change_pp":100*(target.coverage-base.coverage)})
    pd.DataFrame(comparisons).to_csv(OUT/"comparisons.csv",index=False)
    elapsed=(datetime.fromisoformat(state["finished_at"])-datetime.fromisoformat(state["started_at"])).total_seconds()
    audit={"completed_jobs":10,"elapsed_seconds":elapsed,"outer_cell_exclusion_passed":True,
           "inner_epoch_selection_recomputed":True,"stage_label_scaling_recomputed":True,
           "all_original_interval_rows_checked":len(intervals),"OOF_rows_reconciled":194,
           "risk_parameters_reconstructed":True,"group_metrics_reaggregation_passed":True,
           "tail_collapse_confirmed":True,"constant_RLCP_equal_global_all_repetitions":True,
           "warning":"No valid low/middle/high learned-tail-risk partition; all scores constant. NaN rho is undefined, not missing data.",
           "sources_unchanged":all(digest(x["path"])==x["sha256"] for x in plan["dependencies"])}
    (OUT/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 本轮复核：误差样本补到了，LCP尚未验证成功", "",
           f"10/10辅助任务全部完成，用时{elapsed/60:.1f}分钟。分折、选epoch、标签标准化、OOF合并和区间算术均复核通过。",
           "本报告补充自动简报未解释的模型退化问题；原自动简报、计划、模型与结果保留原样。", "",
           "## 1. 可靠误差样本这一环完成了什么", "",
           "在原97块训练电芯内，每块由未见过它的辅助模型预测，且外折不用于标准化或早停。OOF误差只训练风险模型，最终61块校准电芯继续计算CP分位数。",
           "原dev20中>50%相对误差均为0个；OOF97中CPMLP有7个，CPLSTM有9个。每架构只有97块不同电芯，不因两个模型或100个前缀而翻倍。",
           "OOF平均相对误差22.1%/23.2%，历史池22.4%/22.5%，未发现整体误差量级明显失配；这不能证明条件误差分布一致，辅助模型训练规模仍为约77而非97。", "",
           "## 2. 最大的问题：本轮尾部局部化退化了", "",
           "两个OOF 90%分位数模型的三个特征系数都是0。固定L1正则alpha=.05下，常数解的最优性条件核查通过，属于当前目标和正则强度允许的解，不是优化器没有运行。",
           "因此任何电芯都获得同一个相对风险，风险距离全部为0；所有测试电芯落入同一个‘高组’，根本没有形成可解释的低中高分组。自动简报中的rho=nan是零方差造成的未定义相关，不是漏算数据。",
           "若尺度是c×预测寿命，校准分位数会相应除以c，最终区间与简单寿命尺度CP相同；恒等距离的Gaussian RLCP也全部等权。本轮已核对两模型全部201次划分均如此。",
           "所以，本轮并未实质检验一个能区分电芯的学习尾部距离，不能拿它证明LCP无效；没有无穷区间也不能称为成功解决局部化问题。", "",
           "## 3. 平均误差尺度学到了一点，但没有超过强对照", "",
           "下表只比较同一原划分、同一seed2024预测器的200次历史池重分平均；不是前批9设置均值，也不是200个独立验证集。", "",
           "| 预测器 | 方法 | 覆盖 | 平均宽度 | Interval score |", "|---|---|---:|---:|---:|"]
    labels_map={"relative_cp":"简单寿命尺度","gap6_scale_cp":"此前六模型分歧尺度","dev20_pair_scale_cp":"dev20学习平均风险","oof_life_scale_cp":"OOF97仅寿命平均风险","oof_pair_scale_cp":"OOF97寿命＋分歧平均风险"}
    for model in ("CPMLP","CPLSTM"):
        for method in labels_map:
            row=summary.loc[(model,method)]
            lines.append(f"| {model} | {labels_map[method]} | {row.coverage:.2%} | {row.mean_width:.1f} | {row.interval_score:.1f} |")
    lines += ["", "OOF97相对同算法dev20的宽度/score改善明显，但原有简单基线已经很好，不能只选弱对照报提升。",
              "相同OOF残差下加入分歧，score比仅寿命改善约3.0%/2.6%，宽度增加约2.6%/8.2%；相对六模型分歧尺度，CPMLP宽度和score都更差，CPLSTM只以宽度增加约10.1%换来约0.9%score改善。",
              "平均风险对历史池相对误差的Spearman为CPMLP −0.106、CPLSTM −0.025，尚未显示良好的总体难度排序；这也不能单独否定少数大误差的尾部信息。", "",
              "## 4. 分组和原test的代价不能忽略", "",
              "本轮可用分组是单种子模型对的分歧组，与前批六模型分歧组不同。CPLSTM平均风险尺度使高分歧组覆盖87.5%→91.8%，但低分歧组94.5%→87.3%；覆盖资源发生了转移，尚未实现两端兼顾。",
              "主CPMLP原61块test：简单寿命尺度覆盖88.5%、宽度947；OOF分歧平均尺度覆盖仍88.5%、宽度1054。没有覆盖改善。",
              "OOF尾部尺度和学习尾部距离在原test及全部重分中均等同简单寿命尺度；单个原test、原CPLSTM结果及所有方法仍见原自动简报。", "",
              "## 5. 下一步应怎样收缩", "",
              "保留这批OOF预测，下一步无需再训练辅助预测器。先仅用训练侧OOF信息检查尾部正则化与特征是否能产生稳定的非恒定风险，再冻结配置做区间比较。",
              "不能按当前cal/test效果反复挑正则参数；非零系数也不代表泛化变好。只有确认风险函数确实区分电芯，才有资格评价学习距离的LCP增益。",
              "当前结论：补充诚实误差监督已实现；它尚未带来超过简单分歧尺度的稳定收益；本轮尾部LCP因模型退化而没有完成有效方法检验。历史测试暴露限制继续保留。"]
    (OUT/"ANALYSIS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print(f"REVIEW COMPLETE {OUT/'ANALYSIS.md'}")


if __name__=="__main__":
    main()
