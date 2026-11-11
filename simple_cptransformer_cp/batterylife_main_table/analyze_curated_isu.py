"""Audit frozen ISU results and add explicitly post-hoc scale diagnostics.

Does not train predictors or modify the original experiment outputs.  Added
ablations are exploratory analyses after test exposure, not new confirmations.
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


ROOT = Path(__file__).resolve().parent / "curated_isu_pilot"
OUT = ROOT / "analysis_20261027"
MODELS = ("CPMLP", "CPLSTM")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rho(x, y):
    return float(spearmanr(x, y).statistic)


def features(table, metadata, scales):
    pairs = np.asarray([[metadata[n][k][0]["rate_in_C"]
                         for k in ("charge_protocol", "discharge_protocol")]
                        for n in table.cell_file], dtype=float)
    return np.column_stack(((np.maximum(0, table.prediction) - scales["prediction_mean"]) / scales["prediction_std"],
                            (pairs - scales["rate_mean"]) / scales["rate_std"]))


def independent_local(cal_z, test_z, scores, randomized):
    rng = np.random.default_rng(20261027)
    radii, self_masses, effective_sizes = [], [], []
    order = np.argsort(scores)
    for z in test_z:
        center = rng.normal(z, 1.5) if randomized else z
        distances = np.r_[np.sum((cal_z - center)**2, axis=1), np.sum((z - center)**2)]
        logw = -distances / 4.5
        weights = np.exp(logw - logw.max())
        weights /= weights.sum()
        cumulative = 0.0
        q = float("inf")
        for index in order:
            cumulative += weights[index]
            if cumulative >= .9:
                q = scores[index]
                break
        radii.append(q)
        self_masses.append(weights[-1])
        effective_sizes.append((1 - weights[-1])**2 / np.sum(weights[:-1]**2))
    return np.asarray(radii), np.asarray(self_masses), np.asarray(effective_sizes)


def main():
    OUT.mkdir(exist_ok=True)
    manifest = read(ROOT / "dataset_manifest.json")
    phases = manifest["split"]["phases"]
    names = sum(phases.values(), [])
    assert len(names) == len(set(names)) == 239
    metadata = {r["cell_file"]: r["metadata"] for r in manifest["cells"]}
    labels = read(Path(manifest["data_root"]) / "Life labels" / "ISU-ILCC_labels.json")
    train_y = np.asarray([labels[n] for n in phases["train"]])
    audit, ablations, correlations, infinite_rows, risks = [], [], [], [], []
    results = {}
    for model in MODELS:
        folder = ROOT / model
        result, frozen, config = (read(folder / f) for f in ("results.json", "frozen_predictor.json", "config.json"))
        results[model] = result
        assert sha(folder / "best_model.pt") == frozen["checkpoint_sha256"]
        assert sha(ROOT / "dataset_manifest.json") == frozen["manifest_sha256"]
        assert sha(Path(__file__).parent / "run_curated_isu.py") == config["runner_sha256"]
        assert sha(Path(__file__).parent / "curated_cp_evaluation.py") == config["cp_evaluator_sha256"]
        scales = result["scales"]
        np.testing.assert_allclose([train_y.mean(), train_y.std()],
                                   [scales["prediction_mean"], scales["prediction_std"]])
        tables = {p: pd.read_csv(folder / f"{p}_predictions.csv") for p in ("dev", "calibration", "test")}
        z = {}
        for phase, table in tables.items():
            assert len(table) == len(phases[phase]) and set(table.cell_file) == set(phases[phase])
            assert table.cell_file.is_unique and (table.prefix_cycles == 100).all()
            np.testing.assert_allclose(table.eol, [labels[n] for n in table.cell_file], atol=.002, rtol=0)
            z[phase] = features(table, metadata, scales)
            e = np.abs(table.eol.to_numpy() - np.maximum(0, table.prediction.to_numpy()))
            np.testing.assert_allclose(np.mean(e / table.eol), result["point"][phase]["mape_fraction"])
        dev, cal, raw_test = (tables[p] for p in ("dev", "calibration", "test"))
        test = pd.read_csv(folder / "test_intervals.csv")
        cal_error = np.abs(cal.eol.to_numpy() - np.maximum(0, cal.prediction.to_numpy()))
        test_error = np.abs(test.eol.to_numpy() - test.prediction.to_numpy())
        split_q = np.sort(cal_error)[math.ceil((len(cal) + 1) * .9) - 1]
        np.testing.assert_allclose(test.split_cp_radius, split_q)
        for method in result["methods"]:
            lower, upper = test[f"{method}_lower"].to_numpy(), test[f"{method}_upper"].to_numpy()
            width = upper - lower
            observed = result["methods"][method]
            assert int(((test.eol >= lower) & (test.eol <= upper)).sum()) == observed["covered"]
            assert int((~np.isfinite(width)).sum()) == observed["infinite_intervals"]
            if np.isfinite(width).all():
                np.testing.assert_allclose(width.mean(), observed["mean_width_cycles"])
        for method, dims, randomize in (("base_lcp_protocol", 3, False), ("rlcp_prediction", 1, True),
                                         ("rlcp_protocol", 3, True)):
            q, mass, ess = independent_local(z["calibration"][:, :dims], z["test"][:, :dims], cal_error, randomize)
            np.testing.assert_allclose(q, test[f"{method}_radius"])
            np.testing.assert_allclose(ess, test[f"{method}_ess"])
            for index in np.flatnonzero(~np.isfinite(q)):
                row = test.iloc[index]
                assert mass[index] > .1 - 1e-12
                infinite_rows.append({"model": model, "method": method, "cell_file": row.cell_file,
                                      "eol": row.eol, "prediction": row.prediction,
                                      "effective_calibration_n": ess[index], "test_self_mass": mass[index]})

        dev_error = np.abs(dev.eol.to_numpy() - np.maximum(0, dev.prediction.to_numpy()))
        floor = max(1, .1 * np.median(dev_error))
        for name, columns in (("prediction_only", [0]), ("rates_only", [1, 2]),
                               ("prediction_plus_rates", [0, 1, 2]), ("relative_score", None)):
            if columns is None:
                cal_scale = np.maximum(1, np.maximum(0, cal.prediction.to_numpy()))
                test_scale = np.maximum(1, test.prediction.to_numpy())
            else:
                def scale(query):
                    delta = query[:, None, columns] - z["dev"][None, :, columns]
                    logw = -np.sum(delta**2, axis=2) / 4.5
                    weights = np.exp(logw - logw.max(axis=1, keepdims=True))
                    return np.maximum(floor, weights @ dev_error / weights.sum(axis=1))
                cal_scale, test_scale = scale(z["calibration"]), scale(z["test"])
            q = np.sort(cal_error / cal_scale)[55]
            radii = q * test_scale
            if name == "prediction_plus_rates":
                np.testing.assert_allclose(test_scale, test.predicted_difficulty)
                np.testing.assert_allclose(radii, test.normalized_cp_radius)
            lower, upper = np.maximum(0, test.prediction - radii), test.prediction + radii
            width = upper - lower
            score = width + 20 * (np.maximum(lower - test.eol, 0) + np.maximum(test.eol - upper, 0))
            ablations.append({"model": model, "scale": name, "n": len(test),
                              "covered": int(((lower <= test.eol) & (test.eol <= upper)).sum()),
                              "coverage": np.mean((lower <= test.eol) & (test.eol <= upper)),
                              "mean_width": np.mean(width), "interval_score": np.mean(score),
                              "scale_abs_error_spearman": rho(test_scale, test_error),
                              "post_hoc": name != "prediction_plus_rates"})
        for feature in ("prediction", "predicted_difficulty", "charge_rate_C", "discharge_rate_C"):
            correlations.append({"model": model, "feature": feature,
                                 "abs_error_spearman": rho(test[feature], test_error),
                                 "percentage_error_spearman": rho(test[feature], test_error / test.eol)})
        groups = pd.read_csv(folder / "group_diagnostics.csv")
        groups = groups[(groups.group_kind == "difficulty_group") & groups.method.isin(["split_cp", "normalized_cp"])]
        groups.insert(0, "model", model)
        risks.append(groups)
        history = read(folder / "history.json")
        assert min(history, key=lambda row: row["mape_fraction"])["epoch"] == frozen["best_epoch"]
        audit.append({"model": model, "status": "pass", "epochs": len(history),
                      "best_epoch": frozen["best_epoch"], "split_radius": float(split_q),
                      "verified": "checkpoint/code/manifest hashes; train-only label scales; cell and label membership; T=100; point metrics; all local quantiles and ESS; normalized radii; coverage and widths"})

    pd.DataFrame(ablations).to_csv(OUT / "posthoc_scale_ablations.csv", index=False)
    pd.DataFrame(correlations).to_csv(OUT / "feature_error_correlations.csv", index=False)
    pd.DataFrame(infinite_rows).to_csv(OUT / "infinite_interval_diagnostics.csv", index=False)
    risk_table = pd.concat(risks, ignore_index=True)
    risk_table.to_csv(OUT / "risk_allocation.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    colors = {"split_cp": "#607d8b", "normalized_cp": "#d2691e"}
    labels_short = {"split_cp": "Split CP", "normalized_cp": "Normalized CP"}
    for row_index, model in enumerate(MODELS):
        table = pd.read_csv(ROOT / model / "test_intervals.csv")
        ax = axes[row_index, 0]
        ax.scatter(table.predicted_difficulty, table.absolute_error, s=30, alpha=.75, color="#2a6f97")
        ax.set(xlabel="Dev-fitted difficulty (cycles)", ylabel="Test absolute error (cycles)",
               title=f"{model}: difficulty signal, rho={results[model]['difficulty_error_spearman']:.2f}")
        ax.grid(alpha=.2)
        grouped = risk_table[risk_table.model == model]
        for method in colors:
            chosen = grouped[grouped.method == method].set_index("group").loc[["low", "middle", "high"]]
            x = np.arange(3)
            axes[row_index, 1].plot(x, chosen.mean_width_cycles, "o-", color=colors[method], label=labels_short[method])
            axes[row_index, 2].plot(x, chosen.coverage * 100, "o-", color=colors[method], label=labels_short[method])
        counts = grouped[grouped.method == "split_cp"].set_index("group").loc[["low", "middle", "high"]].n_cells
        ticks = [f"{g}\n(n={n})" for g, n in zip(["Low", "Middle", "High"], counts)]
        for column in (1, 2):
            axes[row_index, column].set_xticks(np.arange(3), ticks)
            axes[row_index, column].grid(alpha=.2)
        axes[row_index, 1].set(title=f"{model}: width allocation", ylabel="Mean interval width (cycles)")
        axes[row_index, 2].set(title=f"{model}: coverage allocation", ylabel="Coverage (%)", ylim=(70, 103))
        axes[row_index, 2].axhline(90, color="black", linestyle="--", linewidth=1, label="Nominal 90%")
        axes[row_index, 1].legend()
        axes[row_index, 2].legend()
    fig.suptitle("ISU-ILCC frozen predictions: exploratory difficulty and interval allocation", fontsize=15)
    fig.savefig(OUT / "difficulty_and_allocation.png", dpi=180)
    plt.close(fig)

    lines = ["# ISU-ILCC 首轮结果核查与分析（2026-10-27）", "",
             "两模型均正常完成。检查点/代码/划分指纹、训练标签尺度、逐电芯标签、T=100、分位数、RLCP测试质量与ESS、覆盖和宽度均复算通过。无需重新训练。", "",
             "## 点预测", "", "| 模型 | test MAPE | test MAE（周期） | test RMSE（周期） | 最佳epoch |",
             "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        point = results[model]["point"]["test"]
        epoch = next(a["best_epoch"] for a in audit if a["model"] == model)
        lines.append(f"| {model} | {point['mape_fraction']:.2%} | {point['mae_cycles']:.1f} | {point['rmse_cycles']:.1f} | {epoch} |")
    lines += ["", "两模型均明显优于训练中位数常数（MAPE 59.09%）。这说明预测器学到了有效信号；跨总体的数字不能证明它们优于此前MATR或混合试验。开发集仅20块，寿命最大2036周期，测试最长4999周期；开发集对长寿命尾部的代表性不足，两模型的尾部预测仍有较大误差。", "",
              "## 预定区间方法", "", "| 模型 | 方法 | 覆盖电芯数/61 | 覆盖率 | 平均宽度（周期） | Interval score（越低越好） | 无穷区间 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for name, metric in results[model]["methods"].items():
            width = f"{metric['mean_width_cycles']:.1f}" if metric['mean_width_cycles'] is not None else "∞"
            score = f"{metric['mean_interval_score']:.1f}" if metric['mean_interval_score'] is not None else "∞"
            lines.append(f"| {model} | {name} | {metric['covered']}/61 | {metric['coverage']:.2%} | {width} | {score} | {metric['infinite_intervals']} |")
    lines += ["", "CPMLP的预定normalized CP把平均宽度减少15.7%，覆盖从59/61变为57/61，interval score下降25.7%；不能把覆盖率不同的宽度直接当成同覆盖下的优势。CPLSTM平均宽度只减少1.2%，但覆盖从55/61提高到57/61，interval score下降29.2%。", "",
              "在CPLSTM的预先定义高难度组中，实际MAE为376周期，低难度组为139周期；normalized CP将高组平均宽度从1076增加至1357，覆盖从24/30增加至27/30；低组宽度从1023降至724，覆盖从29/29变为28/29。已有值得继续验证的宽度分配信号。样本和分组均较小，不是条件覆盖证明；CPLSTM中等风险组仅2块。", "",
              "直接局部加权尚无稳定效率改进。加倍率的RLCP出现CPMLP 3个、CPLSTM 2个无穷区间；原因是个别测试点的归一化自权重大于0.1，90%分位数落到测试点的+∞质量上。它们是方法在此样本/带宽下的保守输出，不是缺失文件或数值溢出。", "",
              "## 核心辨析：误差尺度还是更好的相似性", "",
              "仅看绝对误差，预测寿命与误差的Spearman相关为CPMLP 0.457、CPLSTM 0.626；当前综合难度与误差为0.383、0.496。长寿命电芯的绝对周期误差往往更大，已经可以解释一部分异方差。普通Split CP的半径恒定，其裁剪后宽度与误差的相关不能当成自适应证据。", "",
              "下面是查看首轮结果后补充的诊断消融，保持带宽和全部原始划分不变；没有训练点预测器，也未搜索最优参数。这些数字用于形成下一轮假设，不是新增的确认性结果。prediction_only/rates_only/composite用开发误差拟合尺度，relative_score直接用max(1,预测寿命)作尺度。", "",
              "| 模型 | 尺度函数 | 覆盖率 | 平均宽度 | Interval score | 尺度与绝对误差rho |",
              "|---|---|---:|---:|---:|---:|"]
    for value in ablations:
        lines.append(f"| {value['model']} | {value['scale']} | {value['coverage']:.2%} | {value['mean_width']:.1f} | {value['interval_score']:.1f} | {value['scale_abs_error_spearman']:.3f} |")
    lines += ["", "单独倍率尺度与误差相关仅-0.019/0.099，当前没有证据支持‘倍率距离已经识别了额外难度’。CPMLP仅用预测寿命拟合尺度就得到96.72%覆盖、942周期平均宽度，比综合尺度更窄。CPLSTM寿命尺度也可得到93.44%覆盖、994周期宽度，但其interval score略高于综合尺度。应把简单寿命尺度作为必需基线。", "",
              "## 下一步研究边界", "",
              "先冻结这两个预测器，围绕一个问题推进：扣除预测寿命造成的误差尺度以后，早期曲线特征是否仍能预测条件误差尾部，并帮助局部CP改善困难区域覆盖与宽度？先比较Split CP、简单寿命尺度CP、开发拟合尺度CP，再检验在归一化残差上的局部化。只有局部化在这些基线之外有重复出现的增益，才支持提出距离设计的贡献。", "",
              "下一轮应预先固定特征/带宽规则和评价方案，再做划分或种子的稳定性检验。当前61个测试电芯中每块影响覆盖率约1.64个百分点，不能凭2至3块的变化宣称稳定改进。无需立即扩展11模型、重新拼数据集或修订所有协议元数据。", "",
              "![Difficulty and interval allocation](difficulty_and_allocation.png)"]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit, "report": str(OUT / "ANALYSIS.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
