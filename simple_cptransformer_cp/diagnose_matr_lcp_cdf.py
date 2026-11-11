"""Reconstruct legacy fixed-RBF intermediates and plot all 33 weighted CDFs.

The old run saved query weighted means of log(1+calibration error), but not
calibration predictions or weights. Recover the 34 log errors from 33 query
means plus their uniform mean, with the independently reconstructed weights.
Validate against ALL saved quantiles/intervals; never substitute another model.
No GPU training, original-file edits, or bandwidth selection is performed.
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "learnable_localizers"))
import run_comparison as original

SOURCE = HERE / "learnable_localizers/runs_transfer/20260926_213949_940282"
OUT = HERE / "interval_views_20261028/MATR_fixed_RBF_CDF"
SEEDS = (2024, 42, 2021)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def quantile_index(errors, weights, level):
    order = np.argsort(errors, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(int(np.searchsorted(cumulative, level, side="left")), len(errors)-1)
    return order, cumulative, position


def draw_page(cell, errors, weights, cal_names, cal_lives, level, h, plot_path):
    order, cumulative, position = quantile_index(errors, weights, level)
    uniform = np.arange(1, len(errors)+1)/len(errors)
    q = errors[order[position]]
    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=(1., 1.12))
    full, zoom, bars = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    for ax in (full, zoom):
        ax.step(np.r_[0., errors[order]], np.r_[0., uniform], where="post", color="#999999", label="Uniform calibration CDF")
        ax.step(np.r_[0., errors[order]], np.r_[0., cumulative], where="post", color="#315d91", linewidth=1.7, label="Query-weighted calibration CDF")
        ax.axhline(level, color="#ae4a42", linestyle="--", linewidth=1., label="Legacy level 32/34 = 94.12%")
        ax.axhline(.9, color="#b3b3b3", linestyle=":", linewidth=.9, label="90% reference (not a replacement)")
        ax.axvline(q, color="#ae4a42", linestyle=":", linewidth=1.2)
        ax.scatter([q], [cumulative[position]], color="#ae4a42", s=32, zorder=5)
        ax.axvspan(errors[order[-3]], errors[order[-2]], color="#debe88", alpha=.13)
        ax.set_xlabel("Calibration absolute error (cycles)")
        ax.set_ylabel("Cumulative weight")
        ax.grid(alpha=.15)
        ax.set_xlim(0, errors.max()*1.08)
    full.set_ylim(0, 1.02); zoom.set_ylim(.82, 1.005)
    full.set_title("Full weighted empirical distribution")
    zoom.set_title("Tail zoom: location of quantile crossing")
    full.legend(fontsize=7, loc="lower right")
    selected = order[position]
    colors = ["#ca8c3e" if i in order[-2:] else "#7594b8" for i in order]
    colors[position] = "#b44943"
    bars.bar(np.arange(len(order)), weights[order], color=colors)
    bars.axhline(1/len(order), color="grey", linestyle="--", linewidth=.8, label="Uniform weight 1/34")
    bars.set_xticks(np.arange(len(order)), [cal_names[i] for i in order], rotation=65, ha="right", fontsize=6)
    bars.set_ylabel("Individual CDF jump = weight")
    bars.set_xlabel("Calibration cells ordered by ABSOLUTE ERROR, not lifetime")
    bars.set_title("Orange: two largest errors; red: calibration point selected by the legacy quantile")
    bars.legend(fontsize=8)
    before = cumulative[position-1] if position else 0.
    tail_mass = weights[order[-2:]].sum()
    err = abs(cell.eol-cell.prediction)
    fig.suptitle(f"{cell.cell_id} | true EOL={cell.eol:.0f}, prediction={cell.prediction:.1f}, actual error={err:.1f}\n"
                 f"Fixed RBF h={h:.4f} | radius={q:.2f} | top-two-error weight={tail_mass:.2%} | ESS={1/np.sum(weights**2):.1f}", fontsize=12)
    footer = (f"Selected {cal_names[selected]} (true EOL {cal_lives[selected]:.0f}, calibration error {q:.2f}); "
              f"CDF before/after: {before:.5f} / {cumulative[position]:.5f}.\n"
              "A horizontal error gap differs from a large vertical weight jump. All intermediates recovered and checked against the original run.")
    fig.text(.03, .015, footer, fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(plot_path, dpi=135)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cells").mkdir(exist_ok=True)
    plan = {"purpose": "post-hoc diagnostic of archived fixed-RBF CDFs, not new method selection",
            "seeds": list(SEEDS), "plots": "all original 33 test cells, seed2024; other seeds summarized",
            "bandwidth_sensitivity": [.5, 1., 2.], "selection_by_test": False,
            "reconstruction": "linear inversion of 33 saved weighted log-error means plus uniform log-error mean; require full rank and reproduce saved quantiles",
            "scope": "fixed RBF only; SSL localizer states/weights were not archived and are not invented"}
    (OUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    split = read(SOURCE / "split.json")
    names = split["matr_development"] + split["matr_calibration"] + split["matr_evaluation"]
    assert len(names) == len(set(names)) == 169
    cache = HERE / "error_predictability/cache/8691ea902b54f8fe"
    sources = [SOURCE / "split.json", SOURCE / "config.json", Path(original.__file__),
               HERE / "error_predictability/run.py", HERE / "run_t100.py"]
    x = []
    for name in names:
        files = list(cache.glob(Path(name).stem+"_*.npz"))
        assert len(files) == 1
        sources.append(files[0])
        with np.load(files[0]) as values:
            x.append(values["x"])
    for seed in SEEDS:
        sources += [SOURCE / f"seed_{seed}/evaluation.csv", SOURCE / f"seed_{seed}/metrics.json"]
    labels_path = Path(read(SOURCE / "config.json")["data_root"]) / "Life labels/MATR_labels.json"
    sources.append(labels_path)
    before = {str(p): digest(p) for p in sources}
    features = original.time_features(np.stack(x))
    median, scale = original.fit_robust_scaler(features[:102])
    z, h = original.fixed_rbf_fit(original.apply_scaler(features, median, scale), np.arange(102))
    distance2 = np.sum((z[136:, None]-z[None, 102:136])**2, axis=2)
    weights, temperature = original.weights_with_min_ess(-distance2/(2*h*h))
    system = np.r_[weights, np.ones((1, 34))/34]
    assert np.linalg.matrix_rank(system) == 34
    labels = read(labels_path)
    cal_names = np.array([Path(n).stem for n in split["matr_calibration"]])
    cal_lives = np.array([labels[n] for n in split["matr_calibration"]])
    level = 32/34
    audit, summaries, all_cdf, sensitivity, recovered_rows = [], [], [], [], []
    for seed in SEEDS:
        table = pd.read_csv(SOURCE / f"seed_{seed}/evaluation.csv")
        assert table.cell_id.tolist() == [Path(n).stem for n in split["matr_evaluation"]]
        saved = read(SOURCE / f"seed_{seed}/metrics.json")
        np.testing.assert_allclose(h, saved["fixed_rbf_bandwidth"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(np.mean(1/np.sum(weights**2, axis=1)), saved["methods"]["fixed_rbf"]["ess_mean"], atol=1e-10, rtol=0)
        rhs = np.r_[table.fixed_rbf_risk, table.split_risk.iloc[0]]
        log_errors = np.linalg.solve(system, rhs)
        errors = np.expm1(log_errors)
        assert (errors >= 0).all() and np.isfinite(errors).all()
        predicted_q = original.weighted_quantiles(errors, weights, level)
        delta = np.max(np.abs(predicted_q-table.fixed_rbf_q))
        assert delta < 1e-6
        np.testing.assert_allclose(np.exp(np.mean(log_errors))-1, np.expm1(table.split_risk.iloc[0]), rtol=1e-10)
        np.testing.assert_allclose(predicted_q, (table.fixed_rbf_upper-table.fixed_rbf_lower)/2, rtol=1e-9, atol=1e-6)
        order = np.argsort(errors)
        np.testing.assert_allclose(errors[order[31]], table.split_q, rtol=1e-9, atol=1e-6)
        for i in range(34):
            recovered_rows.append({"seed": seed, "calibration_cell": cal_names[i], "true_eol": cal_lives[i],
                                   "recovered_absolute_error": errors[i], "recovered_log1p_error": log_errors[i]})
        for index, cell in enumerate(table.itertuples()):
            row_weights = weights[index]
            order, cumulative, position = quantile_index(errors, row_weights, level)
            selected = order[position]
            largest = np.argmax(row_weights)
            previous_cdf = cumulative[position-1] if position else 0.
            summary = {"seed": seed, "test_cell": cell.cell_id, "true_eol": cell.eol, "prediction": cell.prediction,
                       "actual_test_error": abs(cell.eol-cell.prediction), "saved_q": cell.fixed_rbf_q,
                       "selected_calibration_cell": cal_names[selected], "selected_error": errors[selected],
                       "cdf_before_q": previous_cdf, "cdf_at_q": cumulative[position], "selected_jump_weight": row_weights[selected],
                       "largest_jump_cell": cal_names[largest], "largest_jump_error": errors[largest], "largest_jump_weight": row_weights[largest],
                       "preceding_error_gap": errors[selected]-errors[order[position-1]] if position else errors[selected],
                       "top_two_error_weight": row_weights[order[-2:]].sum(), "tail_budget": 1-level,
                       "ess": 1/np.sum(row_weights**2), "temperature": temperature[index]}
            summaries.append(summary)
            for rank, i in enumerate(order):
                all_cdf.append({"seed": seed, "test_cell": cell.cell_id, "calibration_cell": cal_names[i],
                                "calibration_true_eol": cal_lives[i], "calibration_error": errors[i], "weight": row_weights[i],
                                "cumulative_weight": cumulative[rank], "error_rank": rank+1, "selected_by_quantile": i==selected})
        for factor in (.5, 1., 2.):
            alt, temps = original.weights_with_min_ess(-distance2/(2*(h*factor)**2))
            qs = original.weighted_quantiles(errors, alt, level)
            for index, cell in enumerate(table.itertuples()):
                sensitivity.append({"seed": seed, "test_cell": cell.cell_id, "bandwidth_factor": factor,
                                    "q": qs[index], "top_two_error_weight": alt[index, order[-2:]].sum(),
                                    "ess": 1/np.sum(alt[index]**2), "temperature": temps[index]})
        audit.append({"seed": seed, "weighted_log_means_reconstruction_max_error": float(np.max(abs(system@log_errors-rhs))),
                      "all_33_saved_quantiles_max_error": float(delta), "calibration_errors_nonnegative": True})
        if seed == 2024:
            np.savez_compressed(OUT / "reconstructed_fixed_rbf_seed2024.npz", calibration_names=cal_names,
                                test_names=table.cell_id.to_numpy(dtype=str), calibration_errors=errors, weights=weights,
                                distances_squared=distance2, temperature=temperature, bandwidth=h)
            with PdfPages(OUT / "ALL_33_weighted_CDFs.pdf") as pdf:
                for index, cell in enumerate(table.itertuples()):
                    fig = draw_page(cell, errors, weights[index], cal_names, cal_lives, level, h,
                                    OUT / "cells" / f"{cell.cell_id}.png")
                    pdf.savefig(fig)
                    plt.close(fig)
            fig, axes = plt.subplots(9, 4, figsize=(16, 22), sharex=True, sharey=True)
            for ax, index in zip(axes.ravel(), np.argsort(table.eol.to_numpy())):
                cell = table.iloc[index]
                order, cumulative, _ = quantile_index(errors, weights[index], level)
                ax.step(np.r_[0, errors[order]], np.r_[0, np.arange(1,35)/34], where="post", color="#aaaaaa", linewidth=.8)
                ax.step(np.r_[0, errors[order]], np.r_[0, cumulative], where="post", color="#315d91", linewidth=1.)
                ax.axhline(level, color="#b44943", linestyle="--", linewidth=.7)
                ax.axvline(cell.fixed_rbf_q, color="#b44943", linestyle=":", linewidth=.7)
                ax.set_title(f"{cell.cell_id}\nactual error={abs(cell.eol-cell.prediction):.0f}, radius={cell.fixed_rbf_q:.0f}", fontsize=8)
                ax.set_ylim(0,1.025); ax.grid(alpha=.1)
            for ax in axes.ravel()[33:]:
                ax.axis("off")
            fig.suptitle("MATR fixed RBF, seed2024: all 33 weighted calibration CDFs\nBlue=query weights; grey=uniform; red=legacy quantile threshold and selected radius", fontsize=12)
            fig.supxlabel("Calibration absolute error (cycles)"); fig.supylabel("Cumulative weight")
            fig.tight_layout(rect=(.015,.015,1,.96)); fig.savefig(OUT / "CDF_overview.png", dpi=150); plt.close(fig)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "all_test_quantile_attribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_cdf).to_csv(OUT / "all_weighted_CDF_points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(recovered_rows).to_csv(OUT / "recovered_calibration_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(OUT / "bandwidth_sensitivity.csv", index=False, encoding="utf-8-sig")
    assert all(digest(Path(path)) == value for path,value in before.items())
    manifest = {"source_files_unchanged": True, "source_sha256": before, "reconstruction_equations": 34,
                "reconstruction_rank": int(np.linalg.matrix_rank(system)), "condition_number": float(np.linalg.cond(system)),
                "reconstruction": "unique linear inverse of archived weighted log-error means, NOT saved raw calibration forecasts",
                "quantile_validation": audit, "point_predictor_retrained": False, "SSL_CDF_available": False}
    (OUT / "audit.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    example = summary[(summary.seed==2024)&summary.test_cell.eq("MATR_b2c20")].iloc[0]
    lines = ["# MATR固定RBF：逐电芯CDF与区间归因", "", "本次只诊断旧实验，没有重新训练、替换预测器、修改区间或选择最优带宽。",
             "来源是此前作图的MATR CPTransformer seed2024（另以seed42/2021核对分位数），所有33块原test均保留。", "",
             "**重要来源说明：** 旧脚本没有保存原校准预测或局部权重。本次从缓存曲线精确重建固定RBF权重，再利用33个保存的加权log误差均值及1个均匀均值，解34元满秩线性方程，恢复34个校准绝对误差。",
             "这是中间量的数值反演，不冒充原始校准预测文件。三种子的全部99个旧分位数最大偏差低于1e-6周期，详细残差与矩阵条件数见audit.json。没有恢复误差符号，也没有为SSL组编造权重。", "",
             "**先看哪里：**", "", "- `cells/MATR_b2c20.png`：低寿命、小实际误差、宽区间的具体例子。",
             "- `ALL_33_weighted_CDFs.pdf`：每块一页，含完整CDF、尾部放大图、逐校准电芯权重。",
             "- `CDF_overview.png`：全部33块CDF总览。",
             "- `all_test_quantile_attribution.csv`：选择了哪块校准电芯、越阈值前后的CDF、最大权重点、尾部总权重等。",
             "- `all_weighted_CDF_points.csv`：每个测试电芯×34校准电芯的全部CDF阶梯数据，含校准电芯ID和真实寿命。", "",
             "**CDF读法：** 横轴是校准绝对误差，不是校准寿命；纵轴是累计权重。每个垂直跳跃的高度等于该点权重。横向大空档表示没有落在该误差范围的校准样本。两者不能混称。",
             "旧实现对34个校准点先算ceil(35×.9)=32，再使用32/34=94.12%的加权经验分位水平，故图上红横线不是90%。这是旧basic加权实现，不是当前RLCP的测试点无穷原子构造；不能简单把它改成.9就宣称保留原覆盖保证。", "",
             "**seed2024的具体机制：**", "", "- 三个最大校准误差为251.51、717.08、882.05周期，后两者分别来自MATR_b3c38（真实EOL1932）与MATR_b1c2（2237）。",
             "- 第二大误差与第三大误差之间有约465.57周期的空档。CDF在251.51处是否达到94.12%，决定半径是否跳到717.08。",
             f"- MATR_b2c20真实寿命{example.true_eol:.0f}、预测{example.prediction:.2f}、实际误差{example.actual_test_error:.2f}；两个最大误差点合计权重{example.top_two_error_weight:.2%}，超过5.88%的尾部预算，最终半径{example.saved_q:.2f}。",
             f"- 该例在251.51处的CDF为{example.cdf_before_q:.2%}，低于94.12%；直到717.08处才达到{example.cdf_at_q:.2%}。被选点的权重仅{example.selected_jump_weight:.2%}，最大竖跳反而来自误差约{example.largest_jump_error:.2f}的{example.largest_jump_cell}（权重{example.largest_jump_weight:.2%}）。",
             "- 全33块半径只有四档：146.53（18块）、251.51（5块）、717.08（8块）、882.05（2块）。",
             "- 平均ESS29.58/34，温度全部1。权重总体并非极端集中，当前结果不是ESS平滑触发造成的。部分权重小幅变化经过离散高分位数和大误差空档被放大。", "",
             "**支持与不支持：** 这支持‘少数大校准误差的尾部权重影响了部分宽区间’，但还不能单独认定bandwidth设错、距离无效或这些区间必应缩短。一次test误差小，并不等于事前条件尾部风险小；校准中的长寿命大误差是真实信息，不能按结果随意删除。",
             "已保存h×.5/1/2的全部分位数和权重敏感性，作为机制检查，不计算测试成绩来挑最优h。后续要检验可提前获得的特征是否能区分应借用这些尾部误差的电芯，并同时检查缩短后会不会损失覆盖。"]
    (OUT / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"COMPLETE {OUT}; reconstructed quantiles validated; original sources unchanged")


if __name__ == "__main__":
    main()
