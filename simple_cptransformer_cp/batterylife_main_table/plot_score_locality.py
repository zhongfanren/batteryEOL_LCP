"""Plot directly archived CDFs; plotting never changes experiment parameters."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import run_score_locality as run

COLORS = {"global": "#777777", "life": "#3268a8", "gap": "#21865b"}
LABELS = {"global": "Global split CP", "life": "Life RLCP (h=6)", "gap": "Gap RLCP (h=6)"}


def cdf_data(bank, original):
    points, attribution = [], []
    for si, score in enumerate(run.SCORES):
        scores = bank["cal_scores"][si]
        order = np.argsort(scores, kind="stable")
        for li, loc in enumerate(run.LOCATIONS):
            method = f"{score}__{loc}"
            for j, name in enumerate(bank["test_names"]):
                w = bank["weights"][li, j]
                cum = np.cumsum(w[:-1][order])
                k = int(np.searchsorted(cum, .9, side="left"))
                q = scores[order[k]] if k < len(scores) else np.inf
                radius = q * bank["test_scales"][si, j]
                ref = original[(original.job == run.cp.PRIMARY) & (original.method == method) & (original.cell_file == name)]
                assert len(ref) == 1
                np.testing.assert_allclose(radius, ref.radius.iloc[0], rtol=1e-10, atol=1e-8)
                attribution.append({"cell_file": name, "method": method, "score": score, "locality": loc,
                     "prediction": bank["test_prediction"][j], "eol": bank["test_eol"][j], "radius": radius,
                     "scale": bank["test_scales"][si, j], "quantile_score": q,
                     "cdf_before": cum[k-1] if k else 0., "cdf_after": cum[k] if k < len(scores) else 1.,
                     "selected_calibration_cell": bank["cal_names"][order[k]] if k < len(scores) else "+infinity test atom",
                     "selected_weight": w[order[k]] if k < len(scores) else w[-1], "self_mass": w[-1],
                     "top_two_score_mass": w[:-1][order[-2:]].sum(),
                     "ess": (1-w[-1])**2 / np.square(w[:-1]).sum(),
                     "preceding_score_gap": scores[order[k]]-scores[order[k-1]] if 0 < k < len(scores) else np.nan})
                for rank, idx in enumerate(order):
                    points.append({"cell_file": name, "method": method, "rank": rank+1,
                         "calibration_cell": bank["cal_names"][idx], "calibration_eol": bank["cal_eol"][idx],
                         "calibration_prediction": bank["cal_prediction"][idx], "calibration_score": scores[idx],
                         "weight": w[idx], "cumulative_weight": cum[rank], "selected": rank == k})
    return pd.DataFrame(points), pd.DataFrame(attribution)


def draw_page(bank, si, j):
    score = run.SCORES[si]
    name, pred, truth = bank["test_names"][j], bank["test_prediction"][j], bank["test_eol"][j]
    values = bank["cal_scores"][si]
    order = np.argsort(values, kind="stable")
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1))
    full, zoom, bars = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    detail = []
    for li, loc in enumerate(run.LOCATIONS):
        w = bank["weights"][li, j]
        cum = np.cumsum(w[:-1][order])
        k = int(np.searchsorted(cum, .9, side="left"))
        q = values[order[k]] if k < len(values) else np.inf
        radius = q * bank["test_scales"][si, j]
        for ax in (full, zoom):
            ax.step(np.r_[0, values[order]], np.r_[0, cum], where="post", color=COLORS[loc],
                    linewidth=1.4, label=f"{LABELS[loc]}: radius {radius:.1f}")
            if np.isfinite(q):
                ax.scatter([q], [cum[k]], color=COLORS[loc], s=24, zorder=5)
                ax.axvline(q, color=COLORS[loc], linestyle=":", alpha=.6)
        bars.bar(np.arange(len(values))+(li-1)*.25, w[:-1][order], width=.24, color=COLORS[loc], label=LABELS[loc], alpha=.8)
        selected = bank["cal_names"][order[k]] if k < len(values) else "+infinity test atom"
        before = cum[k-1] if k else 0.
        after = cum[k] if k < len(values) else 1.
        detail.append(f"{loc}: q={q:.4g}, scale={bank['test_scales'][si,j]:.2f}, radius={radius:.2f}; "
                      f"selected {selected}; CDF {before:.4f}->{after:.4f}; test mass={w[-1]:.3%}")
    for ax in (full, zoom):
        ax.axhline(.9, color="#bb4545", linestyle="--", label="90% threshold")
        ax.set_xlim(0, max(values.max()*1.07, .01))
        ax.set_xlabel("Calibration absolute error (cycles)" if score == "absolute" else "Calibration normalized error (dimensionless)")
        ax.set_ylabel("Cumulative weight (including test-atom normalization)")
        ax.grid(alpha=.15)
    full.set_ylim(0, 1.01); zoom.set_ylim(.80, 1.005)
    full.set_title("Full score CDF: different weights, same calibration scores")
    zoom.set_title("Tail zoom: first crossing of 90%")
    full.legend(fontsize=7, loc="lower right")
    bars.axhline(1/(len(values)+1), color="#bb4545", linestyle="--", linewidth=.8)
    bars.set_xticks(np.arange(0, len(values), 5), np.arange(1, len(values)+1, 5))
    bars.set_xlabel("Calibration score rank (ascending); exact cell IDs in weighted_CDF_points.csv")
    bars.set_ylabel("Individual calibration weight")
    bars.set_title("Weight redistribution; the test +infinity atom is recorded below, not shown as a finite bar")
    bars.legend(fontsize=8)
    fig.suptitle(f"{name} | true EOL={truth:.0f}, prediction={pred:.1f}, actual error={abs(truth-pred):.1f}\n"
                 f"Score={score}; one frozen CPMLP predictor; scale shared across the three localities", fontsize=11)
    fig.text(.03, .025, "\n".join(detail), fontsize=7)
    fig.tight_layout(rect=(0, .10, 1, .92))
    return fig


def interval_overview(original, out):
    part = original[original.job == run.cp.PRIMARY]
    order = part[part.method == "relative__global"].sort_values("eol").cell_file.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 16), sharey=True)
    for ax, loc in zip(axes, run.LOCATIONS):
        rows = part[part.method == f"relative__{loc}"].set_index("cell_file").loc[order]
        for j, row in enumerate(rows.itertuples()):
            color = COLORS[loc] if row.covered else "#bc4444"
            right = row.upper if np.isfinite(row.upper) else max(rows.prediction.max()*3, rows.eol.max()*1.2)
            ax.plot([row.lower, right], [j, j], color=color, alpha=.65)
            ax.scatter(row.prediction, j, color=color, s=12)
            ax.scatter(row.eol, j, color="black", marker="x", s=16)
            if not np.isfinite(row.upper):
                ax.annotate("inf", (right, j), fontsize=6)
        ax.set_title(f"{LABELS[loc]}\ncoverage={rows.covered.mean():.1%}, width={rows.width.mean():.1f}")
        ax.set_xlabel("Lifetime (cycles)"); ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(np.arange(len(order)), [name.replace("ISU-ILCC_", "").replace(".pkl", "") for name in order], fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("Same relative-error score: dot=prediction, black x=true EOL, red=miss\nAll test61, sorted by true EOL for display only", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / "relative_intervals_test61.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=run.DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert run.cp.read(out / "state.json")["state"] == "complete"
    source = out / "primary_intermediates.npz"
    source_hash = run.cp.digest(source)
    bank = dict(np.load(source, allow_pickle=False))
    original = pd.read_csv(out / "original_intervals.csv")
    plotdir = out / "visuals"
    plotdir.mkdir(exist_ok=True)
    points, attribution = cdf_data(bank, original)
    points.to_csv(plotdir / "weighted_CDF_points.csv", index=False)
    attribution.to_csv(plotdir / "quantile_attribution.csv", index=False)
    for si, score in enumerate(run.SCORES):
        print(f"PLOT {score}: all {len(bank['test_names'])} original test cells", flush=True)
        with PdfPages(plotdir / f"CDF_{score}_all61.pdf") as pdf:
            for j in range(len(bank["test_names"])):
                fig = draw_page(bank, si, j)
                pdf.savefig(fig)
                # Fixed selection by cell ID only; never select by error or width.
                if j == 0:
                    fig.savefig(plotdir / f"example_{score}.png", dpi=125)
                plt.close(fig)
    interval_overview(original, plotdir)
    assert run.cp.digest(source) == source_hash
    run.save(plotdir / "audit.json", {"source_sha256": source_hash, "plot_script_sha256": run.cp.digest(__file__),
             "cdf_points": len(points), "quantiles_independently_recomputed": len(attribution),
             "all_primary_test_cells": bank["test_names"].tolist(), "methods": run.METHODS,
             "test_atom_included": True, "all_infinities_retained": True})
    readme = """# 逐电芯分布诊断

先看relative_intervals_test61.png，再打开CDF_relative_all61.pdf。
三份PDF分别对应三种分数，每份按原test顺序包含全部61块电芯。

灰=普通split CP，蓝=寿命RLCP，绿=分歧RLCP。
每页三种方法使用同一校准分数、同一测试尺度，只改变权重。
横轴是分数，relative/gap6无量纲；半径=分位分数×测试尺度。
红横线为90%，有限CDF未达到1是因为剩余质量在测试+∞原子上。
若有限分数累计不足90%，分位数就是∞，不能重新归一化有限点来删除它。

与旧MATR图不同：本图保留测试+∞质量，使用RLCP的90%阈值，而非旧basic的32/34。
不要把两批曲线阈值直接混用。实际误差只在标题和区间图作事后诊断，未用于权重、尺度或参数选择。

weighted_CDF_points.csv有每个阶梯的校准ID、原始预测/标签、分数、权重、累计量；
quantile_attribution.csv有半径、过线前后、选中点、尾部质量和ESS。
"""
    (plotdir / "README.md").write_text(readme, encoding="utf-8")
    run.save(out / "visualization_complete.json", {"state": "complete", "updated_at": run.cp.stamp(), "pages": 183, "audit": str(plotdir / "audit.json")})
    print(f"COMPLETE {plotdir}", flush=True)


if __name__ == "__main__":
    main()
