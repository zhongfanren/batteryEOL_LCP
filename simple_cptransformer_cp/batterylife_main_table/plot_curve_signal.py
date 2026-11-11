"""All primary OOF query CDFs for the feature-screening experiment (NOT CP)."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import check_curve_signal as run

COLORS = {"kernel_global": "#777777", "kernel_life": "#336ca6", "family": "#24875b"}
LABELS = {"hand": "9 input summaries -> PCA3", "embedding": "Frozen fold embedding -> PCA3", "curve": "57 time features, six balanced blocks"}


def draw(bank, j):
    scores = bank["reference_scores"]
    order = np.argsort(scores, kind="stable")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    details = []
    for col, family in enumerate(run.FAMILIES):
        for method in ("kernel_global", "kernel_life", f"kernel_{family}"):
            weight = bank[method][j]
            cumulative = np.cumsum(weight[order])
            k = min(int(np.searchsorted(cumulative, .9, side="left")), len(scores)-1)
            q = scores[order[k]]
            color = COLORS.get(method, COLORS["family"])
            label = method.replace("kernel_", "")
            for ax in axes[:, col]:
                ax.step(np.r_[0., scores[order]], np.r_[0., cumulative], where="post", color=color, label=f"{label}: q90={q:.3f}")
                ax.scatter(q, cumulative[k], color=color, s=18, zorder=4)
                ax.axvline(q, color=color, linestyle=":", alpha=.5)
            if method == f"kernel_{family}":
                details.append(f"{family}: selected {bank['reference_names'][order[k]]}, "
                               f"CDF={cumulative[k-1] if k else 0:.3f}->{cumulative[k]:.3f}, "
                               f"ESS={1/np.sum(weight*weight):.1f}")
        for row in range(2):
            ax = axes[row, col]
            ax.axhline(.9, color="#b74646", linestyle="--", linewidth=1)
            ax.set_xlabel("Reference normalized absolute error (dimensionless)")
            ax.set_ylabel("Cumulative reference weight")
            ax.set_xlim(0, max(scores.max()*1.06, .01))
            ax.grid(alpha=.15)
        axes[0, col].set_ylim(0, 1.02)
        axes[1, col].set_ylim(.78, 1.005)
        axes[0, col].set_title(LABELS[family], fontsize=10)
        axes[0, col].legend(fontsize=8, loc="lower right")
        axes[1, col].set_title("Tail zoom")
    fig.suptitle(f"{bank['query_names'][j]} | CPMLP OOF query | actual normalized error={bank['query_scores'][j]:.3f}\n"
                 "Training-side distribution diagnostic, NOT a calibrated CP interval; reference OOF fits overlap", fontsize=12)
    fig.text(.03, .015, "\n".join(details), fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .93))
    return fig


def weight_example(bank, j, out):
    order = np.argsort(bank["reference_scores"], kind="stable")
    fig, ax = plt.subplots(figsize=(14, 4))
    for offset, method in enumerate(("kernel_global", "kernel_life", "kernel_hand")):
        ax.bar(np.arange(len(order))+(offset-1)*.25, bank[method][j][order], width=.25,
               color=COLORS.get(method, COLORS["family"]), label=method)
    ax.set_xlabel("Reference cells sorted by normalized error; IDs in weighted_CDF_points.csv")
    ax.set_ylabel("Individual reference weight")
    ax.set_title(f"{bank['query_names'][j]} | fixed first query, hand-neighbourhood weights")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "example_weights.png", dpi=125); plt.close(fig)


def summary_plot(out, plotdir):
    table = pd.read_csv(out / "summary.csv")
    methods = ["qr_hand", "qr_embedding", "qr_curve", "kernel_hand", "kernel_embedding", "kernel_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, ("CPMLP", "CPLSTM")):
        for offset, phase, color in ((-.18, "oof_cv", "#336ca6"), (.18, "dev20", "#24875b")):
            part = table[(table.model == model) & (table.phase == phase)].set_index("method").loc[methods]
            ax.barh(np.arange(len(methods))+offset, part.gain_pct_vs_life, height=.34, color=color, label=phase)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(model); ax.set_xlabel("Pinball gain (%) vs matched life-only route")
        ax.set_yticks(np.arange(len(methods)), methods); ax.legend(); ax.grid(axis="x", alpha=.2)
    fig.suptitle("Feature screening only: gains vs life-only do NOT imply gains vs global; see REVIEW.md")
    fig.tight_layout()
    fig.savefig(plotdir / "screening_overview.png", dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=run.Path, default=run.OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert run.cp.read(out / "state.json")["state"] == "complete"
    plotdir = out / "visuals"
    plotdir.mkdir(exist_ok=True)
    points, attribution, hashes = [], [], []
    pdf_path = plotdir / "OOF_CDF_all97.pdf"
    pages = 0
    with PdfPages(pdf_path) as pdf:
        for fold in range(5):
            path = out / "neighbourhoods" / f"fold_{fold}_CPMLP_oof_cv.npz"
            bank = dict(np.load(path, allow_pickle=False))
            hashes.append({"path": str(path), "sha256": run.cp.digest(path)})
            order = np.argsort(bank["reference_scores"], kind="stable")
            scores = bank["reference_scores"][order]
            for j, name in enumerate(bank["query_names"]):
                for method in [m for m in run.METHODS if m.startswith("kernel_")]:
                    weight = bank[method][j]
                    cumulative = np.cumsum(weight[order])
                    k = min(int(np.searchsorted(cumulative, .9, side="left")), len(scores)-1)
                    attribution.append({"cell_file": name, "fold": fold, "method": method, "actual_relative_error": bank["query_scores"][j],
                        "q90": scores[k], "selected_cell": bank["reference_names"][order[k]],
                        "cdf_before": cumulative[k-1] if k else 0., "cdf_after": cumulative[k],
                        "ess": 1/np.sum(weight*weight), "top_two_score_weight": weight[order[-2:]].sum()})
                    for rank, index in enumerate(order):
                        points.append({"cell_file": name, "fold": fold, "method": method, "rank": rank+1,
                            "reference_cell": bank["reference_names"][index], "reference_score": bank["reference_scores"][index],
                            "weight": weight[index], "cumulative_weight": cumulative[rank], "selected": rank == k})
                figure = draw(bank, j)
                pdf.savefig(figure)
                if pages == 0:
                    figure.savefig(plotdir / "example_CDF.png", dpi=125)
                    weight_example(bank, j, plotdir)
                plt.close(figure)
                pages += 1
            print(f"PLOT fold{fold} cumulative pages={pages}", flush=True)
    assert pages == len({row["cell_file"] for row in attribution}) == 97
    pd.DataFrame(points).to_csv(plotdir / "weighted_CDF_points.csv", index=False)
    pd.DataFrame(attribution).to_csv(plotdir / "quantile_attribution.csv", index=False)
    summary_plot(out, plotdir)
    run.cp.write(plotdir / "audit.json", {"pages": pages, "cdf_points": len(points), "attribution_rows": len(attribution),
        "source_hashes": hashes, "plot_script_sha256": run.cp.digest(__file__), "all_queries_retained": True,
        "interpretation": "Empirical tail screening, no CP finite-sample correction or infinity atom"})
    (plotdir / "README.md").write_text("""# 这次CDF图与之前有什么区别

这是训练侧的信息检查，不是最终测试区间。只展示原train97的OOF查询，原cal/test没有进入本批分析。

每页对应一块查询电芯。三列分别是手工特征、冻结内部表示、57维曲线邻域；上排完整CDF，下排尾部放大。
灰=参考误差全局分布，蓝=仅寿命邻域，绿=寿命加该特征的邻域。红横线=经验90%分位水平。
每个查询的参考电芯为其他辅助折的77/78块；本页所有方法使用同样的参考误差，只改变权重。

横轴是绝对误差/预测寿命尺度，没有周期单位。标题给出的实际误差只供事后诊断，不用于权重。
本图没有CP有限样本修正或测试∞原子，因此不同于前一轮RLCP图，不能把这个经验q直接当成有保证的区间半径。
参考OOF辅助模型训练重叠，所以这些图也不构成端到端独立验证。

先读../REVIEW.md，尤其检查是否超过灰色全局对照。只超过较差的蓝色寿命邻域不代表发现稳定新增信息。
weighted_CDF_points.csv可定位每个阶梯的参考电芯；quantile_attribution.csv有过线前后、选中点与ESS。
example_CDF.png和example_weights.png固定选择第一个查询，未按表现挑选。
""", encoding="utf-8")
    run.cp.write(out / "visualization_complete.json", {"state": "complete", "pages": pages, "updated_at": run.cp.stamp()})
    state = run.cp.read(out / "state.json")
    state.update(plots="complete", updated_at=run.cp.stamp())
    run.cp.write(out / "state.json", state)
    print(f"COMPLETE {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
