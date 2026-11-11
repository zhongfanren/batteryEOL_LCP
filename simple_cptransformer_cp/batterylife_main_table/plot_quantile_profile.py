"""Complete primary OOF CDF and quantile-profile diagnostics, not final CP."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import run_quantile_profile as run

SHOWN = ("kernel_global", "kernel_hand", "profile_hand_q90", "profile_hand_3q")
COLOR = {"kernel_global": "#888888", "kernel_hand": "#326ca6", "profile_hand_q90": "#bd8a30", "profile_hand_3q": "#28845b"}
NAME = {"kernel_global": "Global", "kernel_hand": "Fixed hand", "profile_hand_q90": "Learned q90", "profile_hand_3q": "Learned q10/q50/q90"}


def page(bank, j):
    score = bank["reference_scores"]
    order = np.argsort(score, kind="stable")
    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(2, 2)
    full, zoom = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    bars, profile_ax = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    details = []
    for m, method in enumerate(SHOWN):
        weight = bank[method][j]
        cumulative = np.cumsum(weight[order])
        k = min(int(np.searchsorted(cumulative, .9, side="left")), len(score)-1)
        q = score[order[k]]
        for ax in (full, zoom):
            ax.step(np.r_[0., score[order]], np.r_[0., cumulative], where="post", color=COLOR[method], label=f"{NAME[method]}: q={q:.3f}")
            ax.scatter([q], [cumulative[k]], color=COLOR[method], s=24, zorder=4)
            ax.axvline(q, color=COLOR[method], linestyle=":", alpha=.5)
        bars.bar(np.arange(len(score))+(m-1.5)*.2, weight[order], width=.19, color=COLOR[method], label=NAME[method])
        details.append(f"{NAME[method]}: {bank['reference_names'][order[k]]}, "
                       f"CDF {cumulative[k-1] if k else 0:.4f}->{cumulative[k]:.4f}; ESS={1/np.sum(weight*weight):.1f}")
    for ax in (full, zoom):
        ax.axhline(.9, color="#b74444", linestyle="--")
        ax.set_xlim(0, max(score.max()*1.06, .01))
        ax.set_xlabel("Reference normalized absolute error")
        ax.set_ylabel("Cumulative reference weight")
        ax.grid(alpha=.15)
    full.set_ylim(0, 1.02); zoom.set_ylim(.78, 1.005)
    full.set_title("Same scores, different neighbourhoods"); zoom.set_title("Tail zoom: first crossing of 90%")
    full.legend(fontsize=7, loc="lower right")
    bars.set_xlabel("Reference error rank (ascending)"); bars.set_ylabel("Individual weight")
    bars.set_title("Tail weights, not the largest bar alone, set q90")
    levels = np.array(run.LEVELS)
    for b, profile in enumerate(bank["query_profile_hand"][:, j]):
        profile_ax.plot(levels, profile, "o-", alpha=.6, label=f"Inner map {b+1}")
    blocks = sorted(set(bank["reference_fold"]))
    mass = np.array([np.mean(bank["reference_fold"] == block) for block in blocks])
    mean = np.einsum("b,bk->k", mass, bank["query_profile_hand"][:, j])
    profile_ax.plot(levels, mean, "ko--", linewidth=2, label="Mean (display only)")
    profile_ax.set_xticks(levels); profile_ax.set_xlabel("Requested error quantile")
    profile_ax.set_ylabel("Estimated normalized error")
    profile_ax.set_title("Query profile varies across inner fitted maps")
    profile_ax.legend(fontsize=7); profile_ax.grid(alpha=.15)
    fig.suptitle(f"{bank['query_names'][j]} | CPMLP OOF | actual relative error={bank['query_scores'][j]:.3f}\n"
                 "Train-side screen only, NOT CP intervals; reference OOF fits overlap", fontsize=12)
    fig.text(.03, .015, "\n".join(details), fontsize=7)
    fig.tight_layout(rect=(0, .09, 1, .93))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=run.Path, default=run.DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert run.base.cp.read(out / "state.json")["state"] == "complete"
    plotdir = out / "visuals"
    plotdir.mkdir(exist_ok=True)
    points, attribution, sources, pages = [], [], [], 0
    with PdfPages(plotdir / "PROFILE_CDF_all97.pdf") as pdf:
        for fold in range(5):
            path = out / "neighbourhoods" / f"fold_{fold}_CPMLP_oof_cv.npz"
            bank = dict(np.load(path, allow_pickle=False))
            sources.append({"path": str(path), "sha256": run.base.cp.digest(path)})
            order = np.argsort(bank["reference_scores"], kind="stable")
            for j, name in enumerate(bank["query_names"]):
                for method in run.WEIGHT_METHODS:
                    weight = bank[method][j]
                    cumulative = np.cumsum(weight[order])
                    k = min(int(np.searchsorted(cumulative, .9, side="left")), len(order)-1)
                    attribution.append({"cell_file": name, "fold": fold, "method": method, "actual_relative_error": bank["query_scores"][j],
                        "q90": bank["reference_scores"][order[k]], "selected_cell": bank["reference_names"][order[k]],
                        "cdf_before": cumulative[k-1] if k else 0., "cdf_after": cumulative[k],
                        "ess": 1/np.sum(weight*weight), "top_two_score_mass": weight[order[-2:]].sum()})
                    for rank, index in enumerate(order):
                        points.append({"cell_file": name, "fold": fold, "method": method, "rank": rank+1,
                            "reference_cell": bank["reference_names"][index], "reference_score": bank["reference_scores"][index],
                            "weight": weight[index], "cumulative_weight": cumulative[rank], "selected": rank == k})
                figure = page(bank, j)
                pdf.savefig(figure)
                if pages == 0:
                    figure.savefig(plotdir / "example_CDF.png", dpi=125)
                plt.close(figure)
                pages += 1
            print(f"PLOT fold{fold}, pages={pages}", flush=True)
    assert pages == len(set(row["cell_file"] for row in attribution)) == 97
    pd.DataFrame(points).to_csv(plotdir / "weighted_CDF_points.csv", index=False)
    pd.DataFrame(attribution).to_csv(plotdir / "quantile_attribution.csv", index=False)
    (plotdir / "README.md").write_text("""# 多分位画像诊断图

每页一块原train97的主CPMLP OOF查询，所有97块保留。
灰=全局经验误差，蓝=固定手工邻域，黄=只用预测q90定义距离，绿=使用三个预测分位数定义距离。
横轴是归一化绝对误差，没有周期单位；红横线是经验90%分位水平。
上排完整CDF与尾部放大；左下逐参考点权重；右下展示每个内折小模型对同一查询的三个分位数估计。

右下黑色均值只供显示。真正权重在每个内折中用匹配的同一个映射比较查询和参考，之后按参考块大小混合CDF。
参考画像模型未使用该参考点自己的误差，但底层OOF辅助预测器仍有训练交叠。这些图不是最终CP、没有测试∞原子或有限样本修正。
实际误差只用于事后诊断，不用于该查询的权重。

先读../REVIEW.md，不能只看某页区间变窄。weighted_CDF_points.csv和quantile_attribution.csv提供每个阶梯与过线点来源。
""", encoding="utf-8")
    run.base.cp.write(plotdir / "audit.json", {"pages": pages, "cdf_points": len(points), "attribution_rows": len(attribution),
        "sources": sources, "all_queries_retained": True, "plot_sha256": run.base.cp.digest(__file__)})
    run.base.cp.write(out / "visualization_complete.json", {"state": "complete", "pages": pages, "updated_at": run.base.cp.stamp()})
    state = run.base.cp.read(out / "state.json")
    state.update(plots="complete", updated_at=run.base.cp.stamp())
    run.base.cp.write(out / "state.json", state)
    print(out / "visuals" / "PROFILE_CDF_all97.pdf")


if __name__ == "__main__":
    main()
