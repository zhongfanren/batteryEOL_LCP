"""Directly archived signed and normalized DSCP CDFs, all original test61."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import run_dscp_local as run


def draw(bank, history, mode, j):
    signed = mode == "raw_signed"
    scores = bank["cal_signed_error"] if signed else bank["cal_abs_relative"]
    order = np.argsort(scores, kind="stable")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colours = {"global": "#888888", "random": "#bb8833", "trained": "#24875b"}
    methods = ("global", "random", "trained")
    rows = []
    for k, method in enumerate(methods):
        w = bank[method][j]
        cdf = np.cumsum(w[order])
        for ax in axes[0]:
            ax.step(np.r_[min(scores.min(), 0)-1e-6, scores[order]], np.r_[0., cdf], where="post", color=colours[method], label=method)
            for level in ((.05, .95) if signed else (.9,)):
                idx = min(int(np.searchsorted(cdf, level, side="left")), len(scores)-1)
                ax.scatter(scores[order[idx]], cdf[idx], color=colours[method], s=20)
                ax.axvline(scores[order[idx]], color=colours[method], linestyle=":", alpha=.5)
                rows.append((method, level, scores[order[idx]]))
        axes[1, 0].bar(np.arange(len(scores))+(k-1)*.25, w[order], width=.24, color=colours[method], label=method)
    for ax in axes[0]:
        for level in ((.05, .95) if signed else (.9,)):
            ax.axhline(level, color="#b74444", linestyle="--", linewidth=.8)
        ax.set_xlabel("Signed calibration residual (cycles)" if signed else "Normalized absolute calibration error")
        ax.set_ylabel("Cumulative calibration weight"); ax.grid(alpha=.15)
    axes[0, 0].set_title("Paper-style empirical CDF (no infinity atom)")
    axes[0, 0].legend(fontsize=8); axes[0, 0].set_ylim(0, 1.02)
    axes[0, 1].set_title("Upper-tail zoom"); axes[0, 1].set_ylim(.8, 1.005)
    axes[1, 0].set_xlabel("Calibration residual rank (ascending)"); axes[1, 0].set_ylabel("Individual weight")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(history.epoch, history.train_pinball_scaled, label="OOF train loss")
    axes[1, 1].plot(history.epoch, history.dev_pinball_scaled, label="Dev loss")
    axes[1, 1].set_xlabel("Encoder training epoch"); axes[1, 1].set_ylabel("Mean three-quantile pinball (scaled)")
    axes[1, 1].set_title("Epoch selected on dev only"); axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"{bank['test_names'][j]} | {mode}, encoder seed2024\n"
                 f"Frozen prediction={bank['test_prediction'][j]:.1f}, true EOL={bank['test_eol'][j]:.0f}; heads are not interval endpoints", fontsize=11)
    unique = list(dict.fromkeys(rows))
    fig.text(.03, .015, "; ".join(f"{method} q{level:g}={q:.3f}" for method, level, q in unique), fontsize=7)
    fig.tight_layout(rect=(0, .05, 1, .94))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=run.Path, default=run.OUT)
    args = parser.parse_args(); out = args.out.resolve()
    assert run.cp.read(out / "state.json")["state"] == "complete"
    plotdir = out / "visuals"; plotdir.mkdir(exist_ok=True)
    points = []
    for mode in run.MODES:
        folder = out / "jobs" / f"{mode}_s2024"
        bank = dict(np.load(folder / "calibration_distributions.npz", allow_pickle=False))
        history = pd.read_csv(folder / "history.csv")
        scores = bank["cal_signed_error"] if mode == "raw_signed" else bank["cal_abs_relative"]
        order = np.argsort(scores, kind="stable")
        with PdfPages(plotdir / f"CDF_{mode}_all61.pdf") as pdf:
            for j, name in enumerate(bank["test_names"]):
                figure = draw(bank, history, mode, j); pdf.savefig(figure)
                if j == 0:
                    figure.savefig(plotdir / f"example_{mode}.png", dpi=125)
                plt.close(figure)
                for method in ("global", "random", "trained"):
                    cumulative = np.cumsum(bank[method][j][order])
                    for rank, index in enumerate(order):
                        points.append({"mode": mode, "test_cell": name, "method": method, "cal_cell": bank["cal_names"][index],
                            "score": scores[index], "weight": bank[method][j, index], "cumulative_weight": cumulative[rank]})
        print(f"PLOT {mode}:61 pages", flush=True)
    pd.DataFrame(points).to_csv(plotdir / "weighted_CDF_points.csv", index=False)
    (plotdir / "README.md").write_text("""# DSCP-Local可视化

两份PDF各含全部61块原test，固定encoder seed2024，不挑种子或电芯。
raw_signed横轴为有符号寿命预测残差，红线为5%/95%，取两端构造非对称区间。
abs_relative横轴为绝对误差/预测寿命尺度，红线90%，分位数乘测试尺度得到对称半径。
灰=全局，黄=同种子未训练encoder，绿=多分位残差训练后的encoder；所有cal分数固定，只换权重。
这些图对应论文式softmax经验CDF，没有测试∞原子，不声明有限样本保证。
单独RLCP结果及其∞计数见RESULTS.md和CSV，不与图中确定性权重混用。
右下是训练/验证损失，验证选epoch没有使用cal/test。实际EOL仅在标题作事后检查。
""", encoding="utf-8")
    run.cp.write(out / "visualization_complete.json", {"pages": 122, "cdf_points": len(points), "updated_at": run.cp.stamp()})
    state = run.cp.read(out / "state.json"); state.update(plots="complete", updated_at=run.cp.stamp()); run.cp.write(out / "state.json", state)


if __name__ == "__main__":
    main()
