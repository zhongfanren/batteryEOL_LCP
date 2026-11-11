"""Read-only exports/plots of all original test cells; no interval recomputation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "interval_views_20261028"
ISU = HERE / "batterylife_main_table/isu_crossfit_risk_20261028/original_intervals.csv"
MATR = HERE / "learnable_localizers/runs_transfer/20260926_213949_940282/seed_2024/evaluation.csv"
BENCH = HERE / "batterylife_matr_benchmark/frozen_split_cp/20261021_193212_072306/test_intervals.csv"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_wide(rows, methods, path):
    first = rows[rows.method == methods[0][0]].set_index("cell_id")
    out = first[["eol", "prediction"]].rename(columns={"eol": "真实寿命_周期", "prediction": "点预测寿命_周期"})
    for name, label in methods:
        values = rows[rows.method == name].set_index("cell_id").loc[out.index]
        np.testing.assert_allclose(values.eol, first.eol)
        np.testing.assert_allclose(values.prediction, first.prediction)
        for column, chinese in (("lower", "下界"), ("upper", "上界")):
            out[f"{label}_{chinese}"] = values[column]
        out[f"{label}_覆盖真实寿命"] = np.where(values.covered, "是", "否")
    out = out.sort_values("真实寿命_周期").rename_axis("电芯ID")
    out.to_csv(path, encoding="utf-8-sig")
    return out


def plot(rows, methods, title, path):
    first = rows[rows.method == methods[0][0]].sort_values(["eol", "cell_id"])
    names = first.cell_id.tolist()
    height = max(8., len(names)*.24)
    fig, axes = plt.subplots(1, len(methods), figsize=(15.5, height), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(names))
    for ax, (method, label) in zip(axes, methods):
        d = rows[rows.method == method].set_index("cell_id").loc[names]
        colors = np.where(d.covered, "#6688ae", "#d3483c")
        ax.hlines(y, d.lower, d.upper, colors=colors, linewidth=2., alpha=.8)
        ax.scatter(d.prediction, y, color="#205084", s=10, zorder=3, label="Point prediction")
        ax.scatter(d.eol, y, color="black", marker="x", s=22, zorder=4, label="True EOL")
        ax.set_title(f"{label}\nCovered {int(d.covered.sum())}/{len(d)}", fontsize=10)
        ax.set_xlabel("Lifetime (cycles)")
        ax.grid(axis="x", alpha=.18)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(title+"\nAll original test cells; ordered by true EOL for display only. Red interval = missed EOL.", fontsize=11)
    fig.tight_layout(rect=(0, .025, 1, .965))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    before = {str(path): digest(path) for path in (ISU, MATR, BENCH)}
    isu = pd.read_csv(ISU)
    isu = isu[isu.model == "CPMLP"].rename(columns={"cell_file": "cell_id"})
    assert isu.cell_id.nunique() == 61
    for _, d in isu.groupby("method"):
        assert len(d) == d.cell_id.nunique() == 61
        np.testing.assert_array_equal(d.covered, (d.lower <= d.eol) & (d.eol <= d.upper))
    isu_methods = [("split_cp", "普通CP"), ("relative_cp", "寿命尺度CP"),
                   ("gap6_scale_cp", "六模型分歧尺度CP"), ("oof_pair_scale_cp", "OOF平均风险尺度CP")]
    save_wide(isu, isu_methods, OUT / "ISU_CPMLP_test61.csv")
    old = pd.read_csv(MATR)
    assert len(old) == old.cell_id.nunique() == 33 and old.seed.eq(2024).all()
    legacy_methods = [("split", "普通CP"), ("fixed_rbf", "固定RBF局部化"), ("hopfield_random", "Hopfield随机初始化"),
                      ("hopfield_ssl_matr", "Hopfield_MATR自监督"), ("hopfield_ssl_matr_hust", "Hopfield_MATR_HUST自监督")]
    records = []
    for method, _ in legacy_methods:
        for row in old.itertuples():
            lower, upper = getattr(row, f"{method}_lower"), getattr(row, f"{method}_upper")
            records.append({"cell_id": row.cell_id, "method": method, "eol": row.eol, "prediction": row.prediction,
                            "lower": lower, "upper": upper, "covered": lower <= row.eol <= upper})
    legacy = pd.DataFrame(records)
    save_wide(legacy, legacy_methods, OUT / "MATR_CPTransformer_seed2024_test33.csv")
    benchmark = pd.read_csv(BENCH)
    assert len(benchmark) == 363 and benchmark.cell_id.nunique() == 33
    assert set(benchmark.cell_id) == set(old.cell_id)
    np.testing.assert_array_equal(benchmark.covered, (benchmark.lower <= benchmark.eol) & (benchmark.eol <= benchmark.upper))
    benchmark.rename(columns={"model": "预测器", "cell_id": "电芯ID", "eol": "真实寿命_周期", "prediction": "点预测寿命_周期",
                              "lower": "普通CP下界", "upper": "普通CP上界", "covered": "覆盖真实寿命", "interval_score": "区间得分"}).to_csv(
                                  OUT / "MATR_11models_splitCP_test33.csv", index=False, encoding="utf-8-sig")
    plot(isu, [("relative_cp", "Lifetime scale CP"), ("gap6_scale_cp", "Lifetime + six-model gap scale CP"),
               ("oof_pair_scale_cp", "OOF mean-risk scale CP")],
         "ISU | frozen CPMLP seed2024 | original 61 test cells | T=100", OUT / "ISU_test61.png")
    plot(legacy, [("split", "Split CP"), ("fixed_rbf", "Fixed RBF: legacy basic LCP"),
                  ("hopfield_ssl_matr_hust", "SSL MATR+HUST: legacy basic LCP")],
         "MATR | historical CPTransformer seed2024 | original 33 test cells | T=100", OUT / "MATR_test33.png")
    assert all(digest(Path(path)) == value for path, value in before.items())
    (OUT / "source_manifest.json").write_text(json.dumps({"sources_unchanged": True, "source_sha256": before,
                 "ISU_test_cells": 61, "MATR_test_cells": 33, "intervals_recomputed": False,
                 "selection": "All original test cells, preset models/seeds; no test-performance selection"}, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text("""# 固定第100周期：逐测试电芯区间查看

只读取和展示既有结果，不训练、不校准、不重算上下界，也不修改原文件。所有原测试电芯均展示；按真实寿命排序只为阅读。

- `ISU_CPMLP_test61.csv`：当前主设置CPMLP seed2024，原61块test，每块一行；含真实寿命、点预测及4种方法的上下界/覆盖。
- `ISU_test61.png`：同样61块，寿命尺度/六模型分歧尺度/OOF平均风险尺度三列区间。最新尾部模型退化成寿命尺度，故没有单列重复曲线；原文件保留全部18方法。
- `MATR_CPTransformer_seed2024_test33.csv`：早期MATR CPTransformer＋LCP试验，固定seed2024，原33块test，每块一行；普通CP、固定RBF和3种Hopfield方法。
- `MATR_test33.png`：上述历史试验的普通CP/固定RBF/自监督Hopfield三列，所有33块均保留。旧LCP是basic weighted诊断，不等于当前RLCP保证。
- `MATR_11models_splitCP_test33.csv`：后来的11模型普通Split CP基准，长表363行，每个模型33块；与前述历史CPTransformer不是同一个预测器检查点，切勿混比。

图中：横轴为寿命周期，纵轴为电芯ID；线段表示区间，蓝点表示点预测，黑叉表示真实EOL。红线表示未覆盖真实EOL。CSV带UTF-8 BOM，适合Excel打开；列名为中文。

这些都是总寿命EOL区间，不是随运行更新的轨迹；当前统一观察到第100周期。旧MATR负下界保留原样，未为画图裁剪。ISU的真实值微小小数来自原保存结果的逆标准化误差，未篡改成整数。

查看不改变训练结果；若据此选模型、距离、参数或筛选电芯，则仍属于使用历史test进行方法开发。三个来源本身都已有test暴露，不能称全新确认性检验。

原始来源和哈希见 `source_manifest.json`。横向任务定义见项目CONTEXT_HANDOFF最新说明。
""", encoding="utf-8")
    print(f"EXPORTED {OUT}; source hashes unchanged; all61/all33 retained")


if __name__ == "__main__":
    main()
