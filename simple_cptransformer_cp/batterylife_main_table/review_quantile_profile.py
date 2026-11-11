"""Independent replay, score arithmetic and profile-fit isolation review."""
import argparse
import numpy as np
import pandas as pd
import run_quantile_profile as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=run.Path, default=run.DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    audit = run.base.cp.read(out / "audit.json")
    assert all(run.base.cp.digest(item["path"]) == item["sha256"] for item in audit["inputs"])
    records = pd.read_csv(out / "predictions.csv")
    assert len(records) == 2*7*(97+5*20)
    loss = .9*np.maximum(records.relative_error-records.predicted_q90, 0)+.1*np.maximum(records.predicted_q90-records.relative_error, 0)
    np.testing.assert_allclose(records.pinball, loss, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(records.below_q, records.relative_error <= records.predicted_q90)
    original = pd.read_csv(run.SOURCE / "predictions.csv").set_index(["model", "fold", "phase", "method", "cell_file"])
    mappings = {"kernel_global": "kernel_global", "kernel_life": "kernel_life", "kernel_hand": "kernel_hand", "qr_hand_q90": "qr_hand"}
    maxdiff = 0.
    for method, previous in mappings.items():
        part = records[records.method == method]
        key = pd.MultiIndex.from_arrays([part.model, part.fold, part.phase, [previous]*len(part), part.cell_file])
        ref = original.loc[key]
        for column in ("predicted_q90", "pinball"):
            np.testing.assert_allclose(part[column].to_numpy(), ref[column].to_numpy(), rtol=1e-10, atol=1e-12)
            maxdiff = max(maxdiff, float(np.max(np.abs(part[column].to_numpy()-ref[column].to_numpy()))))
    manifest = run.base.cp.read(run.base.OLD / "dataset_manifest.json")["split"]["phases"]
    assert set(records.cell_file) <= set(manifest["train"]+manifest["dev"])
    assert not set(records.cell_file) & set(manifest["calibration"]+manifest["test"])
    isolation = run.base.cp.read(out / "inner_fit_audit.json")
    for block in isolation:
        assert not set(block["fitting"]) & set(block["reference"])
        assert not set(block["fitting"]) & set(block["query"])
    checks, diagnostics = 0, []
    for path in sorted((out / "neighbourhoods").glob("*.npz")):
        bank = dict(np.load(path, allow_pickle=False))
        pieces = path.stem.split("_")
        fold, model = int(pieces[1]), pieces[2]
        phase = "oof_cv" if path.stem.endswith("oof_cv") else "dev20"
        assert not set(bank["reference_names"]) & set(bank["query_names"])
        order = np.argsort(bank["reference_scores"], kind="stable")
        scores = bank["reference_scores"][order]
        for method in run.WEIGHT_METHODS:
            weight = bank[method]
            np.testing.assert_allclose(weight.sum(axis=1), 1.)
            expected = []
            for row in weight[:, order]:
                cumulative, selected = 0., scores[-1]
                for value, mass in zip(scores, row):
                    cumulative += mass
                    if cumulative >= .9:
                        selected = value
                        break
                expected.append(selected)
            ref = records[(records.model == model) & (records.fold == fold) & (records.phase == phase) & (records.method == method)].set_index("cell_file").loc[bank["query_names"]]
            np.testing.assert_allclose(expected, ref.predicted_q90, rtol=1e-10, atol=1e-12)
            checks += len(expected)
            if method.startswith("profile_"):
                for block in set(bank["reference_fold"]):
                    keep = bank["reference_fold"] == block
                    np.testing.assert_allclose(weight[:, keep].sum(axis=1), keep.mean())
        for family in ("hand", "life"):
            p = bank[f"reference_profile_{family}"]
            q = bank[f"query_profile_{family}"]
            assert (np.diff(p, axis=-1) >= 0).all() and (np.diff(q, axis=-1) >= 0).all()
            assert np.isfinite(p).all() and np.isfinite(q).all()
        params = run.base.cp.read(out / "models" / f"{path.stem}.json")
        for inner, block in params["inner_profile_models"].items():
            for family in ("hand", "life"):
                d = block[family]
                diagnostics.append({"model": model, "outer_fold": fold, "phase": phase, "inner_fold": int(inner), "family": family,
                    "nonzero_q10": d["nonzero_coefficients"][0], "nonzero_q50": d["nonzero_coefficients"][1], "nonzero_q90": d["nonzero_coefficients"][2],
                    "query_crossings_before_sort": d["query_crossing_rows_before_sort"],
                    "fit_crossings_before_sort": d["reference_crossing_rows_before_sort"]})
    pd.DataFrame(diagnostics).to_csv(out / "profile_model_diagnostics.csv", index=False)
    # Paired cell differences retained; do not select favourable pages from them.
    gains = []
    for (model, phase), part in records.groupby(["model", "phase"]):
        pivot = part.pivot(index=["fold", "cell_file"], columns="method", values="pinball")
        for control in ("kernel_global", "kernel_hand", "profile_hand_q90"):
            for (fold, cell), gain in (pivot[control]-pivot.profile_hand_3q).items():
                gains.append({"model": model, "phase": phase, "fold": fold, "cell_file": cell, "control": control,
                              "primary_loss_reduction": gain})
    pd.DataFrame(gains).to_csv(out / "paired_cell_gains.csv", index=False)
    summary = pd.read_csv(out / "summary.csv")
    lines = ["# 多分位画像未稳定超过手工邻域", "", "这轮主要候选没有通过预定门槛，不推进正式CP比较。", "",
        "## 主要候选与强对照", "", "|模型|阶段|对全局pinball改善|对手工改善|对仅q90画像改善|", "|---|---|---:|---:|---:|"]
    for model in ("CPMLP", "CPLSTM"):
        for phase in ("oof_cv", "dev20"):
            row = summary[(summary.model == model) & (summary.phase == phase) & (summary.method == "profile_hand_3q")].iloc[0]
            lines.append(f"|{model}|{phase}|{row.gain_pct_vs_kernel_global:+.1f}%|{row.gain_pct_vs_kernel_hand:+.1f}%|{row.gain_pct_vs_profile_hand_q90:+.1f}%|")
    lines += ["", "CPMLP：训练侧仅比手工好约1.4%，未到预设2%，dev20明显恶化。CPLSTM：训练侧未超过全局，dev20虽有改善但不能抵消方向不一致。",
        "三个分位数相对只用q90的增益也不一致，不能主张多个分位数更有用。仅寿命画像未显示优势。已有手工邻域线索保留，原文DSCP-Local未被本低容量适配证伪。", "",
        "一个需保留的正面线索：CPLSTM单q90画像OOF对全局改善9.1%、对手工改善12.6%（各3/5折胜），dev20对手工改善4.3%（5/5视角胜）；但主CPMLP上不超过手工，尚未跨预测器复现。它是副对照线索，不能在看到结果后替换预定主要候选来宣布成功。",
        "手工画像的三个头多数具有非零特征系数，这次没有发生上一轮整个风险函数常数退化。输出确实形成不同画像，问题在于这些差异没有转化为稳定的尾部预测增益。", "",
        "## 这轮实际测试了什么", "",
        "只新增每头5输入的线性分位模型，采用固定alpha=.01和既有手工特征/PCA3，没有训练深encoder或重新训练点预测器。",
        "为了不让参考点画像直接拟合自己的误差，使用内层留辅助折的映射及分块CDF混合。这个适配同时包含学习几何和参考画像留出设计；不能据负结果断言所有三维画像或原文隐表示都无效。",
        "所有结果为相对误差尾部的经验分位诊断，不是最终测试区间，也不意味着CP覆盖降低或提高相同百分点。原cal/test未评价。", "",
        "## 下一步", "",
        "暂停本套线性三分位画像，不按dev表现调alpha/带宽或增加网络。优先回到已固定手工邻域的尾点权重归因，判断已有小幅信号到底来自哪些可解释曲线信息及参考样本。",
        "如果以后重开表示学习，应先明确新监督/新表示为何可能补足当前限制，不能把‘更大网络’当默认修复。", "",
        "## 核对结果", "",
        f"四种旧对照精确重放，最大数值偏差{maxdiff:.3g}；重新计算{len(records)}条损失，逐点独立核对{checks}个核分位数。",
        f"{len(isolation)}个内层拟合/参考/查询名单隔离检查通过；画像有序、权重和、块质量及输入hash检查通过。",
        "底层OOF辅助模型的训练仍交叠，dev20仍只有20块、不是五个独立数据集。当前证据仍为探索性。",
        "图：visuals/PROFILE_CDF_all97.pdf。四条主曲线为全局、原手工、单q90画像、三分位画像；全部97主OOF查询保留。"]
    (out / "REVIEW.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    run.base.cp.write(out / "review_audit.json", {"independent_quantiles": checks, "loss_rows": len(records),
        "old_baseline_max_difference": maxdiff, "inner_isolation_checks": len(isolation), "source_hashes_unchanged": True,
        "cal_test_excluded": True, "review_sha256": run.base.cp.digest(__file__)})
    state = run.base.cp.read(out / "state.json")
    state.update(review="complete", updated_at=run.base.cp.stamp())
    run.base.cp.write(out / "state.json", state)
    print(out / "REVIEW.md")


if __name__ == "__main__":
    main()
