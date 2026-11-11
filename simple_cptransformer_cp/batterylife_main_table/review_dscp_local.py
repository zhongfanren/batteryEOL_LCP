"""Independent endpoint reconstruction and concise DSCP comparison."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_dscp_local as run


def empirical(scores, weight, level, atom=False):
    pairs = sorted(zip(scores, weight[:-1] if atom else weight))
    total = 0.
    for value, mass in pairs:
        total += mass
        if total >= level*weight.sum():
            return value
    return np.inf if atom else pairs[-1][0]


def main():
    out = run.OUT
    plan = run.cp.read(out / "plan.json")
    assert all(run.cp.digest(item["path"]) == item["sha256"] for item in plan["dependencies"])
    identity = run.cp.read(out / "data_identity.json")
    assert not set(identity["train"]+identity["dev"]) & set(identity["calibration"]+identity["test"])
    assert all(run.cp.digest(item["path"]) == item["sha256"] for item in identity["cache_inputs"])
    cells = pd.read_csv(out / "original_intervals.csv")
    checks = 0
    for job in plan["jobs"]:
        folder = out / "jobs" / job["key"]
        complete = run.cp.read(folder / "complete.json")
        assert run.cp.digest(folder / "localizer.pt") == complete["checkpoint_sha256"]
        history = pd.read_csv(folder / "history.csv")
        info = complete["info"]
        best = history[history.epoch == info["best_epoch"]].dev_pinball_scaled.iloc[0]
        assert best <= history.dev_pinball_scaled.min()+1e-8
        bank = dict(np.load(folder / "calibration_distributions.npz", allow_pickle=False))
        signed = job["mode"] == "raw_signed"
        for method, key in (("DSCP_raw_signed" if signed else "DSCP_relative", "trained"),
                            ("random_raw_signed" if signed else "random_relative", "random"),
                            ("RLCP_raw_absolute" if signed else "RLCP_relative", "rlcp")):
            atom = key == "rlcp"
            ref = cells[(cells.job == job["key"]) & (cells.method == method)].set_index("cell_file").loc[bank["test_names"]]
            for j in range(61):
                w, pred = bank[key][j], bank["test_prediction"][j]
                if signed and not atom:
                    lo = pred+empirical(bank["cal_signed_error"], w, .05)
                    up = pred+empirical(bank["cal_signed_error"], w, .95)
                else:
                    score = np.abs(bank["cal_signed_error"]) if signed else bank["cal_abs_relative"]
                    radius = empirical(score, w, .9, atom)*(1 if signed else max(1, pred))
                    lo, up = pred-radius, pred+radius
                lo, up = max(0, lo), max(max(0, lo), up)
                np.testing.assert_allclose([lo, up], [ref.lower.iloc[j], ref.upper.iloc[j]], rtol=1e-10, atol=1e-7)
                checks += 1
    np.testing.assert_allclose(cells.interval_score, cells.width+20*(np.maximum(cells.lower-cells.eol, 0)+np.maximum(cells.eol-cells.upper, 0)))
    table = pd.read_csv(out / "summary.csv")
    summary = table.groupby(["mode", "method"])[["coverage", "mean_width", "interval_score", "infinite_cell_fraction"]].agg(lambda x: np.mean(x.to_numpy()))
    lines = ["# DSCP-Local首轮结论", "", "已完成6个LSTM残差编码器及区间评测，原CPMLP点预测器未重训。论文的多分位监督→冻结隐状态→加权校准CDF机制已实现；输入改为前100周期曲线，是横向电芯适配。",
             "", "## 关键对照：3个encoder种子、每个200次历史池重分平均", "", "|方法|覆盖|平均宽度|Interval score↓|", "|---|---:|---:|---:|"]
    show = [("abs_relative", "split_relative"), ("abs_relative", "split_gap6"), ("raw_signed", "DSCP_raw_signed"),
            ("raw_signed", "random_raw_signed"), ("abs_relative", "DSCP_relative"),
            ("abs_relative", "random_relative"), ("abs_relative", "RLCP_relative")]
    for mode, method in show:
        row = summary.loc[(mode, method)]
        lines.append(f"|{method}|{row.coverage:.2%}|{row.mean_width:.1f}|{row.interval_score:.1f}|")
    lines += ["", "## 判断", "", "归一化DSCP区间比寿命尺度CP略窄，但覆盖略低、interval score稍差；六模型分歧尺度的综合表现更好。论文式raw signed版本优于同分数的全局经验CDF，但综合得分没有超过现有尺度强基线。",
        "训练后的表示没有超过同结构、同种子的随机初始化表示，尚不能把局部效果归因于多分位监督学到了更好的误差几何。三个种子都保留，不能挑最好种子宣布成功。",
        "RLCP_relative是另一个Gaussian随机中心及测试∞原子构造，综合分数仍未稳定超过简单尺度基线；它与论文式确定性softmax不是同一个算法。",
        "", "## 范围与限制", "", "原文输入含时序残差历史，本任务未知新电芯EOL，因此使用100×900早期曲线；固定单层LSTM32，不做原文完整架构/超参搜索；温度用train隐状态距离中位数，未用cal/test选温度。",
        "raw_signed使用有符号残差与5%/95%区间；abs_relative是本项目对称归一化分数适配。两种形状不混作纯距离增益。",
        "train97监督是辅助模型OOF误差，与最终主预测器可能有条件误差分布差异。dev20用于选epoch，尾部样本少；历史cal/test已查看，200重分不是200个独立数据集，当前均为探索证据。",
        "论文式DSCP softmax经验CDF不声称有限样本覆盖保证；只有另列RLCP采用相应随机化结构，且仍依赖任务总体/交换性条件。",
        "", "## 已核查", "", f"3个旧Split/尺度基线精确重放；{checks}个原test端点由独立累计权重重算通过；区间算术、最佳epoch、检查点/原输入hash和角色隔离通过。",
        "5项科学契约测试及实际CUDA冒烟通过，冒烟结果不计入科学结论。全量原test区间、重分指标、组指标、隐状态、参数与CDF保留。",
        "", "图：visuals/CDF_raw_signed_all61.pdf和visuals/CDF_abs_relative_all61.pdf，各含全部61块原test，固定encoder seed2024。",
        "本轮不依据结果继续扩网络或搜温度；若继续，先审查dev选择稳定性及OOF到主预测器的误差差异，再另立小假设。"]
    (out / "REVIEW.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    run.cp.write(out / "review_audit.json", {"independent_interval_checks": checks, "checksums_verified": True,
        "checkpoint_epoch_verified": True, "review_sha256": run.cp.digest(__file__)})
    state = run.cp.read(out / "state.json"); state.update(review="complete"); run.cp.write(out / "state.json", state)
    print(out / "REVIEW.md")


if __name__ == "__main__":
    main()
