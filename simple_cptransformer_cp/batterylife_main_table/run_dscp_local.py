"""DSCP-Local sequence-encoder pilot on fixed-T100 battery cells.

Paper mechanism: multi-quantile residual training -> frozen hidden states ->
similarity-weighted calibration CDF. Battery contexts do not contain EOL
residual histories. Paper-style softmax and valid-RLCP adaptations are named
separately; original predictor checkpoints and scripts are never edited.
"""
import argparse
import copy
import logging
import math
import os
import pickle
import random
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from torch import nn
from torch.nn import functional as F

import check_curve_signal as feature_tools
import run_isu_disagreement_cp as cp

HERE = Path(__file__).resolve().parent
OUT = HERE / "dscp_local_20261111"
OOF = HERE / "isu_crossfit_risk_20261028"
MODES = ("raw_signed", "abs_relative")
SEEDS = (2024, 42, 2021)
LEVELS = (.1, .5, .9)
PHASES = cp.PHASES


class SequenceLocalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(900, 32)
        self.encoder = nn.LSTM(32, 32, num_layers=1, batch_first=True)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in LEVELS])

    def forward(self, x):
        _, (hidden, _) = self.encoder(F.relu(self.projection(x)))
        z = hidden[-1]
        raw = torch.cat([head(z) for head in self.heads], dim=1)
        first = raw[:, :1]
        quantiles = torch.cat((first, first+torch.cumsum(F.softplus(raw[:, 1:]), dim=1)), dim=1)
        return quantiles, z


def pinball(q, target):
    levels = torch.tensor(LEVELS, device=q.device, dtype=q.dtype)
    error = target[:, None]-q
    return torch.maximum(levels*error, (levels-1)*error).mean()


def weighted_q(scores, weights, level, test_atom=False):
    order = np.argsort(scores, kind="stable")
    finite = weights[:, :-1] if test_atom else weights
    cdf = np.cumsum(finite[:, order], axis=1)
    thresholds = level*weights.sum(axis=1)
    index = np.sum(cdf < thresholds[:, None], axis=1)
    if test_atom:
        return np.r_[scores[order], np.inf][index]
    return scores[order[np.minimum(index, len(scores)-1)]]


def soft_weights(reference, query, temperature):
    logits = -np.square(cdist(query, reference))/temperature
    weight = np.exp(logits-logits.max(axis=1, keepdims=True))
    return weight/weight.sum(axis=1, keepdims=True)


def temperature(train_embedding):
    values = pdist(train_embedding, metric="sqeuclidean")
    values = values[values > 1e-12]
    return max(float(np.median(values)), 1e-6) if len(values) else 1.


def standardize(train, values):
    mean = train.mean(axis=(0, 1), keepdims=True)
    scale = np.maximum(train.std(axis=(0, 1), keepdims=True), 1e-5)
    return {p: np.clip((x-mean)/scale, -5, 5).astype(np.float32) for p, x in values.items()}, mean, scale


def residual_target(y, pred, mode):
    return y-pred if mode == "raw_signed" else np.abs(y-pred)/np.maximum(1., pred)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        plan = cp.read(out / "plan.json")
        if plan["runner_sha256"] != cp.digest(__file__):
            raise RuntimeError("Frozen runner changed; use a new output directory")
        assert all(cp.digest(row["path"]) == row["sha256"] for row in plan["dependencies"])
        return plan
    dependencies = [Path(__file__), Path(cp.__file__), Path(feature_tools.__file__),
                    OOF / "oof_predictions.csv", OOF / "train_bank.npz", OOF / "dataset_manifest.json"]
    parent = cp.read(cp.ROOT / "plan.json")
    for job in parent["jobs"]:
        if job["split_seed"] == 20261026:
            dependencies.append(cp.ROOT / "analysis" / job["key"] / "features.npz")
    point = cp.ROOT / "runs/split_20261026/seed_2024/CPMLP"
    dependencies += [point / "best_model.pt", point / "frozen_predictor.json"]
    plan = {"created_at": cp.stamp(), "runner_sha256": cp.digest(__file__),
        "dependencies": [{"path": str(p), "sha256": cp.digest(p)} for p in dependencies],
        "paper": "https://arxiv.org/abs/2610.02357; sections2.3/B.3",
        "hypotheses": ["Hsequence: residual-quantile-supervised LSTM hidden states improve local calibration over global/hand/random states", "Hscore: normalized absolute residual training improves over raw signed residual training on this battery task"],
        "predictor": "frozen original CPMLP split20261026 seed2024; no point-model refit",
        "jobs": [{"key": f"{mode}_s{seed}", "mode": mode, "seed": seed} for mode in MODES for seed in SEEDS],
        "architecture": "100 cycles x 900 values -> linear900x32 + ReLU -> 1-layer LSTM32 -> three hidden32 quantile heads; embedding is LSTM state, not quantile outputs",
        "levels": list(LEVELS), "optimizer": "AdamW lr.001 weight_decay.01, batch16, clip_grad_norm1",
        "epochs": 50, "patience": 3, "budget_seconds": 1800,
        "training": "train97 OOF residual supervision; select epoch only with original dev20 residuals of frozen main predictor",
        "input_normalization": "per-position mean/std over train97 and100 cycles only, clip[-5,5]",
        "target_normalization": "divide by one train-only residual std for numerical stability; does not change residual definition",
        "localization": "negative squared hidden-state distance / train-only median pairwise squared distance, softmax; no temperature search",
        "paper_adaptations": ["early cell curves replace time-series covariate/residual history; no unknown test-cell EOL residuals", "single predeclared LSTM configuration, not full paper architecture/hyperparameter search", "OOF97/dev20/cal61/test61 replace paper time splits", "train-only median temperature rule instead of validation search"],
        "normalized_adaptation": "abs_relative target and symmetric normalized-score CDF; separate from paper raw-signed mechanism",
        "methods": ["split_abs", "split_relative", "split_gap6", "global_signed_empirical", "global_signed_finite", "global_relative_empirical", "hand_relative_empirical", "DSCP_raw_signed", "random_raw_signed", "DSCP_raw_absolute", "RLCP_raw_absolute", "DSCP_relative", "random_relative", "RLCP_relative"],
        "rlcp_adaptation": "Gaussian randomized center, hidden state scaled by sqrt(train median squared distance), fixed h3, test+infinity retained; distinct from paper softmax",
        "readout": "all three encoder seeds, originaltest61 plus200 fixed-pool resplits, same-score global/local controls, coverage/width/intervalscore/ESS/infinity and gap groups; no test-based choice",
        "theory": "Paper softmax results exploratory, no finite-sample guarantee asserted. RLCP rows use the separate Gaussian construction under exchangeability; historical label exposure remains.",
        "random_center_seed": 20261111, "resplit_seed": 20261028, "resplits": 200,
        "plot": "all original test61 for fixed encoder seed2024 in both modes; raw signed CDF uses .05/.95, relative uses .9"}
    cp.write(out / "plan.json", plan)
    cp.write(out / "state.json", {"state": "prepared", "updated_at": cp.stamp()})
    return plan


def load_data(out):
    manifest = cp.read(OOF / "dataset_manifest.json")["split"]["phases"]
    old = cp.read(OOF / "plan.json")
    assert cp.digest(OOF / "dataset_manifest.json") == old["manifest_sha256"]
    point = cp.ROOT / "runs/split_20261026/seed_2024/CPMLP"
    assert cp.digest(point / "best_model.pt") == cp.read(point / "frozen_predictor.json")["checkpoint_sha256"]
    data = dict(np.load(cp.ROOT / "analysis" / cp.PRIMARY / "features.npz", allow_pickle=False))
    arrays, sources = {}, []
    directory = cp.ROOT / "shared_cache/split_20261026/processed_cache"
    for phase in PHASES:
        for path in sorted(directory.glob(f"{phase}_*.pkl")):
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            if payload["identity"]["files"] != manifest[phase]:
                continue
            fields = payload["fields"]
            assert fields["files"] == manifest[phase]
            counts = np.array([fields["cell_prefix_counts"][n] for n in manifest[phase]])
            assert np.all(counts == (100 if phase == "train" else 1))
            arrays[phase] = np.stack([fields["total_charge_discharge_curves"][i] for i in np.cumsum(counts)-1]).astype(np.float32)
            sources.append({"path": str(path), "sha256": cp.digest(path)})
            break
        else:
            raise FileNotFoundError(phase)
        np.testing.assert_array_equal(data[f"{phase}_names"], manifest[phase])
    hand = {p: feature_tools.features(x)["hand"] for p, x in arrays.items()}
    values = {p: x.reshape(len(x), 100, 900) for p, x in arrays.items()}
    values, mean, std = standardize(values["train"], values)
    table = pd.read_csv(OOF / "oof_predictions.csv")
    train = table[table.model == "CPMLP"].set_index("cell_file").loc[manifest["train"]]
    assert len(train) == 97 and not set(train.index) & set(manifest["dev"]+manifest["calibration"]+manifest["test"])
    pred = {p: np.maximum(0., data[f"{p}_prediction"]) for p in PHASES}
    train_pred, train_y = np.maximum(0., train.prediction.to_numpy()), train.eol.to_numpy()
    # All feature construction below is outcome-free; cal/test labels enter evaluation only.
    lr, lq, _ = feature_tools.life_design(pred["train"], np.r_[pred["calibration"], pred["test"]])
    _, _, dr, dq, _ = feature_tools.transform_extra(hand["train"], np.concatenate((hand["calibration"], hand["test"])), lr, lq, "hand")
    hand_train = np.column_stack((lr[:, :1], dr))
    hand_pool = np.column_stack((lq[:, :1], dq))
    _, hand_h = feature_tools.kernel_weights(hand_train, hand_train[:1])
    cohort = {j["model"]+str(j["train_seed"]): dict(np.load(cp.ROOT / "analysis" / j["key"] / "features.npz"))
              for j in cp.read(cp.ROOT / "plan.json")["jobs"] if j["split_seed"] == 20261026}
    gap = {}
    for phase in ("dev", "calibration", "test"):
        means = {}
        for architecture in ("CPMLP", "CPLSTM"):
            members = [item for key, item in cohort.items() if key.startswith(architecture)]
            assert len(members) == 3
            for item in members:
                np.testing.assert_array_equal(item[f"{phase}_names"], data[f"{phase}_names"])
            means[architecture] = np.mean([item[f"{phase}_prediction"] for item in members], axis=0)
        gap[phase] = np.abs(means["CPMLP"]-means["CPLSTM"])
    cuts = np.quantile(gap["dev"]/np.maximum(1., pred["dev"]), [1/3, 2/3])
    cp.write(out / "data_identity.json", {"cache_inputs": sources, "train": manifest["train"], "dev": manifest["dev"],
        "calibration": manifest["calibration"], "test": manifest["test"], "feature_normalization_fit": manifest["train"],
        "raw_input_shape": [100, 900], "train_target_source": "OOF auxiliary predictions, not in-sample main predictions"})
    return {"x": values, "input_mean": mean, "input_std": std, "data": data, "pred": pred,
            "train_pred": train_pred, "train_y": train_y, "hand_pool": hand_pool, "hand_h": hand_h,
            "pool_gap": np.r_[gap["calibration"], gap["test"]], "gap_cuts": cuts, "sources": sources}


def infer(model, x, device):
    predictions, embeddings = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            q, z = model(torch.from_numpy(x[start:start+32]).to(device))
            predictions.append(q.cpu().numpy()); embeddings.append(z.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embeddings)


def train_localizer(bundle, mode, seed, folder, plan, smoke=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SequenceLocalizer().to(device)
    initial = copy.deepcopy(model.state_dict())
    target = residual_target(bundle["train_y"], bundle["train_pred"], mode)
    val_target = residual_target(bundle["data"]["dev_eol"], bundle["pred"]["dev"], mode)
    target_scale = max(float(target.std()), 1e-3)
    x = torch.from_numpy(bundle["x"]["train"]).to(device)
    y = torch.tensor(target/target_scale, dtype=torch.float32, device=device)
    vx = torch.from_numpy(bundle["x"]["dev"]).to(device)
    vy = torch.tensor(val_target/target_scale, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best, stale, history, best_state, started = np.inf, 0, [], None, time.monotonic()
    for epoch in range(1, (2 if smoke else plan["epochs"])+1):
        model.train()
        indices = torch.randperm(len(x), device=device)
        total = 0.
        for batch in indices.split(16):
            optimizer.zero_grad(set_to_none=True)
            q, _ = model(x[batch]); loss = pinball(q, y[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite quantile loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            total += float(loss.detach())*len(batch)
        model.eval()
        with torch.inference_mode():
            validation = float(pinball(model(vx)[0], vy))
        if not np.isfinite(validation):
            raise FloatingPointError("Nonfinite development loss")
        history.append({"epoch": epoch, "train_pinball_scaled": total/len(x), "dev_pinball_scaled": validation})
        if validation < best-1e-8:
            best, best_epoch, stale, best_state = validation, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        cp.write(folder / "progress.json", {"epoch": epoch, "best_epoch": best_epoch, "dev_pinball_scaled": validation, "updated_at": cp.stamp()})
        logging.info("%s seed%d epoch%d train%.5f dev%.5f", mode, seed, epoch, total/len(x), validation)
        if stale >= plan["patience"] or time.monotonic()-started > plan["budget_seconds"]:
            break
    model.load_state_dict(best_state)
    learned = {p: infer(model, values, device) for p, values in bundle["x"].items()}
    model.load_state_dict(initial)
    random_states = {p: infer(model, bundle["x"][p], device)[1] for p in ("train", "calibration", "test")}
    torch.save({"state_dict": best_state, "initial_state_dict": initial, "best_epoch": best_epoch,
                "mode": mode, "seed": seed, "target_scale": target_scale,
                "input_mean": bundle["input_mean"], "input_std": bundle["input_std"]}, folder / "localizer.pt")
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    np.savez_compressed(folder / "representations.npz", **{f"{p}_embedding": pair[1] for p, pair in learned.items()},
                        **{f"{p}_head_quantiles": pair[0]*target_scale for p, pair in learned.items()},
                        **{f"random_{p}": values for p, values in random_states.items()})
    return learned, random_states, {"best_epoch": best_epoch, "epochs_run": epoch, "target_scale": target_scale,
        "train_temperature": temperature(learned["train"][1]), "random_temperature": temperature(random_states["train"]),
        "train_embedding_std_mean": float(learned["train"][1].std(axis=0).mean()), "device": device,
        "parameters": sum(p.numel() for p in model.parameters()), "elapsed_seconds": time.monotonic()-started}


def bounds(bundle, mode, learned_pool, random_pool, info, ci, ti, repetition):
    pred = np.r_[bundle["pred"]["calibration"], bundle["pred"]["test"]]
    truth = np.r_[bundle["data"]["calibration_eol"], bundle["data"]["test_eol"]]
    error = truth[ci]-pred[ci]
    scale = np.maximum(1., pred)
    uniform = np.ones((len(ti), len(ci))) / len(ci)
    trained = soft_weights(learned_pool[ci], learned_pool[ti], info["train_temperature"])
    random_w = soft_weights(random_pool[ci], random_pool[ti], info["random_temperature"])
    hand = soft_weights(bundle["hand_pool"][ci], bundle["hand_pool"][ti], 2*bundle["hand_h"]**2)
    result, weights = {}, {"global": uniform, "trained": trained, "random": random_w, "hand": hand}
    for name, s in (("split_abs", np.ones(len(pred))), ("split_relative", scale), ("split_gap6", scale+bundle["pool_gap"])):
        radius = cp.finite_quantile(np.abs(error)/s[ci])*s[ti]
        result[name] = (pred[ti]-radius, pred[ti]+radius)
    for name, w in (("global_relative_empirical", uniform), ("hand_relative_empirical", hand)):
        radius = weighted_q(np.abs(error)/scale[ci], w, .9)*scale[ti]
        result[name] = (pred[ti]-radius, pred[ti]+radius)
    for name, w in (("global_signed_empirical", uniform),):
        result[name] = (pred[ti]+weighted_q(error, w, .05), pred[ti]+weighted_q(error, w, .95))
    full = np.ones((len(ti), len(ci)+1))
    result["global_signed_finite"] = (pred[ti]-weighted_q(-error, full, .95, True), pred[ti]+weighted_q(error, full, .95, True))
    z = learned_pool/np.sqrt(info["train_temperature"])
    normals = np.random.default_rng(20261111+repetition).normal(size=(len(ti), z.shape[1]))
    rw, _, _ = cp.random_weights(z[ci], z[ti], 3., normals)
    weights["rlcp"] = rw/rw.sum(axis=1, keepdims=True)
    if mode == "raw_signed":
        for name, w in (("DSCP_raw_signed", trained), ("random_raw_signed", random_w)):
            result[name] = (pred[ti]+weighted_q(error, w, .05), pred[ti]+weighted_q(error, w, .95))
        radius = weighted_q(np.abs(error), trained, .9)
        result["DSCP_raw_absolute"] = (pred[ti]-radius, pred[ti]+radius)
        radius = weighted_q(np.abs(error), rw, .9, True)
        result["RLCP_raw_absolute"] = (pred[ti]-radius, pred[ti]+radius)
    else:
        for name, w, atom in (("DSCP_relative", trained, False), ("random_relative", random_w, False), ("RLCP_relative", rw, True)):
            radius = weighted_q(np.abs(error)/scale[ci], w, .9, atom)*scale[ti]
            result[name] = (pred[ti]-radius, pred[ti]+radius)
    return result, weights


def evaluate(bundle, job, learned, random_states, info, folder, plan):
    pool = np.r_[learned["calibration"][1], learned["test"][1]]
    random_pool = np.r_[random_states["calibration"], random_states["test"]]
    truth = np.r_[bundle["data"]["calibration_eol"], bundle["data"]["test_eol"]]
    pred = np.r_[bundle["pred"]["calibration"], bundle["pred"]["test"]]
    names = np.r_[bundle["data"]["calibration_names"], bundle["data"]["test_names"]]
    groups = np.searchsorted(bundle["gap_cuts"], bundle["pool_gap"]/np.maximum(1., pred), side="right")
    rng, metrics, cells, grouped = np.random.default_rng(20261028), [], [], []
    for repetition in range(plan["resplits"]+1):
        order = np.arange(122) if repetition == 0 else rng.permutation(122)
        ci, ti = order[:61], order[61:]
        result, weights = bounds(bundle, job["mode"], pool, random_pool, info, ci, ti, repetition)
        for method, (lower, upper) in result.items():
            lower = np.maximum(0., lower); upper = np.maximum(lower, upper)
            width = upper-lower
            covered = (lower <= truth[ti]) & (truth[ti] <= upper)
            score = width+20*(np.maximum(lower-truth[ti], 0)+np.maximum(truth[ti]-upper, 0))
            metrics.append({"job": job["key"], "mode": job["mode"], "seed": job["seed"], "method": method, "repetition": repetition,
                "coverage": covered.mean(), "mean_width": width.mean(), "interval_score": score.mean(),
                "infinite": int(np.isinf(width).sum()), "mean_ess_trained": float(np.mean(1/np.square(weights["trained"]).sum(axis=1)))})
            for group in range(3):
                keep = groups[ti] == group
                if keep.any():
                    grouped.append({"job": job["key"], "method": method, "repetition": repetition, "group": group, "n": int(keep.sum()),
                        "coverage": covered[keep].mean(), "mean_width": width[keep].mean(), "interval_score": score[keep].mean()})
            if repetition == 0:
                for j, i in enumerate(ti):
                    cells.append({"job": job["key"], "mode": job["mode"], "seed": job["seed"], "method": method, "cell_file": names[i],
                        "prediction": pred[i], "eol": truth[i], "lower": lower[j], "upper": upper[j], "width": width[j],
                        "covered": bool(covered[j]), "interval_score": score[j], "gap_group": groups[i]})
        if repetition == 0:
            np.savez_compressed(folder / "calibration_distributions.npz", cal_names=names[ci], test_names=names[ti],
                cal_signed_error=truth[ci]-pred[ci], cal_abs_relative=np.abs(truth[ci]-pred[ci])/np.maximum(1., pred[ci]),
                test_prediction=pred[ti], test_eol=truth[ti], test_head_quantiles=learned["test"][0]*info["target_scale"],
                **weights)
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(cells).to_csv(folder / "original_intervals.csv", index=False)
    pd.DataFrame(grouped).to_csv(folder / "group_metrics.csv", index=False)
    return pd.DataFrame(metrics), pd.DataFrame(cells)


def finish(out, all_metrics, all_cells):
    metrics, cells = pd.concat(all_metrics, ignore_index=True), pd.concat(all_cells, ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False); cells.to_csv(out / "original_intervals.csv", index=False)
    rows = []
    for (mode, seed, method), part in metrics[metrics.repetition > 0].groupby(["mode", "seed", "method"]):
        rows.append({"mode": mode, "seed": seed, "method": method, **{k: float(np.mean(part[k].to_numpy())) for k in ("coverage", "mean_width", "interval_score")},
                     "infinite_cell_fraction": float(part.infinite.mean()/61)})
    summary = pd.DataFrame(rows); summary.to_csv(out / "summary.csv", index=False)
    # Reuse archived original test baselines as an external replay check.
    prior = pd.read_csv(cp.ROOT / "disagreement_cp_20261028/original_intervals.csv")
    prior = prior[prior.job == cp.PRIMARY]
    for method, old in (("split_abs", "split_cp"), ("split_relative", "relative_cp"), ("split_gap6", "gap_scale_cp")):
        reference = prior[prior.method == old].set_index("cell_file")
        for _, part in cells[cells.method == method].groupby("job"):
            reference_values = reference.loc[part.cell_file]
            np.testing.assert_allclose(part.lower, reference_values.lower, atol=1e-7)
            np.testing.assert_allclose(part.upper, reference_values.upper, atol=1e-7)
    np.testing.assert_allclose(cells.width, cells.upper-cells.lower)
    np.testing.assert_array_equal(cells.covered, (cells.lower <= cells.eol) & (cells.eol <= cells.upper))
    lines = ["# DSCP-Local：固定T100横向适配", "", "原CPMLP点预测器保持冻结。LSTM学习残差三个分位数，局部校准使用隐状态而非头输出。所有3随机种子与两种分数全部报告。",
        "raw_signed为论文机制适配，使用有符号残差和经验5%/95%端点；abs_relative为本项目归一化绝对分数适配。",
        "DSCP_*是论文式确定性softmax加权经验CDF，无有限样本覆盖保证声明；RLCP_*另用Gaussian随机中心及测试∞原子。",
        "", "## 固定池200次重分平均（不是200个独立数据集）", "", "|模式|encoder种子|方法|覆盖|宽度|Interval score|∞比例|", "|---|---:|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"|{row.mode}|{row.seed}|{row.method}|{row.coverage:.2%}|{row.mean_width:.1f}|{row.interval_score:.1f}|{row.infinite_cell_fraction:.3%}|")
    lines += ["", "## 解释范围", "", "原始test61单独保存在original_intervals.csv。相同分数下比较global、random、trained；与split_relative/gap6强基线同时比较，不能仅看网络训练损失或挑最好种子。",
        "原文的时序上下文含历史可观测残差，本任务没有同电芯历史EOL残差，因此用前100周期曲线代替；这是方法适配，不是原文数据集完整复现。",
        "train97监督来自外折辅助模型的OOF误差，主模型训练规模和条件误差可能不同。dev20只用于选encoder epoch，cal/test不参与训练或温度选择；历史标签已查看，整体仍为探索。",
        "有符号和对称区间分开列，经验CDF与有限样本校正分开列；不能把分数/区间形状变化全部算为学习距离收益。",
        "输入标准化仅train97，温度仅train隐表示。图保留seed2024两模式全部test61。原始cal/test、模型与旧代码未修改。",
        "论文：https://arxiv.org/abs/2610.02357。详细方法差异与配置见plan.json。"]
    (out / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    cp.write(out / "evaluation_audit.json", {"baseline_replay": True, "interval_rows": len(cells), "all_seeds_reported": True,
        "all_infinities_retained": True, "no_cal_test_training": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); out = args.out.resolve()
    plan = prepare(out)
    if args.prepare:
        print(out / "plan.json"); return
    current_state = cp.read(out / "state.json")["state"]
    if current_state in ("running", "smoke"):
        raise RuntimeError("Queue already marked active; inspect its PID before restarting")
    if current_state == "complete":
        raise RuntimeError("Already complete; preserve results")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / ("smoke.log" if args.smoke else "console.log"), encoding="utf-8")])
    torch.set_num_threads(4)
    started = time.monotonic()
    try:
        cp.write(out / "state.json", {"state": "smoke" if args.smoke else "running", "pid": os.getpid(), "updated_at": cp.stamp(), "completed_jobs": 0, "total_jobs": 6})
        bundle = load_data(out)
        all_metrics, all_cells = [], []
        for i, job in enumerate(plan["jobs"][:1] if args.smoke else plan["jobs"]):
            folder = out / ("smoke" if args.smoke else "jobs") / job["key"]
            folder.mkdir(parents=True, exist_ok=True)
            learned, random_states, info = train_localizer(bundle, job["mode"], job["seed"], folder, plan, args.smoke)
            cp.write(folder / "info.json", info)
            if args.smoke:
                pool = np.r_[learned["calibration"][1], learned["test"][1]]
                random_pool = np.r_[random_states["calibration"], random_states["test"]]
                result, _ = bounds(bundle, job["mode"], pool, random_pool, info, np.arange(61), np.arange(61, 122), 0)
                assert len(result) == 11 and all(len(lo) == len(up) == 61 for lo, up in result.values())
                cp.write(out / "smoke_audit.json", {"passed": True, "device": info["device"], "scientific_results": False})
                cp.write(out / "state.json", {"state": "prepared", "smoke_passed": True, "updated_at": cp.stamp()})
                return
            a, b = evaluate(bundle, job, learned, random_states, info, folder, plan)
            all_metrics.append(a); all_cells.append(b)
            cp.write(folder / "complete.json", {"job": job, "info": info, "checkpoint_sha256": cp.digest(folder / "localizer.pt")})
            cp.write(out / "state.json", {"state": "running", "pid": os.getpid(), "completed_jobs": i+1, "total_jobs": 6, "updated_at": cp.stamp()})
            if time.monotonic()-started > plan["budget_seconds"]:
                raise TimeoutError("Bounded queue exceeded30minutes")
        finish(out, all_metrics, all_cells)
        assert all(cp.digest(row["path"]) == row["sha256"] for row in plan["dependencies"]+bundle["sources"])
        cp.write(out / "state.json", {"state": "complete", "completed_jobs": 6, "updated_at": cp.stamp(), "elapsed_seconds": time.monotonic()-started, "plots": "pending"})
        logging.info("COMPLETE %s", out / "RESULTS.md")
    except Exception as error:
        cp.write(out / "state.json", {"state": "failed", "pid": os.getpid(), "updated_at": cp.stamp(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
