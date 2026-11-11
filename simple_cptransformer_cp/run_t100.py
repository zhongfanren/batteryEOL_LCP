"""Minimal fixed-T=100 CPTransformer + symmetric/asymmetric CP experiment."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = ROOT / "third_party" / "BatteryLife"
sys.path.insert(0, str(UPSTREAM))
from data_provider.data_split_recorder import split_recorder  # noqa: E402
from models.CPTransformer import Model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--lower-tail-fraction", type=float, default=0.5,
                   help="Fraction of alpha assigned to EOL below the lower endpoint")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--pca-components", type=int, default=16)
    p.add_argument("--min-ess", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--e-layers", type=int, default=6)
    p.add_argument("--d-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def split_files(seed):
    official_train = list(split_recorder.MATR_train_files)
    shuffled = official_train.copy()
    random.Random(seed).shuffle(shuffled)
    model_val = shuffled[:12]
    train = shuffled[12:]
    return {"train": train, "model_validation": model_val,
            "calibration": list(split_recorder.MATR_val_files),
            "test": list(split_recorder.MATR_test_files)}


def resample(values, length=150):
    values = np.asarray(values, dtype=np.float64)
    base = np.arange(1, len(values) + 1)
    target = np.linspace(1, len(values) + 1, num=length, endpoint=True)
    return np.interp(target, base, values)


def load_cell(path, eol, t=100, curve_len=300):
    with path.open("rb") as f:
        data = pickle.load(f)
    cycles = data["cycle_data"]
    if eol <= t or len(cycles) < t:
        raise ValueError(f"needs >={t} recorded cycles and EOL>{t}")
    nominal = float(data["nominal_capacity_in_Ah"])
    curves = []
    for cycle in cycles[:t]:
        voltage = np.asarray(cycle["voltage_in_V"], dtype=float)
        current = np.asarray(cycle["current_in_A"], dtype=float)
        charge_cap = np.asarray(cycle["charge_capacity_in_Ah"], dtype=float)
        discharge_cap = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=float)
        # Match upstream bfill after negative-capacity rows are invalidated.
        bad = (charge_cap < 0) | (discharge_cap < 0)
        if np.any(bad):
            import pandas as pd
            frame = pd.DataFrame({"v": voltage, "i": current, "q": charge_cap, "d": discharge_cap})
            frame.loc[bad] = np.nan; frame.bfill(inplace=True)
            voltage, current, charge_cap, discharge_cap = (frame[k].to_numpy() for k in ("v", "i", "q", "d"))
        charge_end = np.nonzero(current / nominal >= 0.01)[0][-1]
        dv, di, dq = voltage[charge_end:], current[charge_end:], discharge_cap[charge_end:]
        keep = np.abs(di / nominal) > 0.01
        dv, di, dq = dv[keep], di[keep], dq[keep]
        cv, ci, cq = voltage[:charge_end], current[:charge_end], charge_cap[:charge_end]
        vr = np.concatenate((resample(cv), resample(dv)))
        ir = np.concatenate((resample(ci), resample(di)))
        qr = np.concatenate((resample(cq), resample(dq)))
        curves.append(np.stack((vr / np.max(vr), ir / nominal, qr / nominal)))
    arr = np.asarray(curves, dtype=np.float32)
    if arr.shape != (t, 3, curve_len) or not np.isfinite(arr).all():
        raise ValueError(f"invalid tensor {arr.shape}")
    return arr


def prepare_data(data_root, splits, out):
    labels = json.loads((data_root / "Life labels" / "MATR_labels.json").read_text())
    result, audit = {}, {}
    for split, files in splits.items():
        xs, ys, ids, excluded = [], [], [], []
        for name in files:
            try:
                if name not in labels: raise ValueError("missing EOL label")
                x = load_cell(data_root / "MATR" / name, float(labels[name]))
                xs.append(x); ys.append(float(labels[name])); ids.append(name.removesuffix(".pkl"))
            except Exception as exc:
                excluded.append({"file": name, "reason": str(exc)})
        result[split] = (np.asarray(xs), np.asarray(ys, dtype=np.float32), ids)
        audit[split] = {"requested": files, "included": ids, "excluded": excluded,
                        "n_included": len(ids)}
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return result


def model_config(args):
    return SimpleNamespace(d_ff=args.d_ff, d_model=args.d_model, charge_discharge_length=300,
        early_cycle_threshold=100, dropout=0.0, e_layers=args.e_layers, d_layers=args.d_layers,
        factor=3, n_heads=args.n_heads, activation="relu", output_num=1)


def infer(model, x, device, batch_size, embeddings=False):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size)
    preds, embs = [], []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device); mask = torch.ones((len(xb), 100), device=device)
            value = model(xb, mask, return_embedding=embeddings)
            if embeddings:
                value, emb = value; embs.append(emb.cpu().numpy())
            preds.append(value.squeeze(-1).cpu().numpy())
    return np.concatenate(preds), (np.concatenate(embs) if embeddings else None)


def point_metrics(y, pred):
    return {"rmse": math.sqrt(mean_squared_error(y, pred)), "mae": mean_absolute_error(y, pred),
            "mape": mean_absolute_percentage_error(y, pred), "r2": r2_score(y, pred)}


def interval_metrics(y, lo, hi, alpha):
    width = hi - lo
    score = width + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width)), "mean_interval_score": float(np.mean(score))}


def weighted_quantile(values, weights, q):
    order = np.argsort(values); v, w = values[order], weights[order]
    return float(v[min(np.searchsorted(np.cumsum(w), q * np.sum(w), side="left"), len(v)-1)])


def conformal_upper_quantile(values, alpha):
    """Finite-sample upper quantile with one-based conformal rank."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("conformal scores must be a non-empty finite 1-D array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        return float("inf"), rank
    return float(np.partition(values, rank - 1)[rank - 1]), rank


def main():
    args = parse_args(); seed_all(args.seed)
    if not 0 < args.lower_tail_fraction < 1:
        raise ValueError("--lower-tail-fraction must be strictly between 0 and 1")
    stamp = datetime.now().strftime("t100_%Y%m%d_%H%M%S_%f")
    out = HERE / "runs" / stamp; out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy(); config["data_root"] = str(config["data_root"]); config["T"] = 100
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    data = prepare_data(args.data_root, split_files(args.seed), out)
    scaler = StandardScaler().fit(data["train"][1].reshape(-1, 1))
    device = torch.device(args.device); model = Model(model_config(args)).to(device)
    train_x, train_y = data["train"][:2]
    y_scaled = scaler.transform(train_y[:, None]).astype(np.float32).ravel()
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y_scaled)),
                        batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr); best, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); mask = torch.ones((len(xb), 100), device=device)
            opt.zero_grad(); loss = torch.mean((model(xb, mask).squeeze(-1) - yb) ** 2)
            loss.backward(); opt.step(); losses.append(loss.item())
        vp, _ = infer(model, data["model_validation"][0], device, args.batch_size)
        vp = scaler.inverse_transform(vp[:, None]).ravel()
        vmape = mean_absolute_percentage_error(data["model_validation"][1], vp)
        history.append({"epoch": epoch, "train_mse_scaled": float(np.mean(losses)), "validation_mape": vmape})
        if vmape < best:
            best, best_state, stale = vmape, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state); torch.save(best_state, out / "model.pt")
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    predictions, embeddings = {}, {}
    for split in data:
        p, e = infer(model, data[split][0], device, args.batch_size, embeddings=True)
        predictions[split] = scaler.inverse_transform(p[:, None]).ravel(); embeddings[split] = e
    cal_signed_resid = data["calibration"][1] - predictions["calibration"]
    cal_resid = np.abs(cal_signed_resid)
    split_q, rank = conformal_upper_quantile(cal_resid, args.alpha)
    alpha_lower = args.alpha * args.lower_tail_fraction
    alpha_upper = args.alpha - alpha_lower
    # A lower miss has score prediction-y=-residual; an upper miss has score
    # y-pred=residual.  Large positive errors therefore expand only the upper side.
    asymmetric_q_lower, asymmetric_lower_rank = conformal_upper_quantile(
        -cal_signed_resid, alpha_lower)
    asymmetric_q_upper, asymmetric_upper_rank = conformal_upper_quantile(
        cal_signed_resid, alpha_upper)
    emb_scaler = StandardScaler().fit(embeddings["train"])
    z = {k: emb_scaler.transform(v) for k, v in embeddings.items()}
    ncomp = min(args.pca_components, len(z["train"]) - 1, z["train"].shape[1])
    pca = PCA(n_components=ncomp, random_state=args.seed).fit(z["train"])
    z = {k: pca.transform(v) for k, v in z.items()}
    td = np.sqrt(np.sum((z["train"][:, None] - z["train"][None, :]) ** 2, axis=2))
    bandwidth = float(np.median(td[np.triu_indices_from(td, k=1)]))
    test_y, test_pred, test_ids = data["test"][1], predictions["test"], data["test"][2]
    lcp_q, ess, fallback = [], [], []
    for zi in z["test"]:
        d2 = np.sum((z["calibration"] - zi) ** 2, axis=1)
        w = np.exp(-d2 / (2 * max(bandwidth, 1e-12) ** 2))
        cur_ess = float(w.sum() ** 2 / np.sum(w ** 2)); use_global = cur_ess < args.min_ess
        lcp_q.append(split_q if use_global else weighted_quantile(cal_resid, w, 1 - args.alpha))
        ess.append(cur_ess); fallback.append(use_global)
    lcp_q, ess = np.asarray(lcp_q), np.asarray(ess)
    slo, shi = test_pred - split_q, test_pred + split_q
    alo, ahi = test_pred - asymmetric_q_lower, test_pred + asymmetric_q_upper
    llo, lhi = test_pred - lcp_q, test_pred + lcp_q
    metrics = {"best_validation_mape": best, "best_epoch": int(np.argmin([h["validation_mape"] for h in history]) + 1),
        "point": point_metrics(test_y, test_pred),
        "split_cp": {**interval_metrics(test_y, slo, shi, args.alpha), "q": split_q, "rank": rank},
        "asymmetric_split_cp": {
            **interval_metrics(test_y, alo, ahi, args.alpha),
            "q_lower": asymmetric_q_lower, "q_upper": asymmetric_q_upper,
            "alpha_lower": alpha_lower, "alpha_upper": alpha_upper,
            "lower_rank": asymmetric_lower_rank, "upper_rank": asymmetric_upper_rank},
        "lcp": {**interval_metrics(test_y, llo, lhi, args.alpha), "bandwidth": bandwidth,
                "pca_components": ncomp, "ess_min": float(ess.min()), "ess_median": float(np.median(ess)),
                "ess_mean": float(ess.mean()), "fallback_count": int(np.sum(fallback))}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "calibration_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["cell_id", "eol", "eol_hat", "signed_residual", "absolute_residual"])
        w.writerows(zip(data["calibration"][2], data["calibration"][1], predictions["calibration"],
                        cal_signed_resid, cal_resid))
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["cell_id","eol","eol_hat","split_lower","split_upper",
            "asymmetric_lower","asymmetric_upper","lcp_lower","lcp_upper","lcp_q","lcp_ess","lcp_fallback"])
        w.writerows(zip(test_ids, test_y, test_pred, slo, shi, alo, ahi, llo, lhi, lcp_q, ess, fallback))
    print(json.dumps({"run_dir": str(out), **metrics}, indent=2))


if __name__ == "__main__":
    main()
