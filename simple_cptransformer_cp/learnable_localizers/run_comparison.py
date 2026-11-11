"""Controlled T=100 comparison of fixed and learnable LCP localizers.

This is a development-only experiment on the 102 official MATR training cells.
For every outer fold, predictor/localizer development, conformal calibration,
and evaluation cells are disjoint.  The weighted intervals are an empirical
basic-LCP diagnostic; no new local-coverage theorem is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.fft import dct
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch import nn


HERE = Path(__file__).resolve().parent
ERROR_DIR = HERE.parent / "error_predictability"
sys.path.insert(0, str(ERROR_DIR))
import run as previous  # noqa: E402
import run_crossfit_robust as robust  # noqa: E402


SEEDS = [42, 2021, 2024]
METHODS = ["split", "fixed_rbf", "hopfield_time", "frequency"]
ALPHA = 0.1
EMBED_DIM = 8
HIDDEN_DIM = 16
MIN_ESS = 10.0
LOCALIZER_EPOCHS = 500
LOCALIZER_PATIENCE = 40
LOCALIZER_LR = 1e-3
LOCALIZER_WEIGHT_DECAY = 1e-3


def save_json(path, value):
    previous.save_json(path, value)


def time_features(x):
    """57-dimensional time-domain summary with cycle order preserved."""
    series = x.astype(np.float64).mean(axis=-1)  # cell, 100 cycles, 3 channels
    grid = np.linspace(0.0, 99.0, 16)
    sampled = np.stack([
        np.stack([np.interp(grid, np.arange(100), row[:, channel])
                  for channel in range(3)], axis=1)
        for row in series
    ])
    centered_cycle = np.arange(100, dtype=float) - 49.5
    slope = np.einsum("ntc,t->nc", series, centered_cycle) / np.sum(centered_cycle ** 2)
    extras = np.column_stack([
        series.std(axis=1), np.diff(series, axis=1).std(axis=1), slope
    ]).reshape(len(series), -1)
    result = np.column_stack([sampled.reshape(len(series), -1), extras])
    assert result.shape[1] == 57 and np.isfinite(result).all()
    return result.astype(np.float32)


def frequency_features(x):
    """57-dimensional cycle-axis spectrum: 16 coefficients plus 3 band RMS values."""
    series = x.astype(np.float64).mean(axis=-1)
    spectrum = dct(series, axis=1, norm="ortho")
    low = spectrum[:, :16].reshape(len(series), -1)
    bands = np.concatenate([
        np.sqrt(np.mean(spectrum[:, start:stop] ** 2, axis=1) + 1e-12)
        for start, stop in [(16, 32), (32, 64), (64, 100)]
    ], axis=1)
    result = np.column_stack([low, bands])
    assert result.shape[1] == 57 and np.isfinite(result).all()
    return result.astype(np.float32)


def fit_robust_scaler(values):
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.where(q75 - q25 > 1e-8, q75 - q25, 1.0)
    return median.astype(np.float32), scale.astype(np.float32)


def apply_scaler(values, median, scale):
    return np.clip((values - median) / scale, -5.0, 5.0).astype(np.float32)


class PairLocalizer(nn.Module):
    """Shared two-stream gated encoder plus Hopfield/attention pair scorer.

    The first 48 inputs encode the ordered trend (time samples or low DCT
    coefficients); the final nine encode local variation/high-band energy.
    Both localizers therefore have exactly the same architecture and capacity.
    """

    def __init__(self, input_dim=57, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        if input_dim != 57:
            raise ValueError("The controlled localizers require 48+9 input features")
        self.low_encoder = nn.Sequential(
            nn.Linear(48, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.high_encoder = nn.Linear(9, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.embed_dim = embed_dim

    def encode(self, x):
        low = self.low_encoder(x[:, :48])
        high = self.high_encoder(x[:, 48:])
        return self.layer_norm(low + self.gate(low) * high)

    def scores(self, query_x, reference_x):
        query = self.query(self.encode(query_x))
        key = self.key(self.encode(reference_x))
        return query @ key.T / math.sqrt(self.embed_dim)


def effective_sample_size(weights):
    return 1.0 / np.sum(np.square(weights), axis=1)


def softmax_rows(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def weights_with_min_ess(logits, min_ess=MIN_ESS):
    """Temperature-smooth each row using a fixed, label-free ESS rule."""
    logits = np.asarray(logits, dtype=float)
    output = np.empty_like(logits)
    temperatures = np.ones(len(logits), dtype=float)
    attainable = min(float(logits.shape[1]), min_ess)
    for row, values in enumerate(logits):
        temperature = 1.0
        weights = softmax_rows((values / temperature)[None, :])[0]
        while 1.0 / np.sum(weights ** 2) + 1e-9 < attainable and temperature < 1024.0:
            temperature *= 2.0
            weights = softmax_rows((values / temperature)[None, :])[0]
        if temperature > 1.0:
            low, high = temperature / 2.0, temperature
            for _ in range(30):
                middle = (low + high) / 2.0
                candidate = softmax_rows((values / middle)[None, :])[0]
                if 1.0 / np.sum(candidate ** 2) < attainable:
                    low = middle
                else:
                    high = middle
            temperature = high
            weights = softmax_rows((values / temperature)[None, :])[0]
        output[row] = weights
        temperatures[row] = temperature
    return output, temperatures


def training_weights(model, features):
    logits = model.scores(features, features)
    logits = logits.masked_fill(torch.eye(len(features), dtype=torch.bool,
                                         device=features.device), float("-inf"))
    return torch.softmax(logits, dim=1)


def localizer_loss(weights, query_target, reference_target, loss_name):
    if loss_name == "mean_mse":
        estimate = weights @ reference_target
        return torch.mean((estimate - query_target) ** 2)
    if loss_name == "crps":
        # CRPS of a discrete weighted residual distribution.  Unlike a mean
        # regression loss, this proper score trains all of the distribution,
        # including the tail subsequently used by the weighted quantile.
        query_reference = torch.abs(query_target[:, None] - reference_target[None, :])
        reference_pairwise = torch.abs(reference_target[:, None] - reference_target[None, :])
        first = torch.sum(weights * query_reference, dim=1)
        second = 0.5 * torch.einsum("bi,ij,bj->b", weights, reference_pairwise, weights)
        return torch.mean(first - second)
    raise ValueError(f"Unknown localizer loss: {loss_name}")


def fit_localizer(features, target, seed, device, loss_name):
    """Select epoch on held-out residuals, then refit on every development cell."""
    previous.base.seed_all(seed)
    order = np.random.default_rng(seed).permutation(len(features))
    nval = max(8, int(round(0.2 * len(order))))
    validation, fitting = order[:nval], order[nval:]
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(target, dtype=torch.float32, device=device)

    def new_model():
        return PairLocalizer(features.shape[1]).to(device)

    model = new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LOCALIZER_LR,
                                 weight_decay=LOCALIZER_WEIGHT_DECAY)
    best_loss, best_epoch, best_state, stale = float("inf"), 1, None, 0
    history = []
    fit_tensor = torch.tensor(fitting, dtype=torch.long, device=device)
    val_tensor = torch.tensor(validation, dtype=torch.long, device=device)
    for epoch in range(1, LOCALIZER_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        weights = training_weights(model, x[fit_tensor])
        residual_loss = localizer_loss(weights, y[fit_tensor], y[fit_tensor], loss_name)
        ess = 1.0 / torch.sum(weights ** 2, dim=1)
        ess_penalty = torch.mean(torch.relu(MIN_ESS - ess) ** 2) / (MIN_ESS ** 2)
        loss = residual_loss + 0.05 * ess_penalty
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_weights = torch.softmax(model.scores(x[val_tensor], x[fit_tensor]), dim=1)
            val_loss = float(localizer_loss(
                val_weights, y[val_tensor], y[fit_tensor], loss_name).cpu())
        history.append({"epoch": epoch, "fit_loss": float(residual_loss.detach().cpu()),
                        "validation_loss": val_loss,
                        "fit_ess_mean": float(ess.detach().mean().cpu())})
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch, best_state, stale = val_loss, epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= LOCALIZER_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Localizer did not produce a finite validation loss")

    # Refit from the same deterministic initialization for the selected epoch.
    previous.base.seed_all(seed)
    model = new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LOCALIZER_LR,
                                 weight_decay=LOCALIZER_WEIGHT_DECAY)
    for _ in range(best_epoch):
        model.train()
        optimizer.zero_grad()
        weights = training_weights(model, x)
        residual_loss = localizer_loss(weights, y, y, loss_name)
        ess = 1.0 / torch.sum(weights ** 2, dim=1)
        loss = residual_loss + 0.05 * torch.mean(
            torch.relu(MIN_ESS - ess) ** 2) / (MIN_ESS ** 2)
        loss.backward()
        optimizer.step()
    return model.eval(), {"selected_epoch": best_epoch,
                          "best_validation_loss": best_loss, "selection_history": history}


def learned_logits(model, query, reference, device):
    with torch.no_grad():
        q = torch.tensor(query, dtype=torch.float32, device=device)
        r = torch.tensor(reference, dtype=torch.float32, device=device)
        return model.scores(q, r).cpu().numpy()


def fixed_rbf_fit(features, dev_ids):
    n_components = min(EMBED_DIM, len(dev_ids) - 1, features.shape[1])
    pca = PCA(n_components=n_components, random_state=2024).fit(features[dev_ids])
    z = pca.transform(features)
    pairwise = np.sqrt(np.sum((z[dev_ids, None] - z[None, dev_ids]) ** 2, axis=2))
    upper = pairwise[np.triu_indices(len(dev_ids), 1)]
    bandwidth = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    return z, max(bandwidth, 1e-8)


def weighted_quantiles(values, weights, level):
    return np.asarray([previous.base.weighted_quantile(values, row, level)
                       for row in weights], dtype=float)


def fold_partition(n, fold, outer_train, outer_eval):
    pool = np.random.default_rng(2024 + fold).permutation(outer_train)
    calibration, development = pool[:20], pool[20:]
    assert len(set(development) & set(calibration)) == 0
    assert len(set(development) & set(outer_eval)) == 0
    assert len(set(calibration) & set(outer_eval)) == 0
    assert len(development) + len(calibration) + len(outer_eval) == n
    return development, calibration, outer_eval


def oof_development_predictions(x, y, development, seed, fold, device):
    prediction = np.full(len(development), np.nan)
    epochs = []
    inner = KFold(4, shuffle=True, random_state=12024 + fold)
    for inner_fold, (train_pos, eval_pos) in enumerate(inner.split(development)):
        pred, epoch = robust.train_and_predict(
            x, y, development[train_pos], development[eval_pos], seed,
            22024 + 100 * fold + inner_fold, device)
        prediction[eval_pos] = pred
        epochs.append({"role": f"inner_{inner_fold}", "epoch": epoch})
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Missing development OOF predictions")
    return prediction, epochs


def weight_diagnostics(weights, temperatures):
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    return {"ess_mean": float(effective_sample_size(weights).mean()),
            "ess_median": float(np.median(effective_sample_size(weights))),
            "max_weight_mean": float(np.max(weights, axis=1).mean()),
            "entropy_mean": float(entropy.mean()),
            "temperature_mean": float(np.mean(temperatures))}


def evaluate_fold(x, y, names, time_raw, frequency_raw, seed, fold,
                  outer_train, outer_eval, device, out, loss_name):
    development, calibration, evaluation = fold_partition(len(x), fold, outer_train, outer_eval)
    dev_oof, epochs = oof_development_predictions(x, y, development, seed, fold, device)
    predict_ids = np.concatenate([calibration, evaluation])
    external_pred, epoch = robust.train_and_predict(
        x, y, development, predict_ids, seed, 32024 + fold, device)
    epochs.append({"role": "final", "epoch": epoch})
    cal_pred = external_pred[:len(calibration)]
    eval_pred = external_pred[len(calibration):]
    dev_log_error = np.log1p(np.abs(y[development] - dev_oof)).astype(np.float32)
    cal_error = np.abs(y[calibration] - cal_pred)
    cal_log_error = np.log1p(cal_error)
    eval_log_error = np.log1p(np.abs(y[evaluation] - eval_pred))

    transformed = {}
    scalers = {}
    for name, raw in [("time", time_raw), ("frequency", frequency_raw)]:
        median, scale = fit_robust_scaler(raw[development])
        transformed[name] = apply_scaler(raw, median, scale)
        scalers[name] = {"median": median, "scale": scale}

    localizers, training = {}, {}
    for offset, name in enumerate(["time", "frequency"]):
        model, info = fit_localizer(transformed[name][development], dev_log_error,
                                    seed * 1000 + fold * 10 + offset, device, loss_name)
        localizers[name] = model
        training[name] = info

    level_rank = math.ceil((len(calibration) + 1) * (1 - ALPHA))
    level = min(1.0, level_rank / len(calibration))
    uniform_weights = np.full((len(evaluation), len(calibration)), 1.0 / len(calibration))
    weights = {"split": uniform_weights}
    temperatures = {"split": np.ones(len(evaluation))}

    time_z, bandwidth = fixed_rbf_fit(transformed["time"], development)
    distance2 = np.sum((time_z[evaluation, None] - time_z[None, calibration]) ** 2, axis=2)
    weights["fixed_rbf"], temperatures["fixed_rbf"] = weights_with_min_ess(
        -distance2 / (2 * bandwidth ** 2))
    for method, source in [("hopfield_time", "time"), ("frequency", "frequency")]:
        logits = learned_logits(localizers[source], transformed[source][evaluation],
                                transformed[source][calibration], device)
        weights[method], temperatures[method] = weights_with_min_ess(logits)

    rows = pd.DataFrame({"cell_id": [names[i][:-4] for i in evaluation],
                         "seed": seed, "outer_fold": fold,
                         "eol": y[evaluation], "prediction": eval_pred,
                         "target_log_error": eval_log_error})
    fold_metrics = {"seed": seed, "outer_fold": fold,
                    "development_n": len(development), "calibration_n": len(calibration),
                    "evaluation_n": len(evaluation), "conformal_rank": level_rank,
                    "weighted_quantile_level": level, "point_epoch": epoch,
                    "fixed_rbf_bandwidth": bandwidth, "localizer_training": training,
                    "point": previous.base.point_metrics(y[evaluation], eval_pred),
                    "methods": {}}
    for method in METHODS:
        risk_prediction = weights[method] @ cal_log_error
        q = weighted_quantiles(cal_error, weights[method], level)
        lower, upper = eval_pred - q, eval_pred + q
        rows[f"{method}_risk"] = risk_prediction
        rows[f"{method}_q"] = q
        rows[f"{method}_lower"] = lower
        rows[f"{method}_upper"] = upper
        metrics = previous.base.interval_metrics(y[evaluation], lower, upper, ALPHA)
        metrics.update({"risk_log_mse": float(np.mean((eval_log_error - risk_prediction) ** 2)),
                        "risk_spearman": (None if np.ptp(risk_prediction) == 0 else
                                          float(spearmanr(eval_log_error, risk_prediction).statistic)),
                        "miss_lower": int(np.sum(y[evaluation] < lower)),
                        "miss_upper": int(np.sum(y[evaluation] > upper)),
                        **weight_diagnostics(weights[method], temperatures[method])})
        fold_metrics["methods"][method] = metrics

    fold_dir = out / f"seed_{seed}" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(fold_dir / "evaluation.csv", index=False)
    save_json(fold_dir / "metrics.json", fold_metrics)
    save_json(fold_dir / "split.json", {
        "development": [names[i] for i in development],
        "calibration": [names[i] for i in calibration],
        "evaluation": [names[i] for i in evaluation],
        "predictor_epochs": epochs,
    })
    torch.save({name: model.state_dict() for name, model in localizers.items()},
               fold_dir / "localizers.pt")
    save_json(fold_dir / "feature_scalers.json", scalers)
    print(f"seed {seed}: learnable-localizer fold {fold + 1}/5", flush=True)
    return rows, fold_metrics


def aggregate_seed(rows, fold_metrics):
    y = rows.eol.to_numpy()
    prediction = rows.prediction.to_numpy()
    result = {"point": previous.base.point_metrics(y, prediction), "methods": {}}
    baseline_mse = float(np.mean((rows.target_log_error - rows.split_risk) ** 2))
    for method in METHODS:
        lower, upper = rows[f"{method}_lower"].to_numpy(), rows[f"{method}_upper"].to_numpy()
        risk = rows[f"{method}_risk"].to_numpy()
        risk_mse = float(np.mean((rows.target_log_error.to_numpy() - risk) ** 2))
        interval = previous.base.interval_metrics(y, lower, upper, ALPHA)
        result["methods"][method] = {
            **interval,
            "miss_lower": int(np.sum(y < lower)), "miss_upper": int(np.sum(y > upper)),
            "risk_log_mse": risk_mse,
            "risk_mse_improvement_vs_uniform": 0.0 if method == "split" else 1.0 - risk_mse / baseline_mse,
            "risk_spearman": (None if np.ptp(risk) == 0 else
                              float(spearmanr(rows.target_log_error, risk).statistic)),
            "positive_risk_folds": int(sum(
                metric["methods"][method]["risk_log_mse"] < metric["methods"]["split"]["risk_log_mse"]
                for metric in fold_metrics)),
            "ess_mean": float(np.mean([m["methods"][method]["ess_mean"] for m in fold_metrics])),
            "temperature_mean": float(np.mean([
                m["methods"][method]["temperature_mean"] for m in fold_metrics])),
        }
    return result


def aggregate_all(seed_metrics):
    summary = {}
    for method in METHODS:
        values = [metrics["methods"][method] for metrics in seed_metrics]
        summary[method] = {
            "coverage_mean": float(np.mean([v["coverage"] for v in values])),
            "mean_width_mean": float(np.mean([v["mean_width"] for v in values])),
            "interval_score_mean": float(np.mean([v["mean_interval_score"] for v in values])),
            "risk_improvement_mean": float(np.mean([
                v["risk_mse_improvement_vs_uniform"] for v in values])),
            "risk_spearman_mean": float(np.mean([
                v["risk_spearman"] for v in values if v["risk_spearman"] is not None])),
            "seeds_with_positive_risk_improvement": int(sum(
                v["risk_mse_improvement_vs_uniform"] > 0 for v in values)),
            "ess_mean": float(np.mean([v["ess_mean"] for v in values])),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--localizer-loss", choices=["mean_mse", "crps"], default="crps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "T": 100, "outer_folds": 5, "outer_split_seed": 2024,
        "calibration_per_fold": 20, "seeds": args.seeds, "alpha": ALPHA,
        "methods": METHODS, "embedding_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
        "localizer_encoder": "shared 48-feature trend + 9-feature variation gated encoder",
        "min_ess": MIN_ESS, "localizer_epochs": LOCALIZER_EPOCHS,
        "localizer_patience": LOCALIZER_PATIENCE, "localizer_lr": LOCALIZER_LR,
        "localizer_weight_decay": LOCALIZER_WEIGHT_DECAY,
        "localizer_loss": args.localizer_loss,
        "weighted_interval_status": "empirical basic-LCP diagnostic; no new local guarantee claimed",
        "predictor": vars(previous.CFG), "device": device,
        "data_root": str(args.data_root.resolve()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save_json(out / "config.json", config)
    print(f"RUN_DIR={out}", flush=True)

    names = list(previous.base.split_recorder.MATR_train_files)
    x, y, audit = previous.load_data(args.data_root, names)
    save_json(out / "data_audit.json", audit)
    time_raw, frequency_raw = time_features(x), frequency_features(x)
    outer = list(KFold(5, shuffle=True, random_state=2024).split(x))
    all_seed_metrics = []
    for seed in args.seeds:
        seed_rows, seed_folds = [], []
        for fold, (outer_train, outer_eval) in enumerate(outer):
            rows, metrics = evaluate_fold(
                x, y, names, time_raw, frequency_raw, seed, fold,
                outer_train, outer_eval, device, out, args.localizer_loss)
            seed_rows.append(rows)
            seed_folds.append(metrics)
        combined = pd.concat(seed_rows, ignore_index=True)
        assert len(combined) == 102 and combined.cell_id.nunique() == 102
        seed_dir = out / f"seed_{seed}"
        combined.to_csv(seed_dir / "evaluation.csv", index=False)
        metrics = aggregate_seed(combined, seed_folds)
        save_json(seed_dir / "metrics.json", metrics)
        all_seed_metrics.append(metrics)
        print(f"SEED {seed}: {json.dumps(metrics['methods'])}", flush=True)
    summary = aggregate_all(all_seed_metrics)
    save_json(out / "summary.json", summary)
    print(f"SUMMARY: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
