"""Experiment B: HUST-assisted representation learning for MATR LCP.

Only official MATR-train and HUST-train cells are used to develop the point
predictor and localizers.  MATR validation calibrates the intervals and the
already-viewed MATR test split provides an exploratory final evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

import run_comparison as comparison
import run_hust_replication as hust


HERE = Path(__file__).resolve().parent
SEEDS = [42, 2021, 2024]
METHODS = [
    "split",
    "fixed_rbf",
    "hopfield_random",
    "hopfield_ssl_matr",
    "hopfield_ssl_matr_hust",
]
SSL_EPOCHS = 300
SSL_MASK_RATE = 0.2
SSL_NOISE_STD = 0.05
SSL_LR = 1e-3
SSL_WEIGHT_DECAY = 1e-3


class MaskedFeatureAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.localizer = comparison.PairLocalizer()
        self.decoder = nn.Linear(comparison.EMBED_DIM, 57)

    def forward(self, values):
        return self.decoder(self.localizer.encode(values))


def pretrain_encoder(features, seed, device):
    """Fixed masked-denoising objective; no labels or validation/test cells."""
    comparison.previous.base.seed_all(seed)
    values = torch.tensor(features, dtype=torch.float32, device=device)
    model = MaskedFeatureAutoencoder().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SSL_LR, weight_decay=SSL_WEIGHT_DECAY
    )
    losses = []
    for _ in range(SSL_EPOCHS):
        model.train()
        optimizer.zero_grad()
        mask = torch.rand_like(values) < SSL_MASK_RATE
        corrupted = values + SSL_NOISE_STD * torch.randn_like(values)
        corrupted = torch.where(mask, torch.zeros_like(corrupted), corrupted)
        reconstruction = model(corrupted)
        loss = torch.sum((reconstruction - values) ** 2 * mask) / mask.sum().clamp_min(1)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.localizer.state_dict().items()
        if not key.startswith("query.") and not key.startswith("key.")
    }
    return state, {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
    }


def initialize_localizer(seed, device, encoder_state=None):
    comparison.previous.base.seed_all(seed)
    model = comparison.PairLocalizer().to(device)
    if encoder_state is not None:
        current = model.state_dict()
        for key, value in encoder_state.items():
            current[key] = value.to(device)
        model.load_state_dict(current)
    return model


def fit_localizer(features, target, seed, device, encoder_state=None):
    """MATR residual CRPS fine-tuning with the existing selection protocol."""
    order = np.random.default_rng(seed).permutation(len(features))
    n_validation = max(8, int(round(0.2 * len(order))))
    validation, fitting = order[:n_validation], order[n_validation:]
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(target, dtype=torch.float32, device=device)
    fitting_tensor = torch.tensor(fitting, dtype=torch.long, device=device)
    validation_tensor = torch.tensor(validation, dtype=torch.long, device=device)

    model = initialize_localizer(seed, device, encoder_state)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=comparison.LOCALIZER_LR,
        weight_decay=comparison.LOCALIZER_WEIGHT_DECAY,
    )
    best_loss, best_epoch, stale = float("inf"), 1, 0
    for epoch in range(1, comparison.LOCALIZER_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        weights = comparison.training_weights(model, x[fitting_tensor])
        residual_loss = comparison.localizer_loss(
            weights, y[fitting_tensor], y[fitting_tensor], "crps"
        )
        ess = 1.0 / torch.sum(weights**2, dim=1)
        penalty = torch.mean(torch.relu(comparison.MIN_ESS - ess) ** 2) / (
            comparison.MIN_ESS**2
        )
        (residual_loss + 0.05 * penalty).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_weights = torch.softmax(
                model.scores(x[validation_tensor], x[fitting_tensor]), dim=1
            )
            validation_loss = float(
                comparison.localizer_loss(
                    validation_weights,
                    y[validation_tensor],
                    y[fitting_tensor],
                    "crps",
                ).cpu()
            )
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= comparison.LOCALIZER_PATIENCE:
                break

    model = initialize_localizer(seed, device, encoder_state)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=comparison.LOCALIZER_LR,
        weight_decay=comparison.LOCALIZER_WEIGHT_DECAY,
    )
    for _ in range(best_epoch):
        model.train()
        optimizer.zero_grad()
        weights = comparison.training_weights(model, x)
        residual_loss = comparison.localizer_loss(weights, y, y, "crps")
        ess = 1.0 / torch.sum(weights**2, dim=1)
        penalty = torch.mean(torch.relu(comparison.MIN_ESS - ess) ** 2) / (
            comparison.MIN_ESS**2
        )
        (residual_loss + 0.05 * penalty).backward()
        optimizer.step()
    return model.eval(), {
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }


def metric_bundle(y, prediction, rows, weights, temperatures):
    result = {
        "point": comparison.previous.base.point_metrics(y, prediction),
        "methods": {},
    }
    target = rows["target_log_error"].to_numpy()
    baseline_mse = float(np.mean((target - rows["split_risk"].to_numpy()) ** 2))
    for method in METHODS:
        lower = rows[f"{method}_lower"].to_numpy()
        upper = rows[f"{method}_upper"].to_numpy()
        risk = rows[f"{method}_risk"].to_numpy()
        risk_mse = float(np.mean((target - risk) ** 2))
        interval = comparison.previous.base.interval_metrics(
            y, lower, upper, comparison.ALPHA
        )
        result["methods"][method] = {
            **interval,
            "miss_lower": int(np.sum(y < lower)),
            "miss_upper": int(np.sum(y > upper)),
            "risk_log_mse": risk_mse,
            "risk_mse_improvement_vs_uniform": (
                0.0 if method == "split" else 1.0 - risk_mse / baseline_mse
            ),
            "risk_spearman": (
                None
                if np.ptp(risk) < 1e-12
                else float(spearmanr(target, risk).statistic)
            ),
            **comparison.weight_diagnostics(weights[method], temperatures[method]),
        }
    return result


def run_seed(
    x,
    y,
    names,
    development,
    calibration,
    evaluation,
    matr_features,
    hust_features,
    seed,
    device,
    output,
):
    dev_oof, predictor_epochs = comparison.oof_development_predictions(
        x, y, development, seed, fold=0, device=device
    )
    predict_ids = np.concatenate([calibration, evaluation])
    external_prediction, epoch = comparison.robust.train_and_predict(
        x, y, development, predict_ids, seed, 32024, device
    )
    predictor_epochs.append({"role": "final", "epoch": epoch})
    cal_prediction = external_prediction[: len(calibration)]
    test_prediction = external_prediction[len(calibration) :]
    dev_log_error = np.log1p(np.abs(y[development] - dev_oof)).astype(np.float32)
    cal_error = np.abs(y[calibration] - cal_prediction)
    cal_log_error = np.log1p(cal_error)
    test_log_error = np.log1p(np.abs(y[evaluation] - test_prediction))

    median, scale = comparison.fit_robust_scaler(matr_features[development])
    features = comparison.apply_scaler(matr_features, median, scale)
    transformed_hust = comparison.apply_scaler(hust_features, median, scale)
    ssl_seed = seed * 1000 + 700
    matr_state, matr_ssl = pretrain_encoder(
        features[development], ssl_seed, device
    )
    combined_state, combined_ssl = pretrain_encoder(
        np.concatenate([features[development], transformed_hust], axis=0),
        ssl_seed,
        device,
    )
    localizer_seed = seed * 1000
    localizers, localizer_training = {}, {}
    for method, state in [
        ("hopfield_random", None),
        ("hopfield_ssl_matr", matr_state),
        ("hopfield_ssl_matr_hust", combined_state),
    ]:
        localizers[method], localizer_training[method] = fit_localizer(
            features[development],
            dev_log_error,
            localizer_seed,
            device,
            encoder_state=state,
        )

    rank = math.ceil((len(calibration) + 1) * (1 - comparison.ALPHA))
    level = min(1.0, rank / len(calibration))
    uniform = np.full(
        (len(evaluation), len(calibration)), 1.0 / len(calibration), dtype=float
    )
    weights = {"split": uniform}
    temperatures = {"split": np.ones(len(evaluation), dtype=float)}
    z, bandwidth = comparison.fixed_rbf_fit(features, development)
    distance2 = np.sum(
        (z[evaluation, None] - z[None, calibration]) ** 2, axis=2
    )
    weights["fixed_rbf"], temperatures["fixed_rbf"] = (
        comparison.weights_with_min_ess(-distance2 / (2 * bandwidth**2))
    )
    for method, localizer in localizers.items():
        logits = comparison.learned_logits(
            localizer, features[evaluation], features[calibration], device
        )
        weights[method], temperatures[method] = comparison.weights_with_min_ess(logits)

    rows = pd.DataFrame(
        {
            "cell_id": [names[index][:-4] for index in evaluation],
            "seed": seed,
            "eol": y[evaluation],
            "prediction": test_prediction,
            "target_log_error": test_log_error,
        }
    )
    for method in METHODS:
        risk = weights[method] @ cal_log_error
        q = comparison.weighted_quantiles(cal_error, weights[method], level)
        rows[f"{method}_risk"] = risk
        rows[f"{method}_q"] = q
        rows[f"{method}_lower"] = test_prediction - q
        rows[f"{method}_upper"] = test_prediction + q

    metrics = metric_bundle(y[evaluation], test_prediction, rows, weights, temperatures)
    metrics.update(
        {
            "seed": seed,
            "development_n": len(development),
            "calibration_n": len(calibration),
            "evaluation_n": len(evaluation),
            "conformal_rank": rank,
            "weighted_quantile_level": level,
            "fixed_rbf_bandwidth": bandwidth,
            "predictor_epochs": predictor_epochs,
            "ssl_training": {
                "matr_only": matr_ssl,
                "matr_hust": combined_ssl,
            },
            "localizer_training": localizer_training,
        }
    )
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    rows.to_csv(seed_dir / "evaluation.csv", index=False)
    comparison.save_json(seed_dir / "metrics.json", metrics)
    print(f"SEED {seed}: {json.dumps(metrics['methods'])}", flush=True)
    return metrics


def aggregate(metrics):
    result = {}
    for method in METHODS:
        values = [record["methods"][method] for record in metrics]
        correlations = [
            value["risk_spearman"]
            for value in values
            if value["risk_spearman"] is not None
        ]
        result[method] = {
            "coverage_mean": float(np.mean([value["coverage"] for value in values])),
            "mean_width_mean": float(np.mean([value["mean_width"] for value in values])),
            "median_width_mean": float(
                np.mean([value["median_width"] for value in values])
            ),
            "interval_score_mean": float(
                np.mean([value["mean_interval_score"] for value in values])
            ),
            "risk_improvement_mean": float(
                np.mean([value["risk_mse_improvement_vs_uniform"] for value in values])
            ),
            "risk_spearman_mean": (
                None if not correlations else float(np.mean(correlations))
            ),
            "seeds_with_positive_risk_improvement": int(
                sum(value["risk_mse_improvement_vs_uniform"] > 0 for value in values)
            ),
            "ess_mean": float(np.mean([value["ess_mean"] for value in values])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = HERE / "runs_transfer" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)

    split = comparison.previous.base.split_recorder
    matr_train = list(split.MATR_train_files)
    matr_calibration = list(split.MATR_val_files)
    matr_test = list(split.MATR_test_files)
    matr_names = matr_train + matr_calibration + matr_test
    hust_train = list(split.HUST_train_files)
    if len(matr_names) != len(set(matr_names)):
        raise ValueError("Official MATR train/validation/test splits overlap")

    config = {
        "experiment": "B: HUST-assisted self-supervised representation for MATR LCP",
        "T": 100,
        "target_dataset": "MATR",
        "auxiliary_dataset": "HUST train only",
        "target_split_sizes": {"train": 102, "validation": 34, "test": 33},
        "auxiliary_n": len(hust_train),
        "seeds": args.seeds,
        "alpha": comparison.ALPHA,
        "methods": METHODS,
        "embedding_dim": comparison.EMBED_DIM,
        "hidden_dim": comparison.HIDDEN_DIM,
        "min_ess": comparison.MIN_ESS,
        "localizer_loss": "crps",
        "ssl": {
            "objective": "masked denoising reconstruction of 57 time features",
            "epochs": SSL_EPOCHS,
            "mask_rate": SSL_MASK_RATE,
            "noise_std": SSL_NOISE_STD,
            "lr": SSL_LR,
            "weight_decay": SSL_WEIGHT_DECAY,
        },
        "predictor": vars(comparison.previous.CFG),
        "device": device,
        "data_root": str(args.data_root.resolve()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_usage": "exploratory; MATR test was viewed in earlier experiments",
        "weighted_interval_status": (
            "empirical basic-LCP diagnostic; no new local guarantee claimed"
        ),
    }
    comparison.save_json(output / "config.json", config)
    comparison.save_json(
        output / "split.json",
        {
            "matr_development": matr_train,
            "matr_calibration": matr_calibration,
            "matr_evaluation": matr_test,
            "hust_ssl_auxiliary": hust_train,
        },
    )
    print(f"RUN_DIR={output}", flush=True)

    x, y, matr_audit = comparison.previous.load_data(args.data_root, matr_names)
    hust_x, _, hust_audit = hust.load_hust_data(args.data_root, hust_train)
    comparison.save_json(
        output / "data_audit.json",
        {"MATR": matr_audit, "HUST_train": hust_audit},
    )
    matr_features = comparison.time_features(x)
    hust_features = comparison.time_features(hust_x)
    development = np.arange(0, len(matr_train))
    calibration = np.arange(len(matr_train), len(matr_train) + len(matr_calibration))
    evaluation = np.arange(len(matr_train) + len(matr_calibration), len(matr_names))

    metrics = []
    for seed in args.seeds:
        metrics.append(
            run_seed(
                x,
                y,
                matr_names,
                development,
                calibration,
                evaluation,
                matr_features,
                hust_features,
                seed,
                device,
                output,
            )
        )
    summary = aggregate(metrics)
    comparison.save_json(output / "summary.json", summary)
    print(f"SUMMARY: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
