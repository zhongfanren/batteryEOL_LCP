"""Experiment C: fixed RBF localizer with controlled representations.

All methods share the same MATR point predictor, calibration residuals, RBF
formula, bandwidth rule, and ESS safeguard.  Only the frozen representation
used to compute distances changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import run_comparison as comparison
import run_hust_replication as hust
import run_matr_hust_transfer as transfer


HERE = Path(__file__).resolve().parent
SEEDS = [42, 2021, 2024]
METHODS = [
    "split",
    "rbf_handcrafted_pca",
    "rbf_random_encoder",
    "rbf_ssl_matr",
    "rbf_ssl_matr_hust",
]


def encode(model, features, device):
    model.eval()
    with torch.no_grad():
        values = torch.tensor(features, dtype=torch.float32, device=device)
        return model.encode(values).cpu().numpy()


def rbf_weights(embedding, development, calibration, evaluation):
    development_z = embedding[development]
    pairwise = np.sqrt(
        np.sum((development_z[:, None] - development_z[None, :]) ** 2, axis=2)
    )
    upper = pairwise[np.triu_indices(len(development), 1)]
    bandwidth = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    bandwidth = max(bandwidth, 1e-8)
    distance2 = np.sum(
        (embedding[evaluation, None] - embedding[None, calibration]) ** 2, axis=2
    )
    weights, temperatures = comparison.weights_with_min_ess(
        -distance2 / (2 * bandwidth**2)
    )
    return weights, temperatures, bandwidth


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
    predict_ids = np.concatenate([calibration, evaluation])
    external_prediction, epoch = comparison.robust.train_and_predict(
        x, y, development, predict_ids, seed, 32024, device
    )
    cal_prediction = external_prediction[: len(calibration)]
    test_prediction = external_prediction[len(calibration) :]
    cal_error = np.abs(y[calibration] - cal_prediction)
    cal_log_error = np.log1p(cal_error)
    test_log_error = np.log1p(np.abs(y[evaluation] - test_prediction))

    median, scale = comparison.fit_robust_scaler(matr_features[development])
    features = comparison.apply_scaler(matr_features, median, scale)
    transformed_hust = comparison.apply_scaler(hust_features, median, scale)
    ssl_seed = seed * 1000 + 700
    matr_state, matr_ssl = transfer.pretrain_encoder(
        features[development], ssl_seed, device
    )
    combined_state, combined_ssl = transfer.pretrain_encoder(
        np.concatenate([features[development], transformed_hust], axis=0),
        ssl_seed,
        device,
    )
    random_model = transfer.initialize_localizer(ssl_seed, device)
    matr_model = transfer.initialize_localizer(ssl_seed, device, matr_state)
    combined_model = transfer.initialize_localizer(ssl_seed, device, combined_state)

    representations = {
        "rbf_random_encoder": encode(random_model, features, device),
        "rbf_ssl_matr": encode(matr_model, features, device),
        "rbf_ssl_matr_hust": encode(combined_model, features, device),
    }
    handcrafted, handcrafted_bandwidth = comparison.fixed_rbf_fit(
        features, development
    )

    uniform = np.full(
        (len(evaluation), len(calibration)), 1.0 / len(calibration), dtype=float
    )
    weights = {"split": uniform}
    temperatures = {"split": np.ones(len(evaluation), dtype=float)}
    bandwidths = {"rbf_handcrafted_pca": handcrafted_bandwidth}
    distance2 = np.sum(
        (handcrafted[evaluation, None] - handcrafted[None, calibration]) ** 2,
        axis=2,
    )
    weights["rbf_handcrafted_pca"], temperatures["rbf_handcrafted_pca"] = (
        comparison.weights_with_min_ess(
            -distance2 / (2 * handcrafted_bandwidth**2)
        )
    )
    for method, embedding in representations.items():
        weights[method], temperatures[method], bandwidths[method] = rbf_weights(
            embedding, development, calibration, evaluation
        )

    rank = math.ceil((len(calibration) + 1) * (1 - comparison.ALPHA))
    level = min(1.0, rank / len(calibration))
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
            "point_epoch": epoch,
            "bandwidths": bandwidths,
            "ssl_training": {
                "matr_only": matr_ssl,
                "matr_hust": combined_ssl,
            },
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
    output = HERE / "runs_representation" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)

    split = comparison.previous.base.split_recorder
    matr_train = list(split.MATR_train_files)
    matr_calibration = list(split.MATR_val_files)
    matr_test = list(split.MATR_test_files)
    matr_names = matr_train + matr_calibration + matr_test
    hust_train = list(split.HUST_train_files)

    config = {
        "experiment": "C: frozen representation comparison under fixed RBF",
        "T": 100,
        "target_dataset": "MATR",
        "auxiliary_dataset": "HUST train only",
        "target_split_sizes": {"train": 102, "validation": 34, "test": 33},
        "auxiliary_n": len(hust_train),
        "seeds": args.seeds,
        "alpha": comparison.ALPHA,
        "methods": METHODS,
        "kernel": "RBF with median development-distance bandwidth",
        "min_ess": comparison.MIN_ESS,
        "ssl": {
            "objective": "masked denoising reconstruction of 57 time features",
            "epochs": transfer.SSL_EPOCHS,
            "mask_rate": transfer.SSL_MASK_RATE,
            "noise_std": transfer.SSL_NOISE_STD,
            "lr": transfer.SSL_LR,
            "weight_decay": transfer.SSL_WEIGHT_DECAY,
            "frozen_for_rbf": True,
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
