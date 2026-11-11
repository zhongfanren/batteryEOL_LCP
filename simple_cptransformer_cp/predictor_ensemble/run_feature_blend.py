"""Honest MATR T=100 Qdlin feature baselines and heterogeneous ensemble.

Uses the same five OOF folds as run_seed_ensemble.py. Qdlin is a discharge
capacity curve on a common voltage grid in BatteryLife's processed cell data.
Only cycles at positions 10 and 100 are read; no later cycles enter features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_seed_ensemble import measures


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "runs" / "20261012_144205_310462" / "predictions.csv"
FEATURE_NAMES = (
    "log10_delta_q_variance",
    "log10_delta_q_abs_mean",
    "log10_delta_q_abs_min",
    "log10_delta_q_abs_max",
    "delta_q_mean",
    "capacity_at_cycle_10",
    "capacity_at_cycle_100",
    "capacity_fade_10_to_100",
)


def cell_features(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        cell = pickle.load(stream)
    cycles = cell["cycle_data"]
    if len(cycles) < 100:
        raise ValueError(f"Insufficient cycles: {path.name}")
    first, last = cycles[9], cycles[99]
    q10 = np.asarray(first["Qdlin"], dtype=float)
    q100 = np.asarray(last["Qdlin"], dtype=float)
    if q10.shape != (1000,) or q100.shape != (1000,):
        raise ValueError(f"Qdlin shape changed: {path.name}")
    delta = q100 - q10
    if not np.isfinite(delta).all():
        raise ValueError(f"Nonfinite Qdlin: {path.name}")
    abs_delta = np.abs(delta)
    capacity10 = float(np.nanmax(first["discharge_capacity_in_Ah"]))
    capacity100 = float(np.nanmax(last["discharge_capacity_in_Ah"]))
    values = np.array([
        np.log10(max(np.var(delta), 1e-12)),
        np.log10(max(np.mean(abs_delta), 1e-12)),
        np.log10(max(np.min(abs_delta), 1e-12)),
        np.log10(max(np.max(abs_delta), 1e-12)),
        np.mean(delta),
        capacity10,
        capacity100,
        capacity100 - capacity10,
    ], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite features: {path.name}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    rows = pd.read_csv(args.input)
    if len(rows) != 102 or rows.cell_id.nunique() != 102:
        raise ValueError("Expected 102 unique development cells")
    features = np.stack([
        cell_features(args.data_root / "MATR" / f"{name}.pkl")
        for name in rows.cell_id
    ])
    y = rows.eol.to_numpy(dtype=float)
    mean5 = rows.mean_5.to_numpy(dtype=float)
    variance_prediction = np.full(len(rows), np.nan)
    ridge_prediction = np.full(len(rows), np.nan)
    fold_info = []
    # The previous predictions are already stored in fixed five-fold order.
    # Training feature regressors uses labels only outside each held-out fold.
    for fold in range(5):
        train = np.flatnonzero(rows.outer_fold.to_numpy() != fold)
        evaluation = np.flatnonzero(rows.outer_fold.to_numpy() == fold)
        var_model = LinearRegression().fit(features[train, :1], np.log(y[train]))
        ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ridge_model.fit(features[train], np.log(y[train]))
        lower, upper = float(y[train].min()), float(y[train].max())
        variance_prediction[evaluation] = np.clip(
            np.exp(var_model.predict(features[evaluation, :1])), lower, upper
        )
        ridge_prediction[evaluation] = np.clip(
            np.exp(ridge_model.predict(features[evaluation])), lower, upper
        )
        fold_info.append({
            "fold": fold,
            "train_n": len(train),
            "eval_n": len(evaluation),
            "log_variance_coef": float(var_model.coef_[0]),
            "log_variance_intercept": float(var_model.intercept_),
        })
    candidates = {
        "cptransformer_mean_5": mean5,
        "variance_one_feature": variance_prediction,
        "ridge_8_features": ridge_prediction,
        "blend_75pct_cp_25pct_variance": .75 * mean5 + .25 * variance_prediction,
        "blend_50pct_cp_50pct_variance": .5 * (mean5 + variance_prediction),
        "blend_75pct_cp_25pct_ridge": .75 * mean5 + .25 * ridge_prediction,
        "blend_50pct_cp_50pct_ridge": .5 * (mean5 + ridge_prediction),
    }
    summary = {
        "protocol": "MATR train 102 only; five OOF folds; T=100; no validation/test",
        "feature_names": FEATURE_NAMES,
        "feature_source": "BatteryLife processed Qdlin from cycles 10 and 100",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "overall": {name: measures(y, pred) for name, pred in candidates.items()},
        "folds": {
            str(fold): {name: measures(y[rows.outer_fold == fold], pred[rows.outer_fold == fold])
                        for name, pred in candidates.items()}
            for fold in range(5)
        },
        "regressions": fold_info,
    }
    out = HERE / "runs_feature_blend" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    for i, name in enumerate(FEATURE_NAMES):
        rows[name] = features[:, i]
    for name, pred in candidates.items():
        rows[name] = pred
    rows.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"RUN_DIR={out}", flush=True)
    print(json.dumps(summary["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
