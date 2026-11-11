"""Extend the existing honest MATR T=100 five-fold ensemble from 3 to 5 seeds.

The existing three seed predictions are read from the fixed, development-only
experiment. Only two new CPTransformer seeds are trained, with identical outer
folds, inner epoch-selection split and architecture. No official validation or
test cells are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "error_predictability"
sys.path.insert(0, str(EXPERIMENTS))
import run as experiment  # noqa: E402
import run_crossfit_robust as robust  # noqa: E402


OLD_SEEDS = (42, 2021, 2024)
NEW_SEEDS = (7, 1024)
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
DEFAULT_PRIOR = (
    EXPERIMENTS / "runs_ensemble_disagreement" / "20260909_214700_006033"
    / "predictions.csv"
)


def measures(y: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - y
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / y)),
        "r2": float(1 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def validate_prior(prior: pd.DataFrame, names: list[str], y: np.ndarray) -> None:
    expected = {name.removesuffix(".pkl") for name in names}
    if len(prior) != len(names) or set(prior.cell_id) != expected:
        raise ValueError("Prior OOF predictions do not cover exactly the MATR train cells")
    if prior.cell_id.duplicated().any():
        raise ValueError("Duplicate prior OOF cell")
    lookup = {name.removesuffix(".pkl"): i for i, name in enumerate(names)}
    for row in prior.itertuples():
        if not np.isclose(row.eol, y[lookup[row.cell_id]]):
            raise ValueError(f"Label mismatch for {row.cell_id}")
        three_mean = np.mean([getattr(row, f"prediction_{s}") for s in OLD_SEEDS])
        if not np.isclose(row.prediction, three_mean, atol=1e-5):
            raise ValueError(f"Prior ensemble mismatch for {row.cell_id}")
    name_to_index = {name.removesuffix(".pkl"): i for i, name in enumerate(names)}
    folds = KFold(5, shuffle=True, random_state=2024)
    for fold, (_, eval_ids) in enumerate(folds.split(names)):
        expected_fold = {names[i].removesuffix(".pkl") for i in eval_ids}
        observed_fold = set(prior.loc[prior.outer_fold == fold, "cell_id"])
        if expected_fold != observed_fold:
            raise ValueError(f"Prior outer fold {fold} does not match the fixed split")


def summarize(rows: pd.DataFrame) -> dict:
    y = rows.eol.to_numpy(dtype=float)
    seed_predictions = np.column_stack(
        [rows[f"prediction_{seed}"].to_numpy(dtype=float) for seed in ALL_SEEDS]
    )
    first_three = seed_predictions[:, :3]
    candidates = {
        "dummy_train_mean": rows.dummy_train_mean.to_numpy(dtype=float),
        **{f"seed_{seed}": seed_predictions[:, i] for i, seed in enumerate(ALL_SEEDS)},
        "mean_3": first_three.mean(axis=1),
        "median_3": np.median(first_three, axis=1),
        "mean_5": seed_predictions.mean(axis=1),
        "median_5": np.median(seed_predictions, axis=1),
        "trimmed_mean_5": np.sort(seed_predictions, axis=1)[:, 1:-1].mean(axis=1),
    }
    for method, prediction in candidates.items():
        rows[method] = prediction
    results = {name: measures(y, prediction) for name, prediction in candidates.items()}
    fold_results = {}
    for fold in range(5):
        subset = rows.outer_fold.to_numpy() == fold
        fold_results[str(fold)] = {
            name: measures(y[subset], pred[subset]) for name, pred in candidates.items()
        }
    error = seed_predictions - y[:, None]
    mean_error = error.mean(axis=1)
    noise = error - mean_error[:, None]
    results["diagnostics"] = {
        "seed_std_mean": float(np.mean(seed_predictions.std(axis=1))),
        "seed_std_median": float(np.median(seed_predictions.std(axis=1))),
        "seed_variance_fraction_of_mean_single_mse": float(
            np.mean(noise ** 2) / np.mean(error ** 2)
        ),
        "mean_5_vs_mean_3_squared_error_delta": float(
            np.mean((candidates["mean_5"] - y) ** 2)
            - np.mean((candidates["mean_3"] - y) ** 2)
        ),
    }
    return {"overall": results, "folds": fold_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This experiment expects the available GPU")

    prior = pd.read_csv(args.prior)
    names = list(experiment.base.split_recorder.MATR_train_files)
    x, y, audit = experiment.load_data(args.data_root, names)
    validate_prior(prior, names, y)
    config = {
        "task": "MATR T=100 development-only extension of existing three-seed OOF ensemble",
        "old_seeds": OLD_SEEDS,
        "new_seeds": NEW_SEEDS,
        "split_seed": 2024,
        "outer_folds": 5,
        "predictor": vars(experiment.CFG),
        "prior": str(args.prior.resolve()),
        "prior_sha256": hashlib.sha256(args.prior.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": device,
        "official_validation_or_test_used": False,
    }
    if args.resume:
        output = args.resume.resolve()
        old_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if old_config != json.loads(json.dumps(config)):
            raise ValueError("Resume configuration mismatch")
    else:
        output = HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        experiment.save_json(output / "config.json", config)
        experiment.save_json(output / "data_audit.json", audit)
    print(f"RUN_DIR={output}", flush=True)

    rows = prior.copy()
    folds = KFold(5, shuffle=True, random_state=2024)
    for fold, (train_ids, eval_ids) in enumerate(folds.split(names)):
        fold_path = output / f"fold_{fold}.json"
        if fold_path.exists():
            record = json.loads(fold_path.read_text(encoding="utf-8"))
        else:
            record = {
                "fold": fold,
                "train_cell_ids": [names[i].removesuffix(".pkl") for i in train_ids],
                "eval_cell_ids": [names[i].removesuffix(".pkl") for i in eval_ids],
                "dummy_train_mean": float(y[train_ids].mean()),
                "new_seed_predictions": {},
                "selected_epochs": {},
            }
        if record["eval_cell_ids"] != [names[i].removesuffix(".pkl") for i in eval_ids]:
            raise ValueError(f"Resume fold {fold} mismatch")
        for seed in NEW_SEEDS:
            if str(seed) in record["new_seed_predictions"]:
                continue
            prediction, epoch = robust.train_and_predict(
                x, y, train_ids, eval_ids, seed, 32024 + fold, device
            )
            record["new_seed_predictions"][str(seed)] = prediction.tolist()
            record["selected_epochs"][str(seed)] = int(epoch)
            experiment.save_json(fold_path, record)
            print(f"fold {fold + 1}/5 seed {seed}: epoch {epoch}", flush=True)
        for cell_id in record["eval_cell_ids"]:
            rows.loc[rows.cell_id == cell_id, "dummy_train_mean"] = record["dummy_train_mean"]
        for seed in NEW_SEEDS:
            for cell_id, prediction in zip(
                record["eval_cell_ids"], record["new_seed_predictions"][str(seed)]
            ):
                rows.loc[rows.cell_id == cell_id, f"prediction_{seed}"] = prediction

    if rows[[f"prediction_{seed}" for seed in ALL_SEEDS]].isna().any().any():
        raise RuntimeError("Missing OOF predictions")
    results = summarize(rows)
    rows.to_csv(output / "predictions.csv", index=False)
    experiment.save_json(output / "metrics.json", results)
    overall = results["overall"]
    print("RESULTS=" + json.dumps({k: overall[k] for k in
        ["dummy_train_mean", "mean_3", "median_3", "mean_5", "median_5", "trimmed_mean_5"]}), flush=True)


if __name__ == "__main__":
    main()
