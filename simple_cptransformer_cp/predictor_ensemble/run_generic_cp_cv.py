"""MATR train-only Split CP using generic voltage-grid ΔQ(V) variance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from run_generic_dq import generic_feature
from run_heterogeneous_cp_cv import ALPHA, interval_metrics
from run_seed_ensemble import OLD_SEEDS, measures


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "error_predictability"
LOCALIZERS = HERE.parent / "learnable_localizers"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(LOCALIZERS))
import run as experiment  # noqa: E402
import run_ensemble_disagreement as ensemble  # noqa: E402
import run_grouped_metric as grouped  # noqa: E402


METHODS = ("cptransformer_mean_3", "generic_variance", "blend_3_50pct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\BatteryLife_Processed"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This experiment expects the available GPU")
    names = list(experiment.base.split_recorder.MATR_train_files)
    x, y, audit = experiment.load_data(args.data_root, names)
    feature = np.array([
        generic_feature(args.data_root / "MATR" / name) for name in names
    ])[:, None]
    output = HERE / "runs_generic_cp_cv" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    experiment.save_json(output / "config.json", {
        "task": "development-only T=100 generic variance and CPTransformer Split CP",
        "outer_folds": 5, "outer_split_seed": 2024,
        "calibration_per_fold": 20, "predictor_seeds": OLD_SEEDS,
        "predictor": vars(experiment.CFG), "alpha": ALPHA,
        "feature": "log10 variance of normalized Qd100(V)-Qd10(V) on 2.05-3.35 V grid",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_validation_or_test_used": False,
        "device": device,
    })
    experiment.save_json(output / "data_audit.json", audit)
    print(f"RUN_DIR={output}", flush=True)
    all_rows, folds = [], {}
    outer = KFold(5, shuffle=True, random_state=2024)
    for fold, (outer_train, outer_eval) in enumerate(outer.split(names)):
        development, calibration, evaluation = grouped.fold_partition(
            len(names), fold, outer_train, outer_eval, calibration_size=20
        )
        to_predict = np.concatenate([calibration, evaluation])
        matrix, epochs = ensemble.train_ensemble(
            x, y, development, to_predict, 32024 + fold, device
        )
        cp_prediction = matrix.mean(axis=1)
        feature_model = LinearRegression().fit(
            feature[development], np.log(y[development])
        )
        generic_prediction = np.clip(
            np.exp(feature_model.predict(feature[to_predict])),
            float(y[development].min()), float(y[development].max())
        )
        predictions = {
            "cptransformer_mean_3": cp_prediction,
            "generic_variance": generic_prediction,
            "blend_3_50pct": .5 * (cp_prediction + generic_prediction),
        }
        rank = math.ceil((len(calibration) + 1) * (1 - ALPHA))
        fold_rows = pd.DataFrame({
            "cell_id": [names[i].removesuffix(".pkl") for i in evaluation],
            "outer_fold": fold, "eol": y[evaluation],
        })
        for method, values in predictions.items():
            scores = np.abs(y[calibration] - values[:len(calibration)])
            q = float(np.partition(scores, rank - 1)[rank - 1])
            fold_rows[f"{method}_prediction"] = values[len(calibration):]
            fold_rows[f"{method}_q"] = q
        all_rows.append(fold_rows)
        folds[str(fold)] = {
            "development_n": len(development),
            "calibration_n": len(calibration),
            "evaluation_n": len(evaluation),
            "rank": rank,
            "epochs": epochs,
            "feature_coefficient": float(feature_model.coef_[0]),
            "point": {method: measures(
                y[evaluation], fold_rows[f"{method}_prediction"].to_numpy()
            ) for method in METHODS},
            "interval": {method: interval_metrics(
                y[evaluation], fold_rows[f"{method}_prediction"].to_numpy(),
                fold_rows[f"{method}_q"].to_numpy()
            ) for method in METHODS},
        }
        print(f"fold {fold + 1}/5 complete", flush=True)
    rows = pd.concat(all_rows, ignore_index=True)
    if len(rows) != 102 or rows.cell_id.nunique() != 102:
        raise ValueError("Outer folds must evaluate each train cell exactly once")
    rows.to_csv(output / "predictions.csv", index=False)
    result = {
        "folds": folds,
        "overall": {
            method: {
                "point": measures(rows.eol.to_numpy(), rows[f"{method}_prediction"].to_numpy()),
                "interval": interval_metrics(
                    rows.eol.to_numpy(), rows[f"{method}_prediction"].to_numpy(),
                    rows[f"{method}_q"].to_numpy()
                ),
            } for method in METHODS
        },
    }
    experiment.save_json(output / "metrics.json", result)
    print("RESULTS=" + json.dumps(result["overall"]), flush=True)


if __name__ == "__main__":
    main()
