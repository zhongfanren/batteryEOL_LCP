"""Compare Split CP after changing only the MATR T=100 point predictor."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "runs_external" / "20261012_144748_794767" / "predictions.csv"
ALPHA = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    rows = pd.read_csv(args.input)
    if (rows.split == "validation").sum() != 34 or (rows.split == "test").sum() != 33:
        raise ValueError("Expected 34 validation/calibration and 33 test cells")
    rows["cptransformer_mean_3"] = rows[[
        "seed_42", "seed_2021", "seed_2024"
    ]].mean(axis=1)
    rows["blend_3_50pct"] = .5 * (
        rows.cptransformer_mean_3 + rows.variance_one_feature
    )
    methods = (
        "cptransformer_mean_3", "cptransformer_mean_5",
        "variance_one_feature", "blend_3_50pct", "blend_50pct",
    )
    calibration = rows[rows.split == "validation"]
    evaluation = rows[rows.split == "test"].copy()
    rank = math.ceil((len(calibration) + 1) * (1 - ALPHA))
    metrics = {
        "protocol": "MATR official train predictor; validation calibration; test exploratory",
        "alpha": ALPHA,
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "rank": rank,
        "historical_test_previously_viewed": True,
        "methods": {},
    }
    y = evaluation.eol.to_numpy(dtype=float)
    for method in methods:
        cal_score = np.abs(calibration.eol.to_numpy() - calibration[method].to_numpy())
        radius = float(np.partition(cal_score, rank - 1)[rank - 1])
        pred = evaluation[method].to_numpy(dtype=float)
        lo, hi = pred - radius, pred + radius
        covered = (y >= lo) & (y <= hi)
        interval_score = (hi - lo) + (2 / ALPHA) * (
            np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
        )
        metrics["methods"][method] = {
            "point_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "point_mae": float(np.mean(np.abs(pred - y))),
            "calibration_q": radius,
            "coverage": float(np.mean(covered)),
            "mean_width": 2 * radius,
            "mean_interval_score": float(np.mean(interval_score)),
            "miss_lower": int(np.sum(y < lo)),
            "miss_upper": int(np.sum(y > hi)),
        }
        evaluation[f"{method}_lower"] = lo
        evaluation[f"{method}_upper"] = hi
    output = HERE / "runs_split_cp_centers" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    evaluation.to_csv(output / "test_intervals.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"RUN_DIR={output}")
    print(json.dumps(metrics["methods"], indent=2))


if __name__ == "__main__":
    main()
