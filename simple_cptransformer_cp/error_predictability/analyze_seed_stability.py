"""Measure how much CPTransformer seed instability contributes to T=100 error."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import run as previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    frames = []
    for seed in previous.SEEDS:
        frame = pd.read_csv(args.run_dir / f"seed_{seed}_predictions.csv")
        frame = frame[["cell_id", "outer_fold", "eol", "prediction"]].rename(
            columns={"prediction": f"prediction_{seed}"})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["cell_id", "outer_fold", "eol"], validate="one_to_one")

    columns = [f"prediction_{seed}" for seed in previous.SEEDS]
    values = merged[columns].to_numpy(float)
    y = merged.eol.to_numpy(float)
    merged["ensemble_mean"] = values.mean(axis=1)
    merged["ensemble_median"] = np.median(values, axis=1)
    merged["seed_std"] = values.std(axis=1, ddof=0)
    merged["ensemble_abs_error"] = np.abs(y - merged.ensemble_mean)

    individual = {str(seed): previous.base.point_metrics(y, merged[f"prediction_{seed}"].to_numpy())
                  for seed in previous.SEEDS}
    ensemble = {
        "mean": previous.base.point_metrics(y, merged.ensemble_mean.to_numpy()),
        "median": previous.base.point_metrics(y, merged.ensemble_median.to_numpy()),
    }
    per_seed_squared_error = np.mean((values - y[:, None]) ** 2)
    ensemble_squared_error = np.mean((values.mean(axis=1) - y) ** 2)
    seed_variance = np.mean(np.var(values, axis=1, ddof=0))
    assert np.isclose(per_seed_squared_error, ensemble_squared_error + seed_variance)
    pairwise = {}
    for left, right in [(42, 2021), (42, 2024), (2021, 2024)]:
        pairwise[f"{left}_{right}"] = float(spearmanr(
            merged[f"prediction_{left}"], merged[f"prediction_{right}"]).statistic)

    summary = {
        "n_cells": len(merged),
        "individual": individual,
        "ensemble": ensemble,
        "mean_individual_rmse": float(np.mean([v["rmse"] for v in individual.values()])),
        "mean_seed_std": float(merged.seed_std.mean()),
        "median_seed_std": float(merged.seed_std.median()),
        "p90_seed_std": float(merged.seed_std.quantile(0.9)),
        "seed_variance_fraction_of_average_mse": float(seed_variance / per_seed_squared_error),
        "seed_std_vs_ensemble_abs_error_spearman": float(spearmanr(
            merged.seed_std, merged.ensemble_abs_error).statistic),
        "pairwise_prediction_spearman": pairwise,
    }
    merged.sort_values("cell_id").to_csv(args.run_dir / "seed_stability_predictions.csv", index=False)
    with (args.run_dir / "seed_stability.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
