"""Post-hoc residual audit for completed CPTransformer experiments.

This script is diagnostic only. Test labels summarized here must not be used to
fit or select a method that is subsequently evaluated on the same test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATE_COLUMNS = [
    "capacity_retention",
    "capacity_slope_10",
    "capacity_variability_10",
    "internal_resistance",
    "temperature_10",
]


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank().corr(y.rank()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--prefix", default="cptransformer_learned_kernels_seed")
    args = parser.parse_args()
    out_dir = args.outputs / "residual_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    misses: list[pd.DataFrame] = []
    calibration_tails: list[pd.DataFrame] = []
    for result_path in sorted(args.outputs.glob(f"{args.prefix}*/results.json")):
        run_dir = result_path.parent
        seed = int(run_dir.name.rsplit("seed", 1)[1])
        pred = pd.read_csv(run_dir / "predictions_all.csv")
        pred = pred[pred["split"].isin(["calibration", "test"])].copy()
        pred["seed"] = seed
        pred["abs_residual"] = pred["residual"].abs()
        pred["true_life"] = pred["true_rul"] + pred["t"]
        pred["relative_abs_error"] = pred["abs_residual"] / pred["true_rul"].clip(lower=1)
        frames.append(pred[[
            "seed", "split", "cell_id", "t", "true_life", "true_rul", "pred_rul",
            "residual", "abs_residual", "relative_abs_error", *STATE_COLUMNS,
        ]])

        for t in sorted(pred["t"].unique()):
            cal_t = pred[(pred["split"] == "calibration") & (pred["t"] == t)]
            calibration_tails.append(
                cal_t.nlargest(5, "abs_residual").assign(tail_rank=lambda x: np.arange(1, len(x) + 1))
            )
            lm = pd.read_csv(run_dir / f"landmark_{int(t):03d}.csv")
            for method in ["split_cp", "diagonal_state", "lowrank_joint", "quantile_profile", "spectral_band"]:
                covered = (lm["true_rul"] >= lm[f"{method}_lower"]) & (lm["true_rul"] <= lm[f"{method}_upper"])
                part = lm.loc[~covered, ["cell_id", "t", "true_rul", "pred_rul", "residual"]].copy()
                part["seed"] = seed
                part["method"] = method
                misses.append(part)

    data = pd.concat(frames, ignore_index=True)
    # Lifetime groups are fixed from unique calibration+test cells, not predictions.
    life = data[["cell_id", "true_life"]].drop_duplicates()
    life["life_group"] = pd.qcut(life["true_life"], 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])
    data = data.merge(life[["cell_id", "life_group"]], on="cell_id", how="left")

    group_summary = (
        data.groupby(["seed", "split", "t", "life_group"], observed=True, as_index=False)
        .agg(
            cells=("cell_id", "size"),
            true_life_mean=("true_life", "mean"),
            bias=("residual", "mean"),
            mae=("abs_residual", "mean"),
            rmse=("residual", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            relative_mae=("relative_abs_error", "mean"),
            max_abs_residual=("abs_residual", "max"),
        )
    )
    correlations: list[dict] = []
    for (seed, split, t), frame in data.groupby(["seed", "split", "t"]):
        row = {"seed": seed, "split": split, "t": t}
        for column in ["true_life", "true_rul", *STATE_COLUMNS]:
            row[f"spearman_absresid__{column}"] = spearman(frame["abs_residual"], frame[column])
        correlations.append(row)

    top_errors = (
        data.sort_values(["seed", "split", "t", "abs_residual"], ascending=[True, True, True, False])
        .groupby(["seed", "split", "t"], as_index=False)
        .head(5)
    )
    recurring = (
        data.groupby(["split", "cell_id"], as_index=False)
        .agg(
            observations=("abs_residual", "size"),
            mean_abs_residual=("abs_residual", "mean"),
            max_abs_residual=("abs_residual", "max"),
            mean_bias=("residual", "mean"),
            true_life=("true_life", "first"),
        )
        .sort_values("mean_abs_residual", ascending=False)
    )
    miss_df = pd.concat(misses, ignore_index=True)
    tail_df = pd.concat(calibration_tails, ignore_index=True)[[
        "seed", "t", "tail_rank", "cell_id", "true_life", "true_rul", "pred_rul", "residual", "abs_residual"
    ]]

    data.to_csv(out_dir / "residual_rows.csv", index=False)
    group_summary.to_csv(out_dir / "lifetime_group_summary.csv", index=False)
    pd.DataFrame(correlations).to_csv(out_dir / "residual_correlations.csv", index=False)
    top_errors.to_csv(out_dir / "top_errors.csv", index=False)
    recurring.to_csv(out_dir / "recurring_hard_cells.csv", index=False)
    tail_df.to_csv(out_dir / "calibration_tail.csv", index=False)
    miss_df.to_csv(out_dir / "test_misses.csv", index=False)

    compact = {
        "rows": int(len(data)),
        "unique_cells": int(data["cell_id"].nunique()),
        "seeds": sorted(map(int, data["seed"].unique())),
        "median_abs_residual": float(data["abs_residual"].median()),
        "q90_abs_residual": float(data["abs_residual"].quantile(0.9)),
        "q99_abs_residual": float(data["abs_residual"].quantile(0.99)),
        "longest_quartile_mean_abs_residual": float(
            data.loc[data["life_group"] == "Q4 longest", "abs_residual"].mean()
        ),
        "other_quartiles_mean_abs_residual": float(
            data.loc[data["life_group"] != "Q4 longest", "abs_residual"].mean()
        ),
        "note": "Post-hoc diagnosis only; do not tune on test labels.",
    }
    (out_dir / "summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(json.dumps(compact, indent=2))
    print("\nHardest recurring cells:")
    print(recurring.head(12).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
