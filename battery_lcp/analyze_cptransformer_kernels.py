"""Aggregate completed CPTransformer learned-kernel experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--prefix", default="cptransformer_learned_kernels_seed")
    args = parser.parse_args()

    rows: list[dict] = []
    point_rows: list[dict] = []
    for path in sorted(args.outputs.glob(f"{args.prefix}*/results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        seed = int(path.parent.name.rsplit("seed", 1)[1])
        for landmark, metrics in result["point"].items():
            point_rows.append({"seed": seed, "landmark": int(landmark), **metrics})
        methods = {"split_cp": result["split_cp"], **result["learned_kernels"]}
        for method, landmarks in methods.items():
            for landmark, metrics in landmarks.items():
                rows.append(
                    {
                        "seed": seed,
                        "landmark": int(landmark),
                        "method": method,
                        "coverage": metrics["coverage"],
                        "mean_width": metrics["mean_width"],
                        "covered": metrics["covered"],
                        "total": metrics["total"],
                        "fallback_fraction": metrics.get("fallback_fraction", 0.0),
                        "ess_mean": metrics.get("ess_mean", np.nan),
                    }
                )

    if not rows:
        raise SystemExit(f"No completed runs matching {args.prefix!r} in {args.outputs}")

    raw = pd.DataFrame(rows).sort_values(["landmark", "method", "seed"])
    points = pd.DataFrame(point_rows).sort_values(["landmark", "seed"])
    summary = (
        raw.groupby(["landmark", "method"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            width_mean=("mean_width", "mean"),
            width_std=("mean_width", "std"),
            target_met_runs=("coverage", lambda x: int((x >= 0.9).sum())),
            runs=("coverage", "size"),
            fallback_mean=("fallback_fraction", "mean"),
            ess_mean=("ess_mean", "mean"),
        )
        .sort_values(["landmark", "method"])
    )
    point_summary = (
        points.groupby("landmark", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mape_mean=("mape", "mean"),
            r2_mean=("r2", "mean"),
        )
        .sort_values("landmark")
    )

    raw.to_csv(args.outputs / "cptransformer_kernel_runs.csv", index=False)
    summary.to_csv(args.outputs / "cptransformer_kernel_summary.csv", index=False)
    points.to_csv(args.outputs / "cptransformer_point_runs.csv", index=False)
    point_summary.to_csv(args.outputs / "cptransformer_point_summary.csv", index=False)

    print("Point prediction (mean +/- sample std across seeds):")
    print(point_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nConformal methods:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
