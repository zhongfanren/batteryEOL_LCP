"""Create tiny BatteryLife-shaped MATR files for an end-to-end smoke test."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def make_cycle(rng: np.random.Generator, cycle: int, life: int) -> dict:
    n = 48
    time = np.linspace(0, 3600, n)
    phase = np.linspace(0, 1, n)
    health = 1.0 - 0.2 * cycle / life
    current = np.where(phase < 0.5, 1.0, -1.0)
    voltage = 3.0 + 1.2 * phase + rng.normal(0, 0.005, n)
    charge = np.maximum(phase * 2.0 * health, 0)
    discharge = np.maximum((1.0 - phase) * 2.0 * health, 0)
    return {
        "cycle_number": cycle,
        "current_in_A": current,
        "voltage_in_V": voltage,
        "charge_capacity_in_Ah": charge,
        "discharge_capacity_in_Ah": discharge,
        "time_in_s": time,
        "temperature_in_C": np.full(n, 25.0) + rng.normal(0, 0.1, n),
        "internal_resistance_in_ohm": np.full(n, 0.03 + cycle / life * 0.01),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=40)
    args = parser.parse_args()

    matr_dir = args.output_dir / "MATR"
    label_dir = args.output_dir / "Life labels"
    matr_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    labels = {}
    for index in range(args.cells):
        filename = f"MATR_smoke_{index:03d}.pkl"
        life = int(500 + 18 * index + rng.integers(-20, 21))
        record = {
            "cell_id": filename.removesuffix(".pkl"),
            "cycle_data": [
                make_cycle(rng, cycle, life) for cycle in range(1, 51)
            ],
            "nominal_capacity_in_Ah": 2.0,
            "form_factor": "cylindrical",
            "anode_material": "graphite",
            "cathode_material": "LFP",
        }
        with (matr_dir / filename).open("wb") as handle:
            pickle.dump(record, handle)
        labels[filename] = life

    (label_dir / "MATR_labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8"
    )
    print(f"Created {len(labels)} smoke cells in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
