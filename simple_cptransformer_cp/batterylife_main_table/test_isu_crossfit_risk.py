"""Isolation and data-contract checks before starting the bounded GPU queue."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_isu_crossfit_risk as runner
from evaluate_isu_crossfit_risk import build_bundle, radii_from_calibration


class CrossfitRiskChecks(unittest.TestCase):
    def test_nested_partitions_exclude_outer_holdout_from_scaling_and_early_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = runner.prepare(Path(directory))
        for model in runner.MODELS:
            jobs = [j for j in plan["jobs"] if j["model"] == model]
            held = sum([j["outer_holdout"] for j in jobs], [])
            self.assertEqual(len(held), 97)
            self.assertEqual(len(set(held)), 97)
            for job in jobs:
                self.assertFalse(set(job["inner_fit"]) & set(job["inner_dev"]))
                self.assertEqual(set(job["inner_fit"] + job["inner_dev"]), set(job["outer_fit"]))
                self.assertFalse(set(job["outer_fit"]) & set(job["outer_holdout"]))

    def test_prefix_mask_and_label_scaling_exclude_heldout_label(self):
        bank = {"names": np.array(["fit", "held"]), "labels": np.array([300., 900.]),
                "curves": np.ones((2, 100, 3, 300), dtype=np.float32)}
        scale = runner.label_scale(bank, ["fit"])
        changed = copy.deepcopy(bank)
        changed["labels"][1] = 1e9
        self.assertEqual(scale, runner.label_scale(changed, ["fit"]))
        dataset = runner.PrefixDataset(bank, ["fit"], *scale)
        x, mask, y = runner.collate([dataset[0], dataset[99]])
        self.assertEqual(mask.sum().item(), 101)
        self.assertTrue(torch.all(x[0, 1:] == 0))
        self.assertTrue(torch.all(x[1] == 1))
        self.assertTrue(torch.all(y == 0))

    def test_query_labels_do_not_change_learned_scale_distance_or_original_intervals(self):
        rng = np.random.default_rng(28)
        truth = rng.uniform(300, 1400, 97)
        rows = []
        for model in runner.MODELS:
            for i in range(97):
                rows.append({"model": model, "cell_file": f"train{i}", "eol": truth[i],
                             "prediction": max(1., truth[i] + rng.normal(0, 100))})
        oof = pd.DataFrame(rows)
        cohort = {}
        for seed in (2021, 42, 2024):
            for model in runner.MODELS:
                values = {}
                for phase, n in (("dev", 20), ("calibration", 61), ("test", 61)):
                    values[f"{phase}_names"] = np.asarray([f"{phase}{i}" for i in range(n)])
                    values[f"{phase}_prediction"] = rng.uniform(200, 1800, n)
                    values[f"{phase}_eol"] = rng.uniform(200, 1800, n)
                cohort[f"s20261026_r{seed}_{model}"] = values
        # All forecasts refer to identical ordered names; labels only matter
        # for fixed dev20 and the synthetic OOF training labels.
        fixed = cohort["s20261026_r2024_CPMLP"]
        prior = {"prediction_mean": 900., "prediction_std": 500.}
        first = build_bundle(oof, fixed, cohort, prior, "CPMLP")
        altered = copy.deepcopy(fixed)
        altered["calibration_eol"] += 1e8
        altered["test_eol"] -= 1e8
        second = build_bundle(oof, altered, cohort, prior, "CPMLP")
        for phase in ("dev", "calibration", "test"):
            for key in first["scales"][phase]:
                np.testing.assert_array_equal(first["scales"][phase][key], second["scales"][phase][key])
            for key in first["coordinates"][phase]:
                np.testing.assert_array_equal(first["coordinates"][phase][key], second["coordinates"][phase][key])
        scales = {k: np.r_[first["scales"]["calibration"][k], first["scales"]["test"][k]] for k in first["scales"]["calibration"]}
        coords = {k: np.r_[first["coordinates"]["calibration"][k], first["coordinates"]["test"][k]] for k in first["coordinates"]["calibration"]}
        pred = np.r_[fixed["calibration_prediction"], fixed["test_prediction"]]
        y = np.r_[fixed["calibration_eol"], fixed["test_eol"]]
        changed_y = y.copy(); changed_y[61:] += 1e8
        a, _ = radii_from_calibration(scales, coords, pred, y, np.arange(61), np.arange(61, 122), 0)
        b, _ = radii_from_calibration(scales, coords, pred, changed_y, np.arange(61), np.arange(61, 122), 0)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
