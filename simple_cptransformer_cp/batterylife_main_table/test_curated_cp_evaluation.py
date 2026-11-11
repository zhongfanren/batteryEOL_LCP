"""Checks for the quantile correction and frozen-test label independence."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from curated_cp_evaluation import evaluate_frozen, local_radii, weighted_radius


class CuratedCPChecks(unittest.TestCase):
    def test_test_mass_and_split_rank(self):
        self.assertEqual(weighted_radius(np.arange(1, 62), np.ones(61), 1), 56)
        self.assertEqual(weighted_radius([1, 2], [1, 9], 1), 2)
        self.assertTrue(np.isinf(weighted_radius([1, 2], [1, 9], 2)))
        self.assertTrue(np.isinf(weighted_radius([1, 2], [1, 1], 1)))

    def test_constant_features_recover_split_cp(self):
        for randomized in (False, True):
            q, ess = local_radii(np.zeros((61, 3)), np.zeros((9, 3)),
                                 np.arange(1, 62), randomized)
            np.testing.assert_array_equal(q, np.full(9, 56))
            np.testing.assert_allclose(ess, 61)

    def test_test_labels_do_not_change_intervals_or_difficulty(self):
        manifest = json.loads((Path(__file__).parent / "protocol_candidates" /
                              "isu_ilcc_t100_candidate.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for phase in ("dev", "calibration", "test"):
                names = manifest["split"]["phases"][phase]
                prediction = rng.uniform(500, 1500, len(names))
                pd.DataFrame({"cell_file": names, "prefix_cycles": 100,
                              "prediction": prediction,
                              "eol": prediction + rng.normal(0, 200, len(names))}).to_csv(
                                  out / f"{phase}_predictions.csv", index=False)
            checkpoint = {"label_mean": 1000.0, "label_std": 300.0}
            first = evaluate_frozen(out, manifest, checkpoint)
            intervals_first = pd.read_csv(out / "test_intervals.csv")
            test = pd.read_csv(out / "test_predictions.csv")
            test.eol += 2000
            test.to_csv(out / "test_predictions.csv", index=False)
            second = evaluate_frozen(out, manifest, checkpoint)
            intervals_second = pd.read_csv(out / "test_intervals.csv")
            columns = [c for c in intervals_first if c.endswith(("_lower", "_upper", "_radius", "_ess"))]
            columns += ["predicted_difficulty", "difficulty_group"]
            pd.testing.assert_frame_equal(intervals_first[columns], intervals_second[columns])
            self.assertNotEqual(first["point"]["test"]["mape_fraction"], second["point"]["test"]["mape_fraction"])
            self.assertIsInstance(json.loads((out / "results.json").read_text(encoding="utf-8")), dict)


if __name__ == "__main__":
    unittest.main()
