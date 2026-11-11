"""Check finite-sample/test-mass handling and label isolation in the pilot."""
import copy
import unittest

import numpy as np

from curated_cp_evaluation import weighted_radius
import run_isu_disagreement_cp as pilot


class DisagreementCPChecks(unittest.TestCase):
    def test_equal_weights_match_corrected_rank(self):
        scores = np.arange(1, 62, dtype=float)
        self.assertEqual(pilot.finite_quantile(scores), 56.)
        np.testing.assert_array_equal(pilot.weighted_quantiles(scores, np.ones((3, 62))), [56.] * 3)

    def test_vectorized_rlcp_matches_independent_weighted_quantile(self):
        rng = np.random.default_rng(28)
        cal, test = rng.normal(size=(61, 2)), rng.normal(size=(12, 2)) * 3
        normals = rng.normal(size=(12, 2))
        scores = rng.uniform(size=61)
        weights, _, mass = pilot.random_weights(cal, test, 3., normals)
        q = pilot.weighted_quantiles(scores, weights)
        expected = [weighted_radius(scores, w[:-1], w[-1]) for w in weights]
        np.testing.assert_array_equal(q, expected)
        self.assertTrue(np.isinf(q[mass > .1]).all())
        concentrated = np.r_[np.ones(61) / 61, 1.][None, :]
        self.assertTrue(np.isinf(pilot.weighted_quantiles(scores, concentrated)[0]))

    def test_test_labels_cannot_change_setup_or_intervals(self):
        # Use the actual frozen data and membership rather than a mock loader.
        plan = pilot.read(pilot.ROOT / "plan.json")
        specs = [s for s in plan["jobs"] if s["split_seed"] == 20261026]
        loaded = {s["key"]: dict(np.load(pilot.ROOT / "analysis" / s["key"] / "features.npz")) for s in specs}
        spec = next(s for s in specs if s["key"] == pilot.PRIMARY)
        scales = pilot.read(pilot.Path(spec["source"]) / spec["model"] / "results.json")["scales"]
        data = loaded[pilot.PRIMARY]
        altered = copy.deepcopy(data)
        altered["calibration_eol"] += 1e6
        altered["test_eol"] -= 1e6
        a, b = pilot.construct(data, loaded, scales), pilot.construct(altered, loaded, scales)
        for name in ("z", "life_scale", "gap_scale", "dev_scale", "gap", "relative_gap"):
            for phase in pilot.PHASES:
                np.testing.assert_array_equal(a[name][phase], b[name][phase])
        for name in ("prediction", "z", "life_scale", "gap_scale", "dev_scale"):
            a[f"pool_{name}"] = np.concatenate((a[name]["calibration"], a[name]["test"]))
        truth = np.r_[data["calibration_eol"], data["test_eol"]]
        changed_truth = truth.copy()
        changed_truth[61:] += 1e6
        args = a, truth, np.arange(61), np.arange(61, 122), 0
        radii, _ = pilot.intervals(*args)
        changed_radii, _ = pilot.intervals(a, changed_truth, args[2], args[3], 0)
        for method in radii:
            np.testing.assert_array_equal(radii[method], changed_radii[method])


if __name__ == "__main__":
    unittest.main()
