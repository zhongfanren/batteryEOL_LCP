"""Tests for scientific contracts of the score/locality experiment."""
import unittest
import numpy as np
import run_score_locality as experiment


class Contracts(unittest.TestCase):
    def test_quantile_contains_test_atom(self):
        scores = np.array([10., 50., 200.])
        weights = np.array([[.2, .7, .1, 0.], [.2, .4, .2, .2]])
        result = experiment.cp.weighted_quantiles(scores, weights)
        self.assertEqual(result[0], 50.)
        self.assertTrue(np.isinf(result[1]))

    def test_uniform_equals_finite_sample_order_statistic(self):
        for n in (4, 9, 34, 61):
            scores = np.linspace(.1, 100., n)[::-1]
            q = experiment.cp.weighted_quantiles(scores, np.ones((1, n+1)))[0]
            self.assertEqual(q, experiment.cp.finite_quantile(scores))

    def test_test_labels_cannot_change_any_radius(self):
        rng = np.random.default_rng(1)
        pred = rng.uniform(300, 1500, 122)
        truth = pred + rng.normal(0, 100, 122)
        ci, ti = np.arange(61), np.arange(61, 122)
        scales = np.array([np.ones(122), pred, pred+50])
        w, _ = experiment.get_weights(rng.normal(size=(122, 2)), ci, ti, 0)
        before = experiment.construct_intervals(pred, scales, truth, ci, ti, w)
        truth[ti] = 1e10
        after = experiment.construct_intervals(pred, scales, truth, ci, ti, w)
        for method in before:
            np.testing.assert_array_equal(before[method], after[method])

    def test_constant_scale_factor_cancels(self):
        rng = np.random.default_rng(9)
        pred = rng.uniform(300, 1500, 122)
        truth = pred+rng.normal(0, 100, 122)
        ci, ti = np.arange(61), np.arange(61, 122)
        scales = np.array([pred, pred*.3, pred*5.])
        w, _ = experiment.get_weights(rng.normal(size=(122, 2)), ci, ti, 0)
        result = experiment.construct_intervals(pred, scales, truth, ci, ti, w)
        for loc in experiment.LOCATIONS:
            np.testing.assert_allclose(result[f"absolute__{loc}"], result[f"relative__{loc}"])
            np.testing.assert_allclose(result[f"absolute__{loc}"], result[f"gap6__{loc}"])

    def test_geometry_does_not_read_outcomes_or_fit_on_test_covariates(self):
        data = {}
        for phase in experiment.cp.PHASES:
            data[f"{phase}_names"] = np.array([f"{phase}_{i}" for i in range(9)])
            data[f"{phase}_prediction"] = np.linspace(200, 1000, 9)
        cohort = {}
        for arch in ("CPMLP", "CPLSTM"):
            for seed in range(3):
                cohort[f"s0_r{seed}_{arch}"] = {k: v.copy() for k, v in data.items()}
                for phase in experiment.cp.PHASES:
                    cohort[f"s0_r{seed}_{arch}"][f"{phase}_prediction"] += (20 if arch == "CPLSTM" else 0) * (seed+1)
        # No *_eol keys exist: even dev outcomes are unavailable.
        first = experiment.prepare(data, cohort)
        data["test_prediction"] *= 100
        second = experiment.prepare(data, cohort)
        self.assertEqual(first["parameters"], second["parameters"])
        np.testing.assert_array_equal(first["z"]["train"], second["z"]["train"])


if __name__ == "__main__":
    unittest.main()
