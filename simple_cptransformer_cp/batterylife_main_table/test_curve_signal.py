"""Isolation and distribution contracts for the train/dev-only signal audit."""
import unittest
import numpy as np
import check_curve_signal as experiment


class SignalContracts(unittest.TestCase):
    def test_descriptor_shapes_and_known_constant_curves(self):
        x = np.ones((4, 100, 3, 300), dtype=np.float32)
        result = experiment.features(x)
        self.assertEqual(result["hand"].shape, (4, 9))
        self.assertEqual(result["curve"].shape, (4, 57))
        np.testing.assert_array_equal(result["hand"][:, :3], 1.)
        np.testing.assert_array_equal(result["hand"][:, 3:], 0.)
        np.testing.assert_array_equal(result["curve"][:, 48:], 0.)

    def test_feature_transforms_never_fit_query_covariates(self):
        rng = np.random.default_rng(5)
        pr, pq = rng.uniform(300, 1200, 35), rng.uniform(300, 1200, 8)
        lr, lq, _ = experiment.life_design(pr, pq)
        ref, query = rng.normal(size=(35, 9)), rng.normal(size=(8, 9))
        a = experiment.transform_extra(ref, query, lr, lq, "hand")
        b = experiment.transform_extra(ref, query+100, lr, lq+100, "hand")
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertEqual(a[-1], b[-1])

    def test_bandwidth_only_uses_reference_covariates(self):
        reference = np.arange(1, 10, dtype=float)[:, None]
        a, ha = experiment.kernel_weights(reference, np.array([[1.], [3.]]))
        b, hb = experiment.kernel_weights(reference, np.array([[1000.], [3000.]]))
        self.assertEqual(ha, hb)
        np.testing.assert_allclose(a.sum(axis=1), 1.)
        np.testing.assert_allclose(b.sum(axis=1), 1.)

    def test_weighted_quantile_first_crossing_and_ties(self):
        scores = np.array([200., 10., 50., 50.])
        weights = np.array([[.1, .2, .3, .4], [.5, .1, .2, .2]])
        np.testing.assert_array_equal(experiment.quantiles(scores, weights), [50., 200.])

    def test_estimation_requires_no_query_outcomes(self):
        rng = np.random.default_rng(7)
        pr, pq = rng.uniform(300, 1200, 40), rng.uniform(300, 1200, 7)
        truth = pr+rng.normal(0, 100, len(pr))
        ref = {"hand": rng.normal(size=(40, 9)), "curve": rng.normal(size=(40, 57)), "embedding": rng.normal(size=(40, 32))}
        query = {key: rng.normal(size=(7, array.shape[1])) for key, array in ref.items()}
        predictions, weights, scores, _ = experiment.estimate(pr, truth, pq, ref, query)
        self.assertEqual(set(predictions), set(experiment.METHODS))
        self.assertTrue(all(np.isfinite(x).all() and (x >= 0).all() for x in predictions.values()))
        for method, weight in weights.items():
            np.testing.assert_allclose(weight.sum(axis=1), 1.)
            for j, row in enumerate(weight):
                pairs = sorted(zip(scores, row))
                accumulated = 0.
                for value, mass in pairs:
                    accumulated += mass
                    if accumulated >= .9:
                        self.assertAlmostEqual(value, predictions[method][j])
                        break


if __name__ == "__main__":
    unittest.main()
