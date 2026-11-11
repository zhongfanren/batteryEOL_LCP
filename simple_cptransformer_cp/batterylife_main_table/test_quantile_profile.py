"""Scientific contracts for cross-fitted quantile-profile screening."""
import unittest
import numpy as np
import run_quantile_profile as experiment


class ProfileContracts(unittest.TestCase):
    @staticmethod
    def data():
        rng = np.random.default_rng(11)
        p = rng.uniform(300, 1800, 35)
        hand = rng.normal(size=(35, 9))
        target = np.abs(rng.normal(.15, .12, 35))
        fold = np.arange(35) % 5
        return p, hand, target, fold, rng.uniform(300, 1800, 6), rng.normal(size=(6, 9))

    def test_reference_profile_excludes_its_own_score(self):
        p, hand, target, fold, qp, qh = self.data()
        first = experiment.crossfit_profiles(p, target, hand, fold, qp, qh)
        target[0] += 100
        second = experiment.crossfit_profiles(p, target, hand, fold, qp, qh)
        for family in ("life", "hand"):
            np.testing.assert_array_equal(first[1][family][0], second[1][family][0])
            np.testing.assert_array_equal(first[2][family][0], second[2][family][0])
        for method in first[0]:
            np.testing.assert_array_equal(first[0][method][:, fold == fold[0]], second[0][method][:, fold == fold[0]])

    def test_constant_profiles_reduce_to_uniform_reference_weights(self):
        p, hand, target, fold, qp, qh = self.data()
        result = experiment.crossfit_profiles(p, np.full(len(p), .2), hand, fold, qp, qh)
        for weight in result[0].values():
            np.testing.assert_allclose(weight, 1/len(p), rtol=1e-10, atol=1e-12)

    def test_profiles_are_ordered_and_weights_preserve_block_mass(self):
        p, hand, target, fold, qp, qh = self.data()
        weights, ref, query, _, _ = experiment.crossfit_profiles(p, target, hand, fold, qp, qh)
        for profile in [*ref.values(), *query.values()]:
            self.assertTrue(np.isfinite(profile).all())
            self.assertTrue((profile >= 0).all())
            self.assertTrue((np.diff(profile, axis=-1) >= 0).all())
        for weight in weights.values():
            np.testing.assert_allclose(weight.sum(axis=1), 1.)
            for group in set(fold):
                np.testing.assert_allclose(weight[:, fold == group].sum(axis=1), np.mean(fold == group))

    def test_query_batch_does_not_change_reference_models(self):
        p, hand, target, fold, qp, qh = self.data()
        first = experiment.crossfit_profiles(p, target, hand, fold, qp, qh)
        second = experiment.crossfit_profiles(p, target, hand, fold, qp[:1], qh[:1])
        for method in first[0]:
            np.testing.assert_allclose(first[0][method][:1], second[0][method], rtol=1e-10, atol=1e-12)
        for family in ("life", "hand"):
            np.testing.assert_array_equal(first[1][family], second[1][family])


if __name__ == "__main__":
    unittest.main()
