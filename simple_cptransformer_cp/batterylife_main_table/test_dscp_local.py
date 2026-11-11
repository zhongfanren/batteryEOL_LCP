"""New DSCP pilot contracts; existing quantile/group helpers stay frozen."""
import unittest
import numpy as np
import torch
import run_dscp_local as run


class DSCPContracts(unittest.TestCase):
    def test_sequence_encoder_heads_ordered_and_trainable(self):
        torch.manual_seed(11)
        model = run.SequenceLocalizer()
        q, hidden = model(torch.randn(3, 100, 900))
        self.assertEqual(tuple(hidden.shape), (3, 32))
        self.assertTrue(torch.all(q[:, 1:] >= q[:, :-1]))
        run.pinball(q, torch.tensor([-.3, .5, 1.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_weighted_endpoints_and_test_infinity_atom(self):
        score = np.arange(-30., 31.)
        finite = np.ones((1, 61))/61
        full = np.ones((1, 62))
        self.assertEqual(run.weighted_q(score, finite, .05)[0], -27.)
        self.assertEqual(run.weighted_q(score, finite, .95)[0], 27.)
        self.assertEqual(run.weighted_q(score, full, .95, True)[0], 28.)
        weights = np.r_[np.ones(61)*.8/61, .2][None, :]
        self.assertTrue(np.isinf(run.weighted_q(score, weights, .9, True)[0]))

    def test_input_scaling_uses_only_training_cells(self):
        rng = np.random.default_rng(1)
        train, test = rng.normal(size=(4, 100, 900)), rng.normal(size=(3, 100, 900))
        first, m1, s1 = run.standardize(train, {"train": train, "test": test})
        second, m2, s2 = run.standardize(train, {"train": train, "test": test+1e6})
        np.testing.assert_array_equal(m1, m2); np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(first["train"], second["train"])

    def test_calibration_test_outcomes_do_not_enter_kernel_weights(self):
        rng = np.random.default_rng(2)
        train, cal, test = rng.normal(size=(97, 32)), rng.normal(size=(61, 32)), rng.normal(size=(61, 32))
        tau = run.temperature(train)
        w = run.soft_weights(cal, test, tau)
        np.testing.assert_allclose(w.sum(axis=1), 1.)
        np.testing.assert_array_equal(w, run.soft_weights(cal, test, tau))
        self.assertTrue(np.isfinite(w).all())

    def test_query_labels_do_not_change_any_interval(self):
        rng = np.random.default_rng(9)
        pred = rng.uniform(300, 1500, 122)
        truth = pred+rng.normal(0, 100, 122)
        bundle = {"pred": {"calibration": pred[:61], "test": pred[61:]},
                  "data": {"calibration_eol": truth[:61].copy(), "test_eol": truth[61:].copy()},
                  "pool_gap": np.zeros(122), "hand_pool": rng.normal(size=(122, 4)), "hand_h": 2.}
        learned, random = rng.normal(size=(122, 32)), rng.normal(size=(122, 32))
        info = {"train_temperature": 30., "random_temperature": 30.}
        ci, ti = np.arange(61), np.arange(61, 122)
        for mode in run.MODES:
            first = run.bounds(bundle, mode, learned, random, info, ci, ti, 0)[0]
            bundle["data"]["test_eol"] += 1e6
            second = run.bounds(bundle, mode, learned, random, info, ci, ti, 0)[0]
            for method in first:
                np.testing.assert_array_equal(first[method], second[method])


if __name__ == "__main__":
    unittest.main()
