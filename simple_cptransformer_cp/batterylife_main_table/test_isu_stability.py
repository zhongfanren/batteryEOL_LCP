"""Boundary checks for the repeated-split batch."""

import random
import unittest
from pathlib import Path

import numpy as np
import torch

import run_isu_stability as batch


class StabilityChecks(unittest.TestCase):
    def test_constant_local_features_recover_corrected_quantile(self):
        for h in (1.5, 3):
            radii, ess = batch.randomized_radius(np.zeros((61, 4)), np.zeros((8, 4)),
                                                np.arange(1, 62), h, 7)
            np.testing.assert_array_equal(radii, np.full(8, 56))
            np.testing.assert_allclose(ess, 61)

    def test_normalized_risk_does_not_have_one_cycle_floor(self):
        result = batch.kernel_scale(np.zeros((3, 1)), np.array([.1, .2, .3]),
                                     np.zeros((2, 1)), min_floor=1e-6)
        np.testing.assert_allclose(result, .2)

    def test_cache_hits_and_misses_do_not_change_training_rng(self):
        original = batch.training.load_dataset
        recorded = []
        def fake_loader(*args, **kwargs):
            recorded.append(args[4])
            return random.random(), np.random.random(), torch.rand(1)
        batch.training.load_dataset = fake_loader
        try:
            random.seed(9); np.random.seed(9); torch.manual_seed(9)
            batch.install_shared_cache(Path('test_cache'), {"split_seed": 77})
            a = batch.training.load_dataset(None, None, None, 'train', Path('original'))
            b = batch.training.load_dataset(None, None, None, 'train', Path('original'))
            self.assertEqual(a[0], b[0]); self.assertEqual(a[1], b[1])
            torch.testing.assert_close(a[2], b[2])
            self.assertEqual(recorded, [Path('test_cache/shared_cache/split_77')] * 2)
        finally:
            batch.training.load_dataset = original


if __name__ == '__main__':
    unittest.main()
