"""Check query-independent feature fitting, a key leakage boundary in the audit."""
import copy
import unittest

import numpy as np
import pandas as pd

from audit_isu_risk_information import PHASES, feature_design, fit_predict


class RiskAuditIsolation(unittest.TestCase):
    def test_other_queries_do_not_change_preprocessing_or_reference_query(self):
        rng = np.random.default_rng(28)
        sizes = (12, 6, 6, 6)
        data, rows, metadata = {}, [], []
        offset = 0
        for phase, size in zip(PHASES, sizes):
            names = np.asarray([f"ISU-ILCC_G{i}C1.pkl" for i in range(offset, offset + size)])
            offset += size
            data[f"{phase}_names"] = names
            data[f"{phase}_prediction"] = rng.uniform(200, 1500, size)
            data[f"{phase}_embedding"] = rng.normal(size=(size, 8))
            for name in names:
                rows.append([name, *rng.normal(size=9)])
                metadata.append({"cell_file": name, "metadata": {
                    "charge_protocol": [{"rate_in_C": 1.}], "discharge_protocol": [{"rate_in_C": .5}]}})
        manifest = {"cells": metadata, "audit_model": "CPMLP"}
        curves = pd.DataFrame(rows).set_index(0)
        cohort = {}
        for model in ("CPMLP", "CPLSTM"):
            for seed in (2021, 42, 2024):
                item = copy.deepcopy(data)
                for phase in PHASES:
                    item[f"{phase}_prediction"] += rng.normal(0, 10, len(item[f"{phase}_names"]))
                cohort[f"s1_r{seed}_{model}"] = item
        original = feature_design(data, manifest, cohort, curves)
        altered, altered_cohort, altered_curves = copy.deepcopy(data), copy.deepcopy(cohort), curves.copy()
        for phase in PHASES[1:]:
            indices = np.arange(1, len(data[f"{phase}_names"])) if phase == "dev" else np.arange(len(data[f"{phase}_names"]))
            altered[f"{phase}_prediction"][indices] *= 100
            altered[f"{phase}_embedding"][indices] += 100
            altered_curves.loc[data[f"{phase}_names"][indices]] += 100
            for item in altered_cohort.values():
                item[f"{phase}_prediction"][indices] *= 100
        changed = feature_design(altered, manifest, altered_cohort, altered_curves)
        for family in original:
            np.testing.assert_allclose(original[family]["train"], changed[family]["train"], atol=1e-10)
            np.testing.assert_allclose(original[family]["dev"][0], changed[family]["dev"][0], atol=1e-10)

    def test_single_class_tail_fit_is_finite(self):
        x = np.arange(20)[:, None] / 20
        for error, expected in ((.1, 0.), (1., 1.)):
            risk, probability = fit_predict(x, np.full(20, error), np.array([[.2], [.8]]))
            self.assertTrue(np.isfinite(risk).all())
            np.testing.assert_array_equal(probability, [expected, expected])


if __name__ == "__main__":
    unittest.main()
