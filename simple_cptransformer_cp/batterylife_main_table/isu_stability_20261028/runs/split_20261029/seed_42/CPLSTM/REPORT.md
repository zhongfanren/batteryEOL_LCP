# Curated ISU-ILCC exploratory result

T=100 test MAPE: 24.60%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.5958751983077737.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 86.89% | 1227.38 | 2275.97 | 0 |
| normalized_cp | 86.89% | 1133.14 | 2139.18 | 0 |
| base_lcp_protocol | 98.36% | 1748.95 | 1932.90 | 0 |
| rlcp_prediction | 85.25% | 1238.34 | 2294.56 | 0 |
| rlcp_protocol | 91.80% | inf | inf | 3 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
