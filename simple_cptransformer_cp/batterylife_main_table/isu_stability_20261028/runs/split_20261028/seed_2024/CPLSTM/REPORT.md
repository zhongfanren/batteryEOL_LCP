# Curated ISU-ILCC exploratory result

T=100 test MAPE: 26.51%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.07625594923320993.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 96.72% | 1656.83 | 1734.98 | 0 |
| normalized_cp | 93.44% | 1152.58 | 1260.41 | 0 |
| base_lcp_protocol | 98.36% | 1556.86 | 1597.47 | 0 |
| rlcp_prediction | 93.44% | 1263.61 | 1540.81 | 0 |
| rlcp_protocol | 95.08% | inf | inf | 2 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
