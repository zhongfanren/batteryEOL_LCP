# Curated ISU-ILCC exploratory result

T=100 test MAPE: 25.08%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.13595980962453727.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 98.36% | 1211.47 | 1283.97 | 0 |
| normalized_cp | 91.80% | 971.39 | 1119.83 | 0 |
| base_lcp_protocol | 100.00% | 1460.23 | 1460.23 | 0 |
| rlcp_prediction | 96.72% | 1110.37 | 1296.25 | 0 |
| rlcp_protocol | 91.80% | inf | inf | 2 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
