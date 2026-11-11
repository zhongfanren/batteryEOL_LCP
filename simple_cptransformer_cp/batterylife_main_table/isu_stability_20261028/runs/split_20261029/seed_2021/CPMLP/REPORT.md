# Curated ISU-ILCC exploratory result

T=100 test MAPE: 24.31%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.670809095716552.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 77.05% | 1005.92 | 2036.93 | 0 |
| normalized_cp | 83.61% | 889.26 | 1703.01 | 0 |
| base_lcp_protocol | 90.16% | 1220.59 | 1621.08 | 0 |
| rlcp_prediction | 78.69% | 990.03 | 2110.40 | 0 |
| rlcp_protocol | 86.89% | inf | inf | 2 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
