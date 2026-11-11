# Curated ISU-ILCC exploratory result

T=100 test MAPE: 26.76%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.6462189317821257.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 83.61% | 999.06 | 2358.77 | 0 |
| normalized_cp | 86.89% | 970.71 | 1971.16 | 0 |
| base_lcp_protocol | 95.08% | 1396.21 | 1751.31 | 0 |
| rlcp_prediction | 86.89% | 1112.00 | 2231.87 | 0 |
| rlcp_protocol | 90.16% | inf | inf | 4 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
