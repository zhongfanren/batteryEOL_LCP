# Curated ISU-ILCC exploratory result

T=100 test MAPE: 27.57%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.5950819672131148.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 88.52% | 1378.27 | 2224.83 | 0 |
| normalized_cp | 86.89% | 1223.66 | 2009.78 | 0 |
| base_lcp_protocol | 93.44% | 1535.33 | 1957.08 | 0 |
| rlcp_prediction | 88.52% | 1197.45 | 2313.26 | 0 |
| rlcp_protocol | 95.08% | inf | inf | 3 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
