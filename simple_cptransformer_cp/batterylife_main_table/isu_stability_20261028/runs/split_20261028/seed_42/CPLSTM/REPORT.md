# Curated ISU-ILCC exploratory result

T=100 test MAPE: 23.73%; n=61.
Dev-fitted difficulty vs test |error| Spearman: -0.093918561607615.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 100.00% | 1705.25 | 1705.25 | 0 |
| normalized_cp | 98.36% | 1180.62 | 1218.33 | 0 |
| base_lcp_protocol | 100.00% | 1503.92 | 1503.92 | 0 |
| rlcp_prediction | 96.72% | 1137.31 | 1338.78 | 0 |
| rlcp_protocol | 95.08% | inf | inf | 2 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
