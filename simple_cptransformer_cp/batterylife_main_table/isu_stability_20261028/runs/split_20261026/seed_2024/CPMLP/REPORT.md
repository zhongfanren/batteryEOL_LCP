# Curated ISU-ILCC exploratory result

T=100 test MAPE: 25.63%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.5104706504494976.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 93.44% | 1023.50 | 1854.46 | 0 |
| normalized_cp | 90.16% | 912.26 | 1309.18 | 0 |
| base_lcp_protocol | 98.36% | 1436.06 | 1548.20 | 0 |
| rlcp_prediction | 91.80% | 1162.77 | 1283.84 | 0 |
| rlcp_protocol | 93.44% | inf | inf | 2 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
