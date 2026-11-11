# Curated ISU-ILCC exploratory result

T=100 test MAPE: 28.33%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.496351136964569.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 90.16% | 1050.71 | 1953.23 | 0 |
| normalized_cp | 93.44% | 1037.94 | 1383.28 | 0 |
| base_lcp_protocol | 96.72% | inf | inf | 1 |
| rlcp_prediction | 93.44% | 1243.43 | 1591.28 | 0 |
| rlcp_protocol | 93.44% | inf | inf | 2 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
