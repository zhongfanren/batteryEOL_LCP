# Curated ISU-ILCC exploratory result

T=100 test MAPE: 26.19%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.2387096774193548.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 98.36% | 1371.52 | 1437.48 | 0 |
| normalized_cp | 95.08% | 1071.46 | 1142.29 | 0 |
| base_lcp_protocol | 96.72% | 1341.33 | 1407.75 | 0 |
| rlcp_prediction | 91.80% | 995.32 | 1313.35 | 0 |
| rlcp_protocol | 93.44% | inf | inf | 2 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
