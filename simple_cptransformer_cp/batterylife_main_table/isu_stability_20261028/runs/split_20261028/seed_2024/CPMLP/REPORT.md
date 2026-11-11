# Curated ISU-ILCC exploratory result

T=100 test MAPE: 27.09%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.2610259122157588.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 100.00% | 1760.19 | 1760.19 | 0 |
| normalized_cp | 95.08% | 1033.21 | 1230.78 | 0 |
| base_lcp_protocol | 96.72% | 1360.54 | 1542.80 | 0 |
| rlcp_prediction | 91.80% | 1071.57 | 1583.79 | 0 |
| rlcp_protocol | 93.44% | inf | inf | 3 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
