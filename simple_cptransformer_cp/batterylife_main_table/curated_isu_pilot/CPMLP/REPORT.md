# Curated ISU-ILCC exploratory result

T=100 test MAPE: 25.77%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.3834479111581173.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 96.72% | 1157.58 | 1871.28 | 0 |
| normalized_cp | 93.44% | 975.82 | 1390.82 | 0 |
| base_lcp_protocol | 98.36% | 1493.56 | 1798.43 | 0 |
| rlcp_prediction | 91.80% | 1174.64 | 1687.67 | 0 |
| rlcp_protocol | 98.36% | inf | inf | 3 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
