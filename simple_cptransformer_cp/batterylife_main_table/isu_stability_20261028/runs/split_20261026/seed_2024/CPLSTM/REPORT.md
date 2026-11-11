# Curated ISU-ILCC exploratory result

T=100 test MAPE: 24.40%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.4528820729772606.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 96.72% | 1174.73 | 1513.69 | 0 |
| normalized_cp | 96.72% | 1064.33 | 1091.45 | 0 |
| base_lcp_protocol | 100.00% | inf | inf | 1 |
| rlcp_prediction | 96.72% | 1285.35 | 1366.27 | 0 |
| rlcp_protocol | 95.08% | inf | inf | 3 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
