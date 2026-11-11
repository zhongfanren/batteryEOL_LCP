# Curated ISU-ILCC exploratory result

T=100 test MAPE: 26.14%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.6797990481226863.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 85.25% | 1400.37 | 1773.30 | 0 |
| normalized_cp | 91.80% | 1267.56 | 1655.58 | 0 |
| base_lcp_protocol | 96.72% | inf | inf | 1 |
| rlcp_prediction | 90.16% | 1255.69 | 2058.38 | 0 |
| rlcp_protocol | 96.72% | inf | inf | 4 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
