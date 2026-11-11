# Curated ISU-ILCC exploratory result

T=100 test MAPE: 29.01%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.40105764145954514.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 95.08% | 1347.78 | 1918.67 | 0 |
| normalized_cp | 95.08% | 1129.01 | 1434.88 | 0 |
| base_lcp_protocol | 100.00% | 1738.96 | 1738.96 | 0 |
| rlcp_prediction | 90.16% | 1360.21 | 1839.21 | 0 |
| rlcp_protocol | 93.44% | inf | inf | 3 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
