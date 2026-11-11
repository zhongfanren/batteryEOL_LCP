# Curated ISU-ILCC exploratory result

T=100 test MAPE: 25.58%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.5061343204653621.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 81.97% | 1059.26 | 2206.31 | 0 |
| normalized_cp | 80.33% | 987.89 | 1885.86 | 0 |
| base_lcp_protocol | 91.80% | inf | inf | 1 |
| rlcp_prediction | 81.97% | 1070.09 | 2370.67 | 0 |
| rlcp_protocol | 88.52% | inf | inf | 4 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
