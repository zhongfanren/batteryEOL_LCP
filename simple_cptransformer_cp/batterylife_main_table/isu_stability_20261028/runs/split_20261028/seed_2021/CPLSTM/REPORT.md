# Curated ISU-ILCC exploratory result

T=100 test MAPE: 38.92%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.19682707562136434.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 96.72% | 1776.01 | 1936.13 | 0 |
| normalized_cp | 95.08% | 1522.69 | 1639.88 | 0 |
| base_lcp_protocol | 98.36% | 1860.08 | 1987.10 | 0 |
| rlcp_prediction | 96.72% | 1505.32 | 1733.96 | 0 |
| rlcp_protocol | 95.08% | inf | inf | 1 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
