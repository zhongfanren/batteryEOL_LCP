# Curated ISU-ILCC exploratory result

T=100 test MAPE: 22.33%; n=61.
Dev-fitted difficulty vs test |error| Spearman: 0.343204653622422.

| Method | Coverage | Mean width (cycles) | Interval score | Infinite |
|---|---:|---:|---:|---:|
| split_cp | 93.44% | 1439.65 | 1904.04 | 0 |
| normalized_cp | 95.08% | 1142.75 | 1237.60 | 0 |
| base_lcp_protocol | 100.00% | 1666.68 | 1666.68 | 0 |
| rlcp_prediction | 96.72% | 1334.36 | 1426.72 | 0 |
| rlcp_protocol | 93.44% | inf | inf | 1 |

Inspect group_diagnostics.csv for risk groups and protocol conditions. Small groups are descriptive.
Widths and interval scores must be interpreted together with coverage. Different data populations are not a controlled improvement over MATR/mixed data.
