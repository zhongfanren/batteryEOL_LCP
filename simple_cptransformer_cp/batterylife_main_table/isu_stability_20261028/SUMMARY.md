# ISU-ILCC stability batch

Exploratory; overlapping cell resplits are not independent replications.
No model, feature or bandwidth is selected by these results. Infinite widths are retained.

| Model | Method | Completed fits | Mean resplit coverage | Infinite-repeat fraction | Mean width across all fits/resplits |
|---|---|---:|---:|---:|---:|
| CPLSTM | curve_scale_cp | 9 | 90.10% | 0.00% | 1355.502615530838 |
| CPLSTM | normalized_rlcp_curve_h1.5 | 9 | 92.08% | 99.50% | inf |
| CPLSTM | normalized_rlcp_curve_h3 | 9 | 91.03% | 53.00% | inf |
| CPLSTM | normalized_rlcp_prediction_h1.5 | 9 | 90.91% | 46.78% | inf |
| CPLSTM | normalized_rlcp_prediction_h3 | 9 | 90.59% | 3.17% | inf |
| CPLSTM | prediction_scale_cp | 9 | 90.04% | 0.00% | 1131.674374576809 |
| CPLSTM | protocol_scale_cp | 9 | 90.21% | 0.00% | 1167.55246409513 |
| CPLSTM | relative_cp | 9 | 90.00% | 0.00% | 1168.9554043006724 |
| CPLSTM | split_cp | 9 | 90.05% | 0.00% | 1263.1164626952905 |
| CPMLP | curve_scale_cp | 9 | 90.19% | 0.00% | 1290.9158771718828 |
| CPMLP | normalized_rlcp_curve_h1.5 | 9 | 92.31% | 100.00% | inf |
| CPMLP | normalized_rlcp_curve_h3 | 9 | 91.18% | 46.78% | inf |
| CPMLP | normalized_rlcp_prediction_h1.5 | 9 | 90.95% | 29.50% | inf |
| CPMLP | normalized_rlcp_prediction_h3 | 9 | 90.71% | 0.72% | inf |
| CPMLP | prediction_scale_cp | 9 | 90.18% | 0.00% | 1021.9043312494618 |
| CPMLP | protocol_scale_cp | 9 | 90.21% | 0.00% | 1048.4612803290545 |
| CPMLP | relative_cp | 9 | 90.27% | 0.00% | 1117.4003007262397 |
| CPMLP | split_cp | 9 | 90.18% | 0.00% | 1179.9239512336883 |

Read summary.csv for each split/initialization. Original test and resplit results are separate.
Each analysis/job directory contains raw repetition metrics, original cell radii, feature diagnostics and feature checkpoint identity.
Interpret width and interval score with coverage. Larger or infinite intervals alone do not establish useful localization.
