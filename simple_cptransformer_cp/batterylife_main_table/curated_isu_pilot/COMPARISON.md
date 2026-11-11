# ISU-ILCC frozen-split model comparison

Exploratory. Both models are preset; do not select a model or localizer by these test scores.

| Model | Method | Test MAPE | Coverage | Mean width | Infinite |
|---|---|---:|---:|---:|---:|
| CPMLP | split_cp | 25.77% | 96.72% | 1157.5843705974644 | 0 |
| CPMLP | normalized_cp | 25.77% | 93.44% | 975.8244940497443 | 0 |
| CPMLP | base_lcp_protocol | 25.77% | 98.36% | 1493.5571279056737 | 0 |
| CPMLP | rlcp_prediction | 25.77% | 91.80% | 1174.635673147733 | 0 |
| CPMLP | rlcp_protocol | 25.77% | 98.36% | None | 3 |
| CPLSTM | split_cp | 28.33% | 90.16% | 1050.7116929351307 | 0 |
| CPLSTM | normalized_cp | 28.33% | 93.44% | 1037.9381308829174 | 0 |
| CPLSTM | base_lcp_protocol | 28.33% | 96.72% | None | 1 |
| CPLSTM | rlcp_prediction | 28.33% | 93.44% | 1243.4280925813268 | 0 |
| CPLSTM | rlcp_protocol | 28.33% | 93.44% | None | 2 |
