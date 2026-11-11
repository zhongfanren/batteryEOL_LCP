# 首次提取尝试：失败记录保留

本次只完成一个分折模型的提取，未产出科学结果。混合117块电芯的推理批次后，CPLSTM少数预测相对旧OOF出现最多0.0126周期的CUDA数值差，严格重放校验中止。

`runner_failed_snapshot.py`保留当时脚本；其SHA256对应本目录`plan.json`中的code_sha256。没有修改旧模型或调宽核对容差。

改为按原outer_holdout顺序和原batch size分别推理后，完整结果另存`../curve_signal_20261104_v2/`，旧194条OOF最大偏差5.7e-14周期。
