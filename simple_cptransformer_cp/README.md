# Simple CPTransformer Conformal Baseline

> 当前研究请先读 [RESEARCH_START_HERE.md](RESEARCH_START_HERE.md)。以下为初始 MATR 实验的历史说明；当前 ISU 方向、最新复核及 GitHub 同步范围见研究入口和[仓库首页](../README.md)。

本目录用于重新建立一个最小、清晰、可审计的电池寿命 conformal prediction 实验。

旧实验与已有结果继续保留在工作区原位置。本目录不依赖移动或删除旧文件；后续只把经过确认的必要实现加入这里。

## 当前唯一研究问题

给定一块 MATR 电芯前固定 100 个循环的完整充放电曲线，使用 BatteryLife 原始 CPTransformer 预测总循环寿命 EOL，并比较：

1. CPTransformer 点预测；
2. CPTransformer + Split CP；
3. CPTransformer + asymmetric Split CP；
4. CPTransformer + basic embedding LCP。

初始阶段不做逐周期 RUL、多 landmark、learnable kernel、频域核、CQR、HopCPT 或 KOWCPI。

## 数据与统计单位

- 数据路径：`D:\BatteryLife_Processed`
- BatteryLife 本地仓库：`D:\Battery-LCP\third_party\BatteryLife`
- 独立样本单位：电芯，而不是循环或前缀窗口。
- Train、model validation、calibration、test 必须按电芯互斥。
- Calibration 和 test 在固定 `T=100` 时每块电芯各贡献一个评价样本。
- 训练集内部可以使用同一训练电芯的多个前缀作为数据增强，但损失和采样需要避免长寿命电芯因前缀更多而获得不合理权重。

## 预测与区间

点预测目标为：

```text
前100循环完整电压/电流/容量曲线 -> 预测 EOL
```

固定 `T=100` 时，可以通过 `RUL = EOL - 100` 转换为 RUL。EOL 区间整体减去 100 后即为 RUL 区间，覆盖率和宽度不变。

Split CP 使用 calibration 电芯的绝对残差全局分位数。asymmetric Split CP 使用有符号残差分别校准上下两侧，使超长寿命电芯造成的大正残差主要扩张上界。basic LCP 使用冻结 CPTransformer 的 embedding，在同一 `T=100` 下按 calibration 电芯相似度计算局部加权残差分位数。

## 开始新聊天

在 Codex 中继续使用工作区 `D:\Battery-LCP`，然后发送：

> 请先读 `simple_cptransformer_cp/RESEARCH_START_HERE.md` 和当前任务对应的最新复核；按问题定位历史记录与源码，不要修改或删除冻结实验。

详细上下文见 [CONTEXT_HANDOFF.md](./CONTEXT_HANDOFF.md)。

## 可运行入口

实现与审计说明见 [AUDIT_AND_DESIGN.md](./AUDIT_AND_DESIGN.md)。在已配置的
`my_torch` 环境中运行：

```powershell
conda run -n my_torch python simple_cptransformer_cp/run_t100.py
```

每次运行都会新建 `simple_cptransformer_cp/runs/t100_<timestamp>/`，不会覆盖旧结果。

当前非对称 Split CP 的首次结果与分析见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)。
