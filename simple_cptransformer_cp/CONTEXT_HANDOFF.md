# 新聊天交接：BatteryLife 预测器复现与 MATR-only 校准/测试

## 2026-11-11：多分位输出画像适配已运行完成（当前最新）

- 用户授权“来试一下”此前解释的低容量三分位误差画像。已实现并完成`batterylife_main_table/quantile_profile_20261111/`，优先`REVIEW.md`，全表`BRIEF.md`。脚本`run_quantile_profile.py`、`review_quantile_profile.py`、`plot_quantile_profile.py`，4契约测试`test_quantile_profile.py`。未训练点预测器，未重新提取embedding，未评价原cal/test；复用curve_signal_v2手工9维摘要和train97 OOF、dev20辅助预测。
- 固定分位水平.1/.5/.9，QuantileRegressor alpha.01。原life2（标准logp及平方）+扣寿命关系/PCA3手工，共5输入，每头含截距6系数、三头18系数。目标直接R=abs_error/max1pred。预测先非负clip、再排序，q90-only画像用同一个排序后三维画像的末维，避免换模型。不是原文DSCP-Local深encoder复现，是输出分位向量的适配。
- 关键隔离：参考点画像不能用自己的R拟合。内层留原辅助fold，CV参考77/78内4折，dev参考97内5折；每个inner map对本块reference和query同映射，profile kernel h=.5内层fit画像正距离中位数（常数则h1）；每块权重先块内归一，再乘nblock/nref，CDF混合。是监督画像+分块CDF设计，不能把结果归因为只改一个纯几何，也不是单一冻结map的正式RLCP。底层OOF训练仍交叠，不能称端到端嵌套独立检验。
- 7预设方法：kernel_global/life/hand（旧对照）、qr_hand_q90（直接头）、profile_life_3q、profile_hand_q90、profile_hand_3q（主要候选）。两个假设：学习几何超过全局及手工；三分位超过单q90及life-only。主门槛预定OOF对global与hand均>=2%及各>=3/5胜，dev对两者均正且各>=3/5胜；没按结果调alpha/h。多分位增量要求CV/dev相对q90均正且>=3/5CV胜。
- 结果：主要三分位CPMLP OOF pinball.0544，对全局+6.2%、对手工仅+1.4%、对q90-only+2.8%；dev.0291，对全局−2.5%、对手工−21.0%、对q90−3.5%。CPLSTM主要OOF.0598，对全局−.3%、对手工+3.5%、对q90−10.3%；dev.0243，对全局+17%、对手工+7.8%、对q90+3.6%。两架构几何门槛和多分位增量检查均未过，**没有进入正式CP评测**。
- 副对照正面线索保留：CPLSTM profile_hand_q90 OOF对global+9.1%、对hand+12.6%，各3/5折胜；dev对hand+4.3%、5/5视角胜。但CPMLP OOF对hand−1.4%、dev对hand−16.9%，未跨架构复现，不能看到结果后替换主候选宣布成功。手工三头大多数非零系数，输出实际变化，非上轮常数退化；多分位画像差异未转化为稳定预测收益。
- 保存2758逐查询损失、模型JSON、20邻域NPZ（六种权重、参考画像、各inner查询画像、原reference scores）、profile_rows（查询画像均值只供显示，权重用matched inner map）、90项inner拟合/参考/查询隔离清单。四种旧对照全数精确重放最大差0；独立核对2364核分位、2758损失、画像有序/权重和/块质量/原输入hash，均通过。
- `visuals/PROFILE_CDF_all97.pdf`固定全部97主CPMLP OOF查询，每页global/hand/q90画像/三分位画像的完整CDF、尾部放大、逐参考权重与各inner查询画像。全量CDF点/归因CSV、首ID示例图。无CP修正/测试∞，这些是信息筛查，不能把经验fraction_below_q当最终覆盖。图和数值状态独立落盘，完成状态见state和visualization_complete。
- 下一建议：暂停当前三分位线性画像，不因失败增大网络/按dev搜参数；回到已有固定手工邻域的尾点权重归因，解释原小幅信号。当前课题/目标总体/冻结预测器都未变，历史负结果保留。

## 2026-11-08：已读用户Zotero的Lee等2026论文（仅文献判断，未启动新实验）

- 论文`Conformal Prediction for Time Series with Deep Sequence Models`，Lee/Lee/Xie，arXiv2610.02357v1。用户本地PDF位于`C:/Users/wjndyajl/Zotero/storage/TKAQ8UZ3/`；60页实际读取，出处hash在`literature_notes/lee_2026_sequence_cp/source.json`，详细中文判断`READING_NOTE.md`，可检索全文`paper.txt`。MiKTeX pdftotext初始化受限，最终使用已安装系统Python313的fitz读取，不修改系统环境。
- 三路线DSCP-QR、DSCP-QFE、DSCP-Local，固定点预测器、原文分数是有符号残差，区间为点预测+两端残差条件分位。最相关Local用多分位pinball训练encoder，再丢弃预测头，用embedding相似性加权校准CDF取分位；附录B.3第45页训练水平{.1,.5,.9}，cal数据按60/15/25分训练/验证/留出权重参考。不是直接复用MSE点预测embedding或曲线重建SSL。
- 理论§3.3是固定encoder下、平稳beta-mixing、局部质量/连续平滑CDF、随样本量收缩带宽等条件下的渐近H条件覆盖；升级完整Z条件覆盖额外要求残差充分性。**附录第45页明确：实验softmax点积/cos/欧氏权重不直接被紧支撑核/带宽条件的定理覆盖。** 不能说直接解决当前小样本Guan/RLCP有效性。
- 时序实验使用历史协变量/残差/可选真实观测，下一时刻Y有逐步反馈；本项目固定T100横向EOL没有同一新电芯前99周期已知真实EOL残差。只能借表示监督目标，不引入伪在线残差或把100前缀当独立误差样本。单序列数千上万观测不等于我们的97电芯监督规模。
- 建议：保留当前手工邻域线索与归因，下一候选可低容量多分位残差画像[q.1,q.5,q.9]几何，与固定手工核和同分数尺度CP比；这“输出分位向量作距离”是本项目适配设想，原文用encoder隐向量，勿混同。先train侧OOF，保留最终cal隔离和选定有效CP/RLCP，不重训11预测器。此前Hopfield/CRPS/六参数距离已做残差监督，不能把此方向说成全新。没有启动任何新训练或CP评测。

## 2026-11-04：曲线信息初筛已完成（当前最新）

- 用户明确授权按手工特征、冻结预测器表示、直接曲线邻域继续检查。已实际完成`batterylife_main_table/curve_signal_20261104_v2/`，优先`REVIEW.md`（强对照复核），全表`BRIEF.md`。新脚本`check_curve_signal.py`、`review_curve_signal.py`、`plot_curve_signal.py`；未改旧冻结代码/点预测器，原cal/test未评价。仅原train97 OOF + dev20。
- 两假设：附加早期信息能否改善寿命控制后的相对误差90%分位；不先学风险标量的直接邻域是否有增量。hand为模型归一化输入9维early/change/RMS摘要，非物理SoH；curve复用57维（每通道16采样均值趋势+std/差分std/斜率）六特征块平衡；embedding使用每个原OOF分折冻结模型CPMLP32维/CPLSTM256维。10模型只inference train97+dev20，未新训练。每个验证折encoder不曾训练该折，不拼接不同模型坐标；PCA3/标准化/寿命关联扣除仅拟合参考侧。
- OOF CV留原辅助外折19/20，参考其他77/78 OOF残差。参考辅助模型可能见过当前验证折标签，因此仍不是端到端嵌套独立验证；此限制保留。dev20未进任何辅助训练或其epoch选择，5折各预测同20块，不是100独立样本；参考97为不同辅助模型混合，不能直接推论最终主模型。
- 9方法：qr_life/hand/embedding/curve（固定alpha.01，目标直接R=abs_err/max1pred，非此前logR）；kernel_global/life/hand/embedding/curve（固定RBF h=.5参考距离中位数，无监督规则，不搜h）。附加特征先扣除参考侧logp二次项关联，regress用PCA3，直接curve距离用57维六块等权。CDF经验阈值.9，**没有CP有限样本修正/测试∞，仅信息筛查**，不得把fraction_below_q当最终覆盖率。
- 预定初筛门槛对匹配life-only：OOF pinball改善>=2%及>=3/5折胜，dev改善>0且>=3/5视角胜。两架构kernel_hand和kernel_curve过门槛，但**global在OOF也比life-only更好，初筛通过不等于发现稳定新增信息**。独立REVIEW同时比较预先保留global：CPMLP hand对life +16.6%、对global仅+4.9%(4/5折)，dev对global+15.2%；curve OOF对global仅+.4%、dev+18.9%。CPLSTM hand/curve OOF对global−3.9%/−5.0%，dev+10.0%/+14.3%。跨架构不一致。内部embedding无稳定强对照收益；附加特征QR在dev全恶化。保留简单手工邻域作为线索，不能宣称LCP成功。
- v1`curve_signal_20261104`首次混合批次推理使packed LSTM最多0.0126周期浮点差，严格校验失败，无科学结果；保留failed state、README、原脚本快照runner_failed_snapshot.py（对应旧plan源码hash）。v2按原outer_holdout顺序/原batch size重放，不放宽容差，194旧OOF最大差5.7e-14。不是数据错位或训练失败。
- 5契约测试通过；独立复核3546条pinball/分位指标、1970个kernel分位数、名单隔离及源hash。保存input_descriptors、10冻结features、20个CV/dev邻域NPZ、变换参数/模型、predictions/folds/summary、强global比较与tail_support。`visuals/OOF_CDF_all97.pdf`固定全部97主CPMLP训练侧查询，三列hand/embedding/curve，每列global/life/feature同参考score对照，另例图/权重CSV/分位归因。图必须标记诊断而非CP；原始测试数据未进图。
- 下一步建议（未启动新CP）：围绕固定手工邻域查哪些尾部参考点被加/减权、跨折是否一致，再冻结方法做主预测器同分数有效CP消融并保留CPLSTM复核；不围绕dev/test改h或挑图，不训练复杂网络。统一入口和总账已更新，历史负结果不删。

## 2026-11-04：分数×局部性第一轮完成（当前入口已整理）

- 用户明确授权继续探索分数与局部性、每轮CDF诊断、优化实验管理，并允许删除确无用文件。新入口`RESEARCH_START_HERE.md`、总账`EXPERIMENT_INDEX.md`。未发现历史产物阻碍计算，本轮没有删除数据/模型/负结果，也未改任何旧冻结脚本。主线仍固定T100横向预测，在ISU开发、MATR后续复核，不扩大数据或重新训练点预测器。
- 已实现并实际完成新CPU批次`batterylife_main_table/score_locality_20261104/`；入口`run_score_locality.py`，绘图`plot_score_locality.py`，独立审阅`analyze_score_locality.py`。优先读`BRIEF.md`，完整表`REPORT.md`，冻结`plan.json`。共18设置×201原/重分×9法=32562行指标，9882行原test区间，全部保存，训练预测器0次，无后台训练。
- 两个预设问题：Htail仅训练侧OOF97检查弱正则尾部泛化；Hlocal同分数下1D分歧局部性是否提供额外价值。三分数absolute、relative、gap6（尺度1、p、p+d6）×三校准global、life、gap。局部均1D标准化特征logp或log1p(d6/p)，只用train97拟合标准化，h6固定。组别用dev20可观察协变量三等分，不用误差。原有Hore–Barber Gaussian RLCP，保留随机中心和测试∞原子；不是Guan校正LCP的直接实现，也不是旧MATR32/34。
- OOF风险诊断固定5候选：常数、life alpha.01、pair alpha.05(旧)、pair alpha.01(主要)、pair alpha0(敏感性)。按原辅助外折留一风险CV，折内重算标准化；固定OOF库辅助训练重叠，可能见过其他风险验证折标签，不是端到端嵌套独立验证。主要候选需对常数和life pinball均提升2%、各≥3/5折胜、≥4/5非恒定、full std>.01；两模型未过。alpha.01非恒定但CPMLP对常数损失恶化6.9%、CPLSTM恶化.3%，对life也差；alpha0更差。alpha.05仍常数。没有依据当前test选alpha，没有把此风险模型接成LCP。
- 新gap局部化同分数高组覆盖有小幅增加：按重复呈现加权，relative CPMLP81.90→83.20%、CPLSTM80.99→82.55%；每架构9×200×61中分别12/2次∞，三分数权重共用导致∞位置相同，不是独立三批。∞保留所以总平均宽/score∞。life局部化无∞但relative/gap6平均宽+1–2%、score稍差。六模型gap尺度原有小幅收益精确重现，不能称新局部性成果。主CPMLP原61testrelative global88.52%/宽947.4，relative gap90.16%/宽1023.0，只属覆盖成本交易，不能挑单次成绩宣布成功。
- 新CDF直接保存源预测/标签、score、尺度、每query权重、测试∞质量和随机噪声`primary_intermediates.npz`，无需反演。`visuals/CDF_{absolute,relative,gap6}_all61.pdf`每份全部61主test，共183页；`relative_intervals_test61.png`、三张固定首ID示例、33489条CDF全点、549条q归因。横轴归一化分数时无量纲、纵轴权重含测试∞质量，阈值90%而非旧MATR94.12%。测试实际误差只作标题/事后分析。
- 5科学契约测试通过（test标签不改半径、协变量预处理不读标签/不fit测试、有限样本rank、测试∞、常数尺度抵消）；五种旧方法全设置重放最大偏差9.1e-13；549q逐一独立CDF重算，组计数/覆盖/成本回收总体，输入哈希不变。Windows Python3.9对单行极长中文README字面量报UTF8错误，拆成多行后绘图完成；不是数值实验问题。
- 下一建议（**尚未启动**）：停止在同一pair分歧上调alpha或历史test搜h，先仅建模侧检查少量低维早期曲线特征对归一化残差上尾是否有额外信息，寿命only作对照。必须有稳定信息才接新距离/分数；没有信息也可保留简单尺度结论，不因希望做LCP而增加复杂度。每轮先冻结两假设/强对照，再计算、全量图、简报决策，历史结果仍探索性。

## 2026-10-28：MATR逐测试电芯加权CDF诊断已完成（当前最新）

- 用户观察旧MATR fixed_RBF某些小实际误差电芯区间很宽，提出画每块加权经验CDF，查kernel/bandwidth/尾部大跳来源。已实际完成诊断，未训练预测器、未修改旧区间。入口 `diagnose_matr_lcp_cdf.py`，目录 `interval_views_20261028/MATR_fixed_RBF_CDF/`，README、全部33页PDF `ALL_33_weighted_CDFs.pdf`、33张cells/{ID}.png、CDF_overview.png、全权重/CDF/归因CSV、h×.5/1/2敏感性表。
- 精确来源仍旧transfer批次20260926_213949_940282 seed2024（主图）及42/2021（量化校验）。旧脚本没有保存cal预测或权重。本次从169块缓存曲线重建原57特征、robust/PCA8和RBF权重（h7.59229850769043）；通过33个保存的query weighted log-error means及1个uniform mean，解34×34满秩线性系统，数值反演恢复34个校准绝对误差。矩阵condition377107.7；三种子全部99个原q独立校验最大误差8.25e-9周期，非负/方程残差/普通CP q/区间均复核通过。不是保存的原cal预测文件，也不能恢复误差符号；须说明此provenance，不能冒充原始中间产物。未恢复SSL权重或其CDF，不能编造。源哈希不变见audit.json。
- 旧basic加权实现对ncal34使用ceil(35*.9)=32，经验加权分位水平32/34=.941176，而非.9（也不是当前RLCP的+∞原子构造）。最大三个校准误差251.5099（MATR_b2c47，EOL712）、717.0760（MATR_b3c38，EOL1932）、882.0514（MATR_b1c2，EOL2237）。251→717之间存在465.566周期空档。
- seed2024全部33半径四档：146.5322×18、251.5099×5、717.0760×8、882.0514×2。平均ESS29.58/34、temp全1，权重总体不极端集中，未触发ESS平滑。宽区间主要由尾部质量跨越5.88%预算和残差支持大空档共同产生，并非最大竖跳必然在选中点。
- 例MATR_b2c20：EOL501、pred476.7187、实际误差24.28，却q717.076。两个最大误差点合计权重7.1515%；CDF在251.51处92.8485%<94.1176%，在717.08处96.5558%达到阈值。选中点自身weight3.7074%，最大竖跳是MATR_b2c22/error71.0632/weight5.5548%。最大竖跳与选中q点只在3/33案例相同。必须分清竖直跳跃=单点权重、水平空档=残差样本稀疏。
- 固定敏感性h×.5时该例q120.47、tail2mass4.78%、ESS11.39；h×2仍q717.08。仅作机制诊断，未计算测试成绩挑h，不能声称减半带宽方法已成功。全部33及3种子敏感性保留。
- 用户“实际预测准就应区间短、只有长寿命才应宽”的直觉需校正：单次事后误差不等于可提前识别的条件尾部风险，校准中也确有长寿命大误差点。支持权重/尾部分位数问题值得调查，但未证bandwidth本身设错或宽区间都不合理。后续须用可观测特征决定尾部借用、并评估缩短后的覆盖，不能按test误差定宽度或直接删大cal误差。

## 2026-10-28：用户明确横向任务，并查看全部测试电芯区间（当前范围）

- 用户明确“随运行过程更新寿命区间不是想做的”，选择固定时点横向任务，当前T100。不再引入动态/纵向预测路线。共同定义 `batterylife_main_table/LCP_EXPERIMENT_SPEC.md`已补充。难度/风险适应是这个设定下的评价方向，不要求必须先学一个难度模型才能用LCP。
- 用户要看ISU/MATR每块原test电芯的区间和真实寿命。已用只读导出 `export_test_interval_views.py` 创建 `interval_views_20261028/`：ISU_CPMLP_test61.csv及ISU_test61.png；MATR_CPTransformer_seed2024_test33.csv及MATR_test33.png；MATR_11models_splitCP_test33.csv。中文CSV每块一行，真实寿命、点预测、各方法上下界/覆盖；11模型表长格式363行。图按真实寿命排序仅为显示，全部电芯包含，不筛漏覆盖。
- ISU源 `batterylife_main_table/isu_crossfit_risk_20261028/original_intervals.csv`，模型CPMLP2024原61test（原文件还含CPLSTM与18方法）；图展示寿命尺度、六模型分歧尺度、OOF平均风险尺度，退化尾部未重复作图。
- MATR早期LCP源 `learnable_localizers/runs_transfer/20260926_213949_940282/seed_2024/evaluation.csv`，历史CPTransformer2024原33test，含Split、fixed_RBF及3Hopfield方法。图是旧basic LCP诊断，不当当前RLCP保证；旧负下界原样保留。MATR11模型普通CP源 `batterylife_matr_benchmark/frozen_split_cp/20261021_193212_072306/test_intervals.csv`，这与早期CPTransformer不是同一检查点，不混比；两源33test IDs核对一致。
- 导出不训练、不重校准、不改任何上下界；源哈希核对保持不变，见interval_views/source_manifest.json。读取不影响训练；若据图选方法或调参仍属历史test开发，不能称未见检验。用户当前想理解结果，不自动开启新训练。

## 2026-10-28：用户澄清原始研究意图——横截面与动态预测尚需定题

- 用户明确：最初是逐周期寿命预测＋应用读到的LCP论文；T100是为节约时间和复杂度采用的固定时点简化，并非一开始就选定“难电芯宽、易电芯窄”为唯一研究主题。之前助理把这一近期候选主线表述得过于确定，后续需纠正，不能把它当作用户/导师既定要求。
- 用户正在辨析横向（固定观测周期，不同电芯）与纵向（同一新电芯在10、20…周期随历史更新寿命区间）。LCP局部指特征/状态邻域，不限定横向或纵向。T100当前研究是固定landmark子问题，可作为动态预测的一个切片；动态是否成为总框架仍待讨论，本轮不自动换实验。
- 相同周期数不是同一健康状态或同一寿命消耗比例。当前ISU训练实际用1..100前缀，校准/评价只取T100；历史全轨迹可用于按前缀训练，但t输入仅用截至t的观测，独立单位仍是电芯。239样本按T100存活资格筛选，拓展至更早时点的目标总体须重新解释。
- 动态覆盖必须区分固定每个t跨新电芯的边际覆盖、同一轨迹时间平均覆盖、所有预设时点同时覆盖；重复窗口不能直接视为iid。EOL标签直到失效才知道，20周期时并不能获得10周期寿命预测残差，不能直接套要求即时反馈的在线CP。纵向区间可更新但不保证每次必然收窄或自动获得轨迹覆盖。
- 文献核对：Guan LCP https://arxiv.org/abs/2106.08460；Hore–Barber RLCP https://arxiv.org/html/2310.07850；Lin/Trivedi/Sun temporal dependence https://arxiv.org/abs/2205.12940 区分横截面与纵向覆盖。用作概念解释，不已验证为当前EOL重复预测的现成解法。

## 2026-10-28：用户要求理解全局进展并提问

- 用户希望比较到目前与旧MATR的正面表现、研究问题上的进展和待解决方向。已整理完整说明 `RESEARCH_PROGRESS_SUMMARY_20261028.md`，保留9月旧总结。此轮仅总结，没有启动新训练或调参。
- 核心口径：在固定T100和预测器下，提前识别误差尾部风险，合理覆盖和宽度成本下改善风险区域分配，再检验局部校准超出简单尺度的价值。用“半径=s(x)×q(x)”区分尺度CP和LCP：现有正面证据主要在尺度s，局部q增量尚未证明。
- 尊重旧MATR正面证据：三种子ensemble+固定六组RBF那批，Split/固定六组覆盖都90.20%，宽617.2→597.9、score974.4→944.1；可学六组935.2对固定只改善约.94%、逐折不稳。不能把旧实验说成全部无效，也不能将其basic weighted quantile与当前Gaussian RLCP当相同构造。最新OOF不是交叉拟合首次出现，旧阶段已有相关探索。
- 新正面进展：ISU合格239 vs MATR169、cal/test61/61 vs34/33或CV20/20；18冻结设置中寿命尺度收益一致；六模型分歧尺度有小幅同向收益和高组覆盖约81→83/84%的分配线索；获得可复用OOF97。没有证明ISU预测性能/LCP绝对指标优于MATR，没有获得可主张的稳定学习LCP方法贡献。
- 当前需要用户理解：不同批分组定义不同、200重分不是新增独立数据、开发/校准职责不同、历史test暴露未消除、模型分歧信号在六模型与单模型对之间尚不稳、mean risk和tail radius不是同一目标。下一步推荐仅训练侧非退化/泛化诊断→同分数局部增量→冻结后MATR跨数据集复核；不按历史test搜alpha，不直接加复杂网络或扩大数据课题。

## 2026-10-28：OOF批次完成并复核——尾部模型发生常数退化（当前最新）

- 用户告知跑完并要求分析。10/10辅助任务于19:16完成，18:41:26→19:16:00共2074秒（34.6分钟），无失败/超时。新独立复核入口 `batterylife_main_table/analyze_isu_crossfit_risk.py`，结果 `isu_crossfit_risk_20261028/analysis_review/ANALYSIS.md`，优先于原自动BRIEF解读；原计划、运行代码和实验输出未改。
- 分折/inner epoch选择/outer refit标签scaler/checkpoint训练名单/194行OOF合并均核对通过，97块每架构恰一次heldout，不包含原dev/cal/test；10个inner所选epoch均未触达30上限。2196行原test区间算术、全体分组回收总体、全部风险参数从JSON重建及源文件hash检查通过，见analysis_review/audit.json、fold_audit.csv。
- 样本来源确实改善：dev20的>50%相对误差两架构均0；OOF97中CPMLP7、CPLSTM9。OOF平均相对误差22.1%/23.2%，历史pool22.4%/22.5%，总体量级未明显失配，但条件分布一致未证明，OOF辅助训练77/78与主97及随机种子/epoch不同仍是限制。不是新增97块物理数据，OOF训练集重叠，不宣称残差独立。
- **关键诊断：两个OOF quantile90回归的三个系数全0**，L1alpha=.05使当前目标最优解为常数。常数解KKT绝对特征梯度最大CPMLP.02256/CPLSTM.02556<.05；并非优化器没跑。风险分数/距离恒定，两个dev三分位阈值重合、所有122块进group2，低/中风险组空。自动BRIEF的rho=nan是零方差未定义，不是漏算。不能把这个‘高组’解释为真正高难组。
- OOFTail尺度=c*预测寿命，校准常数抵消c，严格退化relative_cp；learned_risk RLCP所有权重相同。已核对两模型全部201次划分全等。**因此本轮没有实质检验有效学习尾部距离，不能据此判LCP无效；无∞也不是有效局部化解决方案。** 原自动简报漏解释退化，追加报告已明确标记，勿再直接据原BRIEF方法同分判断。
- 非恒定Ridge平均误差尺度：当前两个固定seed2024设置200重分，CPMLP relative宽1052.9/score1545.3，六模型gap1035.8/1473.0，OOF寿命1099.0/1549.0，OOF寿命+pair1127.5/1503.0；CPLSTM relative1069.2/1523.6，六模型gap1045.2/1459.6，OOF寿命1063.2/1484.7，OOFpair1150.9/1446.5。总体覆盖约90%。不是前批跨9设置均值，勿混比。
- OOFpair对同算法dev20（宽1896/1917、score2021/2026）改善很大，但对强基线没稳定优势。对同OOF寿命only，score改善约3.0%/2.6%，宽度增加2.6%/8.2%；对六模型gap，CPMLP宽/score都差，CPLSTM以宽度+10.1%换score约−0.9%。平均风险对pool相对误差Spearman−.106/−.025，总体风险排序弱，不据此单独否定尾部信号。
- 单种子pair_gap组不同于此前六模型gap组。CPLSTM OOF平均风险尺度高pair组覆盖87.5→91.8%，低pair组94.5→87.3%，是再分配而非兼顾。主CPMLP原test relative覆盖88.5%、宽947；OOFpair仍88.5%、宽1054。不能声称原主test成功。
- 下一步推荐：保留OOF预测，**无需再跑GPU训练**；先在训练侧OOF检查正则化/特征能否形成稳定非恒定尾部风险，再冻结配置评估LCP。不能按已看cal/test反复调alpha；非零系数本身不证明泛化。当前只做复核分析，没有额外启动训练或新调参实验。后续历史标签暴露限制继续保留。

## 2026-10-28 18:41：训练侧OOF风险学习启动记录（现已完成）

- 用户授权解决可靠误差样本不足，再判断是否转化为有效LCP。本轮已实现并启动，不等待再次批准，不持续监控。目录 `batterylife_main_table/isu_crossfit_risk_20261028/`，当前事实以 `state.json`、各折 `progress.json` 为准；`README.md`说明监控，`BRIEF.md`只在自动评价后生成。启动核验：可见PowerShell PID34928、Python队列PID35132、首任务fold_0_CPMLP子PID7752，GPU已工作。PID仅是启动时证据，之后必须实时检查。
- 入口 `run_isu_crossfit_risk.py`，可见启动器 `start_isu_crossfit_risk.ps1`。原train97内cell-level shuffled5折（seed20261028），每折CPMLP/CPLSTM各一个固定种子，共10任务；每任务先inner fit约61/inner dev约16选择epoch，再从头outer fit约77按选定epoch重训，外折约20的标签不参与训练/标准化/早停。原dev20/cal61/test61不进辅助训练。保留1..100前缀训练、T100逐电芯OOF；单任务内选轮数最多30、patience6、least5。任务20分钟、队列90分钟上限，超时/失败停止并保留进度。
- 辅助模型只算单种子模型对分歧，查询侧也用原seed2024的同一对；六模型分歧尺度另列强对照。主预测中心仍原CPMLP2024，CPLSTM2024固定复核。OOF预测只用于风险拟合，绝不直接充当最终模型的校准残差；辅助训练77与主97的条件误差差异未假定可忽略。不同外折训练集重叠，不把OOF误差称统计独立样本；按cell折匹配当前随机电芯目标，不宣称跨未见protocol泛化。
- 自动入口 `evaluate_isu_crossfit_risk.py`，全部10任务完成后合并194行OOF（97×2架构）。低容量风险特征log预测寿命、其标准化二次项、log1p(模型对gap/预测寿命)。Ridge alpha10拟合log相对误差平均趋势，QuantileRegressor quantile.9/L1alpha.05拟合log相对误差尾部；误差floor.02、预测log尺度clip[-4,2]，全部运行前冻结。dev20同算法作样本来源对照，OOF寿命only作分歧增量对照。
- 18预设构造：Split、relative、旧dev寿命尺度、六模型gap尺度、单对gap尺度、dev20平均风险尺度、OOF寿命平均尺度、OOF分歧平均尺度、dev20分歧90%尾部尺度、OOF分歧90%尾部尺度；两种分数(relative/OOFtail)×两个1D距离(life/learned tail risk)×h3/6共8RLCP。原61test与200次固定池61/61重分均保留，分数组内全局/局部对照；无穷保留。pair_gap及learned_risk分组用dev可观察变量/预测风险三等分点，不同于此前六模型gap组。
- `plan.json`正式启动前已冻结代码、配置、模型实现、六套特征及manifest指纹，不应在队列运行中编辑相关脚本/依赖。曲线bank来自既有train97缓存，已核对原曲线和prefix masks及训练标签；每阶段只用当阶段fit电芯重算label均值/std。来源hash在各worker和评价前检查。
- 启动前6项分折/标签/分位数隔离测试通过；两个架构实际CUDA训练和预测smoke通过。临时合成OOF下完整7236行评价（2模型×201×18）走通，四个已有基线与前批重分精确一致；临时合成结果已丢弃，绝非科学结果。smoke目录 `isu_crossfit_risk_smoke_20261028`只作代码核验，未用于正式OOF。正式结果尚未产出，不能提前判定方法有效。
- 完成后用户再触发分析；独立核对OOF来源/epoch选择、尾部样本数、原test与重分、同分数局部收益、高风险实际误差和组覆盖/宽度及∞比例。若监督增加仍无收益，不能继续把所有负结果归因于样本数或直接加复杂网络。历史测试暴露不因换划分/OOF而消失，本轮仍探索。

## 2026-10-28：分歧尺度 vs 分歧RLCP试验已完成（当前最新）

- 用户认可风险信息思路并授权继续尝试。本轮冻结全部18设置，保持原单模型预测中心，未训练预测器，未复用上一审计中在历史cal/test上监督拟合的风险模型。入口 `batterylife_main_table/run_isu_disagreement_cp.py`，独立复核 `analyze_isu_disagreement_cp.py`，结果 `isu_stability_20261028/disagreement_cp_20261028/BRIEF.md`（优先读）、`REPORT.md`、`state.json` complete18/18。没有后台任务。
- 运行前固定12方法：Split、旧dev寿命尺度、relative_cp尺度p=max(1,预测寿命)、gap_scale_cp尺度p+d；d=CPMLP三种子均值与CPLSTM三种子均值绝对差，系数1不调参。relative与gap两种分数分别搭配1D log(p)与2D[log(p),log1p(d/p)]的Gaussian RLCP，train97侧标准化，h=3和6全部报告。测试原子+∞；随机第一坐标共用，尺度对照局部权重共用。同122池原61/61与200次重分，不是新增独立数据。
- 分歧尺度有小幅一致线索：CPMLP覆盖90.27→90.32%、宽度1117.4→1101.9、score1555.3→1512.4；CPLSTM覆盖90.00→90.05%、宽度1169.0→1127.6、score1735.3→1649.7。两模型score各9/9设置改善；宽度CPMLP7/9、CPLSTM9/9。设置重叠，仅描述方向一致。
- **关键局限**：主CPMLP2024原61test，relative覆盖88.5%、宽度947.4；gap尺度86.9%、968.1（原test没有同步改善）；高分歧组20块两者均15/20=75%。跨设置重分高分歧组覆盖CPMLP81.8→84.3%、CPLSTM80.8→82.8%，仍不足90%。低组约96→94%、97→96%；高组宽、低组窄且实际误差高组更大。本轮gap分组是dev20可观察d/p三等分点，不是先前曲线归一化风险组；不能混用。
- 当前距离没有稳定收益：2D life+gap RLCP h3及h6都出现无穷；h6逐电芯约0.003%–0.012%，全体平均宽度/score仍∞，未删区间。1D life RLCP h6全体有限，但比同分数全局CP更宽、score略差。不能把分歧尺度收益称为LCP距离创新。完整组宽度/score及事后真实大误差组均保存；真实误差>50%组仅事后诊断，无90%组保证。
- 三项分位数/∞测试原子/标签隔离测试通过。10854行三种旧基线与前批全部重分精确对齐；13176原test逐电芯结果算术复核、8784RLCP区间独立逐项重算、分组合计回收总体、原文件hash复核全通过，见verification.json。
- 下一步建议保留gap尺度强对照，研究更充足的建模侧诚实残差监督（proper-training内交叉拟合），而非围绕历史test继续搜带宽或复杂网络。需明确辅助模型epoch选择、OOF与最终预测器误差分布差异及最终校准隔离；OOF残差不能直接冒充最终模型的独立校准残差。尚未启动该训练。历史标签已用于发现方向，本轮也只属探索。

## 2026-10-28：已完成可利用风险信息审计（上一阶段）

- 用户授权先找预测寿命之外的风险信息，助理直接执行；未请求新的点预测器训练。本轮已完成全部18个冻结设置，无GPU训练。简报 `batterylife_main_table/isu_stability_20261028/risk_information_20261028/BRIEF.md`；完整主报告 `REPORT.md`；追加拆解 `disagreement_followup/REPORT.md`。入口 `audit_isu_risk_information.py`、`audit_isu_disagreement_followup.py`，计划均在计算各自结果前保存，追加拆解明确是看到主结果后的探索。
- 比较信息：倍率、train-PCA3曲线embedding、前100周期归一化曲线变化摘要PCA3、模型分歧、train embedding5近邻距离。非线性log预测寿命样条作对照；附加特征在train97侧扣除可拟合寿命关系再标准化。低容量风险回归Ridge alpha10预测log1p相对误差；Logistic C=.1预测相对误差>0.5。相对误差分母max(1,预测寿命)，不是MAPE，亦非直接估计条件90%分位数。
- 三模式：dev20拟合→历史pool122；pool122内按G条件5折，每折固定20块风险训练；完全相同折用全部约97块。风险CV明确使用历史cal/test标签，不能再称未用于方法开发的最终校准/测试；不是新的CP保证或独立测试。18设置重叠，不解释为独立统计复现。
- 重要发现：原划分六模型dev20均有0个>50%相对误差事件；主CPMLP2024历史pool122有9个。尾部监督不足也可能包含划分/epoch选择影响，不能只断言样本数是唯一原因。
- 较一致信号来自跨架构分歧：CPMLP三种子均值与CPLSTM三种子均值的差/当前预测寿命（log1p）；点预测中心不变。全样本风险CV相对仅寿命对照，CPMLP大误差Brier损失平均减少7.1%（8/9），CPLSTM10.6%（9/9）；log相对误差损失改善6.3%（8/9）/10.0%（9/9）。同架构seed分歧较弱。20→约97风险训练样本，跨架构特征的Brier两架构各9/9改善。绝对误差敏感性增益较小，主CPMLP的跨架构分歧绝对误差损失无改善；不泛称所有指标改善。
- 这是风险信息线索，不是LCP区间收益。下一步建议“预测寿命＋跨架构分歧”的小容量尺度/距离对照；同分数下比较全局/局部，保留relative_cp。当前计算此分歧需已有6模型推理，不能声称一对模型已验证。不改变当前ISU主底座，不盲目加复杂encoder。新方法训练/最终校准隔离方案尚未实施，不能直接拿审计CV当最终CP。
- 两项隔离/单类尾部测试通过；756行保存指标独立重算、180折电芯及G条件隔离与每电芯恰好一次评价检查通过。所有原checkpoint/清单/预测CSV/特征及读取缓存哈希保持不变。没有后台训练在本轮启动。

## 2026-10-28：后续方法实验的共同定义

- 用户要求先解释术语与入选规则，再改LCP。统一说明为 `batterylife_main_table/LCP_EXPERIMENT_SPEC.md`。简单寿命尺度CP对应 `relative_cp`，分数 `abs(error)/max(1,prediction)`；开发集拟合寿命尺度CP对应 `prediction_scale_cp`，两者不要混称。dev是通行的development/validation概念；本项目还用其残差拟合辅助尺度，属于具体建模选择，train97+dev20均为CP建模侧。
- 精确核查239的来源：官方列出240，本地pkl240、标签键240。唯一排除 `ISU-ILCC_G40C3.pkl`，发布EOL标签16、可用前缀0，未通过原加载器EOL>100和100可用前缀资格。239不是按相似曲线选出。此前“no label-based filtering”的表述过于笼统，已在文档纠正；旧冻结JSON保留原始字节。
- 当前主开发底座建议固定原ISU划分seed20261026、CPMLP训练seed2024（同一原dev中MAPE16.99%，按开发集选），路径 `isu_stability_20261028/runs/split_20261026/seed_2024/CPMLP/`。按曲线相似性扩充可作为独立候选审验，规则须训练侧固定、校准与测试来自同一预定筛选总体，不保证从823池筛出的候选必然大于239；当前尚未实际构建这种扩大后的曲线筛选集。

## 2026-10-28：有界稳定性批次

- 批次已于13:06:48完成18/18设置（约85分钟，无失败/超时），完整复核见 `isu_stability_20261028/analysis_review_20261028/ANALYSIS.md`，分析入口 `analyze_isu_stability.py`。核对全部指纹、划分、特征ID及每项4次重分的独立计算通过。
- H1有重复证据：开发寿命尺度CP相对Split，CPMLP宽度1179.9→1021.9（-13.4%）、CPLSTM1263.1→1131.7（-10.4%），平均覆盖均约90%，两模型各9/9设置宽度与interval score改善。
- H2尚不支持稳定增益：曲线尺度CP宽度比开发寿命尺度增加26.3%/19.8%，interval score平均仅改善1.9%/2.9%（分别6/9与4/9设置改善）；曲线RLCP h1.5约9%的逐电芯区间无穷，h3约1.1–1.2%，不能只报告有限子集。
- 新定义的归一化高风险组，Split覆盖约75%，开发寿命尺度约78%，直接寿命尺度约87–89%，曲线尺度约86–88%。整体90%不等于局部充分覆盖；本轮重分仅记录各组覆盖，未保存各组宽度，不据此声称条件效率改进。
- 事后风险消融：同122块池上曲线风险/预测寿命风险与归一化误差的平均rho，CPMLP0.296/0.296、CPLSTM0.346/0.330。归一化是除以指定函数，未保证已消除所有寿命效应。当前不能将正rho直接当曲线新信息。下一步建议保留全部冻结预测，先在MATR复核简单尺度基线；局部方法需关注20块开发残差的尾部估计不足，避免直接加复杂网络或按test挑带宽。

- 用户外出并授权约2到3小时实验，要求可见终端、AI不持续监听。`batterylife_main_table/start_isu_stability.ps1` 于本地11:41启动；实际阶段应读取 `isu_stability_20261028/state.json`，不要仅凭本文认定完成。
- 两个假设：尺度CP收益是否在训练初始化/电芯划分变化下重复出现；去除预测寿命尺度后，冻结模型的早期曲线表示能否为局部化提供额外信息。3划分×3训练种子×2模型=18设置，复用原来的2个模型、新拟合16个。原训练入口与旧冻结输出保持原样。
- `run_isu_stability.py` 为新入口，配置运行前写入 `isu_stability_20261028/plan.json`。train97/dev20/cal61/test61，PCA3仅拟合train编码器表示；比较固定尺度CP基线及在归一化残差上的RLCP，h=1.5和3全报告。每个冻结模型还在其122块未参与训练/开发的电芯上做200次61/61重分，不能当200个独立数据集。
- 新风险组针对归一化残差风险，与上轮绝对误差难度组含义不同。所有阈值由dev定义；不按test挑方法。两模型特征导出、冻结预测核对、RLCP常特征退化及缓存RNG隔离已检查。
- 新增训练使用共享缓存及RNG隔离；复用历史两模型保留原随机数过程。这一差别已记录，不宣称逐位可比。主批次最多3小时、单项30分钟；未完成时保留部分结果。输出 `SUMMARY.md`、`summary.csv` 每任务完成更新，详细解释见该目录README。完成后再由用户触发分析。

## 2026-10-27：ISU-ILCC 单来源、多协议 CP/LCP 首轮

- 本轮已于18:14完成，两模型各13 epoch，CPMLP最佳epoch2、CPLSTM最佳epoch5。独立复算通过，详见 `batterylife_main_table/curated_isu_pilot/analysis_20261027/ANALYSIS.md` 和 `audit.json`，分析入口为 `analyze_curated_isu.py`。Test MAPE为25.77%/28.33%；预定normalized CP分别为93.44%覆盖、976/1038周期平均宽度。CPLSTM高难度组覆盖从80%升至90%，低难度组区间收窄；这只是探索性信号。
- 核心限定：直接加倍率的RLCP出现3/2个无穷区间，没有稳定效率优势。查看结果后的诊断消融显示，单用预测寿命拟合误差尺度已解释大部分收益，倍率单独估计难度的rho仅-0.019/0.099。不能把normalized CP的改善直接算作LCP距离创新；下一步问题应是扣除简单寿命尺度后，早期曲线特征能否预测剩余误差尾部。补充消融未重新训练预测器，也不是新的确认性结果。

- 当前研究问题是：电芯相似性是否能预测残差难度，使局部区间在合理覆盖下向困难电芯分配更宽的区间。用户要求避免围绕次要元数据问题停滞，并允许在新候选集上训练表现较好的预测器；训练只需可见终端，不要求助手持续监控。
- `batterylife_main_table/curated_isu_pilot/` 固定了 239 块 ISU-ILCC 电芯，97 train / 20 dev / 61 calibration / 61 test。沿用 `protocol_candidates/isu_ilcc_t100_candidate.json` 的 seed=20261026 划分，原候选清单保持原样，实际冻结副本为 `dataset_manifest.json`。
- 新入口 `batterylife_main_table/run_curated_isu.py`；可见 PowerShell 启动器 `start_curated_isu.ps1` 串行 CPMLP、CPLSTM。train 用原 BatteryLife 1..100 前缀，dev 用 T=100 MAPE 早停（最多100、patience8），之后冻结并对 calibration/test 每电芯仅预测 T=100 一次。两模型已通过一电芯 CUDA 前向/反向和 T=100 loader smoke。
- `curated_cp_evaluation.py` 自动对比 Split CP、开发残差尺度的 normalized CP、探索性 baseLCP、预测值距离 RLCP、预测值加倍率 RLCP（Hore–Barber，arXiv:2310.07850 §3），带宽1.5、随机种子20261027；不使用尚未核实的 v4 SOC 距离。输出难度/误差相关、风险组/协议组覆盖和宽度、interval score、ESS及无穷区间数。测试标签独立性和分位数测试已通过。
- 进度与 PID 在 `curated_isu_pilot/queue_state.json`、各模型 `progress.json`；完整日志为 `CPMLP.log` / `CPLSTM.log`，完成后生成 `COMPARISON.md` 及各模型报告。是否真的启动/完成应读实时状态，不从本文推断。已有训练不会被覆盖。预测 CSV 冻结后可用 `--model CPMLP --evaluate-only` 复算区间，无需训练。
- 本轮是历史数据上的探索性试验，不能把换总体后的指标差异宣称为受控改进，也不能根据本次测试挑选模型/局部距离再宣称确认性保证。研究与使用说明见 `curated_isu_pilot/README.md`。

## 2026-10-21 execution update (current state)

This section supersedes the older statements below that downloads or large experiments have not begun; the research protocol in the 2026-10-15 section still applies.

- Full v4-equivalent `MIX_large` data are staged under `C:\BatteryLife_v4`; `batterylife_main_table/hf_download_state.json` is complete (460/460 supplemental files). Five failing Tongji downloads were repaired from the official Zenodo v4 ZIP and recorded in `zenodo_tongji_repair_manifest.json`.
- One-GPU approximate Li-ion main-table runs completed for seed 2021: DLinear test MAPE 0.57207 / 15%-Acc 0.19551; CPMLP with the v4 selected Li-ion hyperparameters test MAPE 0.16882 / 15%-Acc 0.63835. The CPMLP value is numerically close to the upstream README's 0.179 / 0.620, but one seed and local provenance differences prevent a strict replication claim. See `batterylife_main_table/PREFLIGHT.md` and the run directories there.
- Both architectures were independently retrained with zero MATR in train, validation, scaler fitting, or early stopping. Frozen checkpoints and all 169 MATR T=100 predictions are in `batterylife_main_table/runs_no_matr/20261021_113413_087865/` (DLinear) and `20261021_121422_040624/` (CPMLP). Each has `frozen_predictor.json` with SHA-256 hashes.
- The critical unresolved problem is target-domain transfer: held-out DLinear MATR MAPE 7.4075; held-out CPMLP MATR MAPE 0.7113 and RMSE 631.22 cycles. CPMLP's non-MATR source test at the same T=100 has MAPE 0.2085; a constant fitted to non-MATR train labels beats CPMLP on MATR. Do not present either held-out model as a useful LCP predictor without addressing this failure. The original MATR test was viewed earlier, so its results are exploratory.
- The initial cell-level calibration/test plan is `batterylife_main_table/matr_cal_test_split.json`: original MATR train+validation 136 cells for calibration, original test 33 cells for exploratory testing. Repartitioning these 169 MATR cells does **not** retrain the frozen base predictors; recompute calibration/interval/test outputs. Any supervised MATR point correction or localizer would require refitting on its own development set and an untouched calibration set.
- The nine-model queue finished successfully at 18:57 on 2026-10-21; all 11 `MIX_large` model classes now have one-seed, single-GPU approximate main-table results. See `batterylife_main_table/remaining_models_state.json` and the completed result table at the end of `batterylife_main_table/PREFLIGHT.md`. This does not establish MATR-held-out transfer: only DLinear and CPMLP have independently trained, MATR-excluded checkpoints.
- The earlier MATR-only 11-model seed-2024 screen already has frozen checkpoints and T=100 predictions under `batterylife_matr_benchmark/runs/20261014_222508_198380/`: 90 MATR train cells fit each model, 12 other train cells select its epoch, 34 official validation cells can calibrate, and 33 official test cells evaluate. `evaluate_frozen_split_cp.py` used those files without retraining; its first 90% Split CP output is `batterylife_matr_benchmark/frozen_split_cp/20261021_193212_072306/`. Test labels have been seen, so conclusions remain exploratory. Reusing all 136 non-test MATR cells for calibration is incompatible with training the predictor on MATR; the two protocols answer different questions.

更新时间：2026-10-15。**请以本节为当前任务；后面的 2026-09-02 内容是历史方案，不再作为下一步执行指令。** 本轮只更新交接文档，尚未启动下述大型实验。不要修改、移动或删除任何旧实验目录。

## 现在要回答的研究问题

主问题仍是：固定前 `T=100` 个周期、每块电芯预测一次 EOL 时，能否在合理的点预测器上，通过局部化 conformal prediction（LCP）得到有用的自适应区间。导师希望先提高/核准预测器质量，再回到 learnable localizer，而不是无限增加 kernel 或直接放弃 CP。

**接下来分两个不同实验，不能混为一谈：**

1. **BatteryLife 主表口径复现。** 用本机已在 MATR-only 脚本中跑通的 11 个原始模型类，尽量按 BatteryLife 原始 `MIX_large`（混合 Li-ion）训练/评估协议复现其 Li-ion 主表，核对 MAPE 和 15%-Acc。先审计数据版本、所需数据集、划分、前缀采样、预处理、训练超参数、种子和指标。若无法严格一致，明确标注“近似复现”和每处差异；不要拿 MATR-only RMSE 与主表 Li-ion MAPE 直接比较。
2. **目标域仅 MATR 的固定 T=100 实验。** 从头训练**完全不含 MATR 电芯**的预测器（可沿用步骤 1 确认的架构和训练口径），冻结后在 MATR 上做点预测及 CP/LCP；MATR 只进入校准与测试，不进入预测器的训练、验证、早停、标准化拟合或超参数选择。步骤 1 的 `MIX_large` 模型含 MATR 训练电芯，**不能直接拿其权重**用于步骤 2。

步骤 1 可先从一个模型、一个种子打通全链路，再扩大到 11 个；不强求所有模型都能在 8 GB GPU 上完成混合数据训练。步骤 2 在完成预测器口径和外部数据准备后再实施；本轮不要直接启动。

## 已确认的事实与现有结果

- 工作区 `D:\Battery-LCP`；环境 `my_torch`，本机约 8 GB GPU；已处理数据根目录 `D:\BatteryLife_Processed` **目前只有 MATR、HUST 和 Life labels**。完整 `MIX_large` 所需其他 Li-ion 子数据集尚未在本地，必须先按原仓库清单核对/补齐，不能把 HUST+MATR 冒充论文混合 Li-ion 主表。
- BatteryLife 原仓库：`third_party/BatteryLife/`。原始入口 `run_main.py`、加载器 `data_provider/data_loader.py`、划分 `data_provider/data_split_recorder.py`、各模型脚本 `train_eval_scripts/*.sh`。`MIX_large_{train,val,test}_files` 在划分文件中分别拼接多个 Li-ion 子数据集，**包括 MATR**。原加载器会为同一电芯构造多个早期周期前缀，它们共享同一个 EOL；train/cal/test 的独立单位必须仍是电芯。
- 原仓库 README 的 Battery Life Prediction 主表列是 **Li-ion / Zn-ion / Na-ion / CALB**，各报 **MAPE、15%-Acc**；它没有单列 MATR 的主表结果。训练的样本单位、混合数据、早停与我们固定 T=100 的一电芯一样本实验不同。优先核对 Li-ion 列及论文/仓库对应版本，不能据主表直接判断 MATR-only 的 RMSE 是否“合格”。
- 现有 MATR-only 比较：`simple_cptransformer_cp/batterylife_matr_benchmark/README.md`、`run_matr_models.py`、`run_cv.py`。此脚本复用 BatteryLife **未修改的模型类**，但使用我们已审计的固定 T=100 曲线适配器，每块 MATR 电芯仅一条样本；**不是**原始多前缀训练协议，也不是主表复现。官方 MATR 划分为 train 102 / validation 34 / test 33。
- 已在上述本机筛选脚本跑通的 11 个模型：`DLinear, MLP, CPMLP, PatchTST, iTransformer, CPTransformer, CNN, CPGRU, CPBiGRU, CPLSTM, CPBiLSTM`。一模型种子、包含探索性官方集预测的结果在 `batterylife_matr_benchmark/runs/20261014_222508_198380/results.json`；例如 CPLSTM 的 MATR test RMSE 约 213.22 cycles、MAE 约 133.19、MAPE 约 13.57%，**不是**论文 Li-ion 主表数字。
- 已做的 MATR train 内部嵌套五折：`batterylife_matr_benchmark/nested_cv_runs/`；具体批次和局限见该目录 README。`20261014_222028_539246/summary.json` 的模型种子 2024 显示 CPLSTM OOF RMSE 约 183.78、CPMLP 约 218.30；种子 42/2021 仅部分模型完成。`cv_runs/` 早期版用外层评价折选择 epoch，**不能用于排名**。这些都是固定 T=100、MATR-only 的开发期证据。
- 此前更完整的 LCP、可学习表示、HUST、自监督、ensemble 结论见 `RESEARCH_PROGRESS_SUMMARY_20260930.md` 及后续独立实验目录；不能因为更新交接就覆盖旧结果。官方 MATR test 以前已经被查看，今后凡使用它的结论都标记为**探索性验证**。
- 本机筛选脚本未跑通其他 7 个原始模型的原因见 benchmark README：Autoformer 显存不足；Transformer 需要未由原入口提供的 decoder 输入；MICN 在 T=100 存在特征维度不匹配；GRU/BiGRU/LSTM/BiLSTM 在 30,000 步序列上不现实。不要把这误写为“缺 accelerate 包”。

## 新聊天的建议执行顺序

1. 先完整读本文件当前节、`batterylife_matr_benchmark/README.md`、BatteryLife 原仓库 README 和对应源代码/脚本。只做必要的口径审计：论文表格的训练数据到底有哪些，当前 processed 数据的缺口、版本和下载成本，以及原始 `run_main.py` 在本机运行所需依赖。**先列清单，再下载/训练**；保持产物最少。
2. 主表复现至少对齐原始电芯划分、多前缀生成、输入预处理、损失、early stopping 指标、模型脚本配置、种子和 MAPE/15%-Acc 计算。优先使用官方加载器/训练入口，若为本机做轻量适配，要明确说明差异。11 个模型是目标清单，不保证都能在混合数据规模和显存限制下跑通；遇到失败记录原因，不伪造对照。先试一个种子，再按可行性扩展。模型比较和论文值都应报告对应的数据、seed 与版本。
3. 随后独立建立 **MATR-held-out predictor**：从所有混合训练/验证列表中剔除 MATR，确认其他数据可用；不复用含 MATR 权重的模型。冻结模型后，MATR 以电芯级单样本固定 T=100 评估。要同时报告非 MATR 验证误差和 MATR 目标域误差，防止主表复现成功却在跨数据集迁移时失效。
4. 固定域外训练方案和 CP/LCP 方法后，一个可操作的首版是：MATR 原 train+validation 共 136 块作为 calibration，原 test 33 块作为探索性 test；如要用 MATR 标签选择模型/localizer/核带宽，应先从 136 块中**另留开发集**，不能一边用这些标签调参一边当作未触碰 calibration。所有电芯 ID 和 protocol/batch 信息尽可能保存。对 11 个模型可先全部报告，**不要按 MATR test 最优挑模型**。
5. 对每个冻结预测器，同一批 MATR 电芯、同一固定 T=100 输入，先比较 Dummy/点预测 RMSE、MAE、MAPE，再比较普通 Split CP 与预先指定的 LCP：覆盖率、上下侧漏覆盖、均值/中位宽度、interval score、ESS 和逐电芯宽度。LCP localizer 若需有监督学习，须有独立开发残差，不能用同一 calibration 残差既训练 localizer 又声称未经调整的 CP 保证。
6. **交换性边界要如实陈述。** 预测器可以在外部数据上训练；Split CP 的目标域覆盖解释依赖 MATR calibration/test 电芯在目标分布下可交换，随机分电芯并非对采集批次/protocol 差异的自动证明。MATR 约 169 块、81 种 protocol；分析目标是现有 MATR 电芯总体，而非未来全新 protocol。可做 protocol/batch 分层误差和覆盖诊断，但不要把它误写成跨 protocol 保证。

## 给新聊天直接使用的开场提示

> 请先完整阅读 `D:\Battery-LCP\simple_cptransformer_cp\CONTEXT_HANDOFF.md` 顶部 2026-10-15 更新，以及其中指向的 benchmark README、BatteryLife 原仓库 README 和关键代码。先审计复现 BatteryLife Li-ion 主表所需的数据/依赖/训练口径，确认本机已有数据缺口；之后用 11 个已在 MATR-only 脚本跑通的模型做尽可能忠实的主表复现。再单独训练完全不含 MATR 的预测器，固定 T=100，只在 MATR 上校准和测试 CP/LCP。不要把 MIX_large 含 MATR 的 checkpoint 用于 held-out 实验；不要覆盖或删除旧实验；先给我简短口径与数据清单，再开展较大的下载或训练。官方 MATR test 已被看过，最终只可称探索性验证。

---

## 以下为 2026-09-02 历史交接，仅供理解最初固定 T=100 设计

# 新聊天交接：BatteryLife CPTransformer + Split CP/basic LCP

更新时间：2026-09-02

## 1. 研究背景

目标是研究电池寿命预测中的 conformal prediction，数据主要来自 BatteryLife 的 MATR 数据集。此前已经实现过逐周期 RUL 预测、多个固定时刻校准和多种 learnable kernel，但实验链路逐渐复杂，且区间结果不理想。现在决定保留全部旧内容，另建本目录，从最简单、最接近 BatteryLife 原始任务的版本重新开始。

## 2. 已确认的概念

### 电池任务不是普通的一步时间序列预测

电池寿命标签由最终失效时刻 EOL 决定。在电芯尚未达到 EOL 时，真实 RUL `EOL - t` 不可观测，因此测试阶段不能像普通一步时序预测那样逐步获得真实 residual。

统计独立单位必须是电芯。同一电芯的多个循环、窗口或前缀共享同一个 EOL，不能被当成独立 calibration 样本，也不能跨 train/calibration/test 划分。

### BatteryLife 原始 CPTransformer 任务

BatteryLife 原仓库使用已完成寿命试验、已知 EOL 的训练电芯，学习早期充放电曲线与总寿命之间的关系，再预测未参与训练的测试电芯 EOL。它不是在单块电芯上预测下一循环的容量、电压等参数。

原始 CPTransformer 的主要输入是早期循环内经过重采样的完整电压、电流和容量曲线。数据加载器可以为同一训练电芯生成多个不同可见前缀，但这些前缀的标签均为该电芯的同一个 EOL。

### 旧的 CPTransformer-RUL adapter 不是原始 CPTransformer 复现

旧代码 `battery_lcp/run_cptransformer_kernels.py` 使用每循环统计量序列，并把 BatteryLife 的模块改造成因果逐周期 RUL 输出模型。它是适配器，不应被称为原始 CPTransformer baseline。

旧代码和结果必须保留，不要移动、覆盖或删除。

## 3. 新阶段的唯一任务

固定观测时刻：

```text
T = 100 cycles
```

输入：每块电芯前 100 个循环的完整充放电曲线。

标签：总循环寿命 EOL。

点预测：

```text
EOL_hat = CPTransformer(X_1:100)
```

按顺序比较四种结果：

1. CPTransformer 点预测；
2. CPTransformer + Split CP；
3. CPTransformer + asymmetric Split CP；
4. CPTransformer + basic embedding LCP。

暂不加入：

- 多个 landmark；
- 逐周期 RUL 输出；
- learnable kernel；
- Diagonal-state、Low-rank joint、Quantile-profile、Spectral-band；
- CQR/LCQR；
- HopCPT/KOWCPI；
- 第二数据集和复杂消融。

## 4. 预期数据划分

优先沿用 BatteryLife MATR 的官方电芯划分，并从官方 train 电芯中留出 model validation：

- Train：约 90 块电芯；
- Model validation：约 12 块电芯；
- Calibration：官方 validation，约 34 块电芯；
- Test：官方 test，约 33 块电芯。

实际数量以成功读取、具有至少 100 个有效循环且具有 EOL 标签的电芯为准，必须在输出中记录排除原因和最终 cell ID。

模型选择只能使用 model validation。Calibration 标签不能用于选择 predictor epoch、训练 embedding、学习带宽或其他模型参数。

## 5. Split CP 定义

对 calibration 电芯：

```text
s_i = abs(EOL_i - EOL_hat_i)
```

使用有限样本修正的 conformal 分位数得到 `q`，测试区间：

```text
[EOL_hat - q, EOL_hat + q]
```

初始目标覆盖率可用 90%（`alpha=0.1`），但它是实验设置而不是固定结论，应在配置文件中明确记录。

如果报告固定 `T=100` 的 RUL：

```text
RUL_hat = EOL_hat - 100
RUL_interval = EOL_interval - 100
```

二者覆盖率和区间宽度完全相同。

### 非对称 Split CP

定义有符号残差 `r_i = EOL_i - EOL_hat_i`，分别校准两个方向：

```text
q_lower = conformal_quantile(-r, 1 - alpha_lower)
q_upper = conformal_quantile( r, 1 - alpha_upper)
interval = [EOL_hat - q_lower, EOL_hat + q_upper]
```

初始使用等尾分配 `alpha_lower = alpha_upper = 0.05`。这样，大正残差主要影响上界，不再让下界同步扩张。尾部比例不能根据最终 calibration/test 结果选择；以后如需优化，必须使用 Train 的 cell-level OOF residual 或独立 model-validation 数据。

当 calibration 约有 34 块电芯时，等尾设置的每个单侧分位数秩为 34，即相应方向的 calibration 最大误差。这是有限样本下等尾保证带来的保守性，需要与对称 Split CP 的覆盖率和宽度一起报告。

## 6. basic LCP 定义

先冻结只用 Train/Model-validation 训练和选择的 CPTransformer。从模型倒数层提取 calibration/test embedding。

最初只采用一个透明的固定距离方法：

1. embedding 仅用 Train 数据拟合的变换进行标准化；
2. 如维度过高，可使用只在 Train embedding 上拟合的 PCA；
3. 使用欧氏距离；
4. 使用 Gaussian/RBF 权重；
5. 对 calibration absolute residual 求局部加权 conformal 分位数；
6. 输出 ESS，并在 ESS 太小时回退到 Split CP。

不要把 calibration/test EOL 用于训练 embedding、PCA 或距离模型。若带宽需要调参，应使用 Train 的 cell-level OOF residual 或预先固定规则，不能根据 test coverage 选择。

## 7. 第一版评价指标

点预测：

- RMSE；
- MAE；
- MAPE；
- 可选 `R^2`。

区间：

- empirical coverage；
- mean width；
- median width；
- interval score/Winkler score；
- LCP ESS 最小值、中位数和均值；
- 每个 test 电芯的预测与区间 CSV。

由于 test 约 33 块，coverage 的粒度约为 `1/33 = 3.03%`。例如 90.91% 表示覆盖 30/33，并不自动代表效果差；必须与目标覆盖率、宽度和有限样本不确定性一起解释。

## 8. 本地路径与旧材料

- 工作区：`D:\Battery-LCP`
- 数据：`D:\BatteryLife_Processed`
- BatteryLife 仓库：`D:\Battery-LCP\third_party\BatteryLife`
- 原始 CPTransformer：`third_party/BatteryLife/models/CPTransformer.py`
- 原始数据加载：`third_party/BatteryLife/data_provider/data_loader.py`
- 官方训练脚本：`third_party/BatteryLife/train_eval_scripts/CPTransformer.sh`
- 旧适配实验：`battery_lcp/run_cptransformer_kernels.py`
- 旧总体技术日志：`TECHNICAL_LOG.md`
- 旧实验设计：`CPTRANSFORMER_KERNEL_DESIGN.md`
- 旧残差分析：`RESIDUAL_AUDIT.md`

旧材料仅用于参考，不要在第一版最小实验中把复杂模块重新带进来。

## 9. 新聊天应先完成的工作

1. 审计原始 BatteryLife CPTransformer 的输入张量、mask、标签和固定 `T=100` 的评估方式；
2. 明确是否直接复用原始 Dataset/Model，尽量少改原代码；
3. 在本目录内写出最小实验设计和可运行入口；
4. 先跑点预测并核对与 BatteryLife 官方任务的一致性；
5. 再加入 Split CP；
6. 最后加入单一 basic embedding LCP；
7. 每一步完成后运行、保存配置、划分、预测和指标，并记录异常。

实施过程中保持本目录简洁，不修改和清理旧目录。
