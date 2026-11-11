# RL training log


<!-- fullmap_4096_e4_interrupted_20261107 -->
## 全尺寸大预算基准 v10（预算切换归档，2026-11-07）

旧1237301独立网络，8×H200，每卡4096真实对局、双座current、T512/E4/M64、physical1024、seed2026102901。按用户要求在U60保留checkpoint后停止，改用2048对局/卡及T512/E1/M256，所有快速消融从头训练。

最后记录U60，fresh玩家样本2,013,265,920；固定面板U20/U40平均均0.375%，aggressive 0.5%，expansion 0%，defensive 1%，random 0%。U60评估未由训练父进程完成登记，不计入曲线。近期约36,900玩家样本/s，单轮约909s（采样69s＋PPO840s）。这是中断归档，未完成224轮。模型保留在H200旧实验runs/v10_full_joint_ref/model_60.pt，源码和原始日志保留。

[旧报告](fullmap_t512_ablation_20261106.md)；[快速消融](fullmap_t512_fast_ablation_20261107.md)。

![旧基准曲线](assets/fullmap_t512_ablation_20261106/learning_curves.png)


<!-- fullmap_e1_interrupted_for_map9_20261107 -->
## 全尺寸E1基准 v10（切换9×9归档，2026-11-07）

8×H200，每卡2048真实对局、双座current、T512/E1/M256、physical1024、seed2026102901。按用户要求停止并改为9×9全组重训。最后记录U101，fresh玩家样本1,694,498,816；U100 checkpoint保留。最新固定面板U100平均0.375%、aggressive 0.50%。近期每轮约144s（采样36s＋PPO108s），约116,000玩家样本/s。未完成224轮。

[全尺寸归档](fullmap_t512_fast_ablation_20261107.md)；[新9×9消融](map9_t512_ablation_20261107.md)。

![曲线](assets/fullmap_t512_fast_ablation_20261107/learning_curves.png)


<!-- map9_t512_20261107:v10_9x9_joint_ref -->
## v10_9x9_joint_ref

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=joint_independent，gamma=default，reward=constant，保留比例=1.0。

最佳固定面板U224：平均97.500%，aggressive 92.50%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v10_9x9_joint_ref_curve.png)


<!-- map9_t512_20261107:v11_9x9_early_shared -->
## v11_9x9_early_shared

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=early_shared，gamma=default，reward=constant，保留比例=1.0。

最佳固定面板U200：平均99.250%，aggressive 98.00%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v11_9x9_early_shared_curve.png)


<!-- map9_t512_20261107:v12_9x9_post_attention -->
## v12_9x9_post_attention

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=post_attention，gamma=default，reward=constant，保留比例=1.0。

最佳固定面板U160：平均99.125%，aggressive 97.50%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v12_9x9_post_attention_curve.png)


<!-- map9_t512_20261107:v13_9x9_gamma1_fixed -->
## v13_9x9_gamma1_fixed

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=joint_independent，gamma=fixed1，reward=constant，保留比例=1.0。

最佳固定面板U40：平均4.125%，aggressive 0.00%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v13_9x9_gamma1_fixed_curve.png)


<!-- map9_t512_20261107:v14_9x9_gamma1_ramp -->
## v14_9x9_gamma1_ramp

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=joint_independent，gamma=ramp1，reward=constant，保留比例=1.0。

最佳固定面板U180：平均92.000%，aggressive 82.00%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v14_9x9_gamma1_ramp_curve.png)


<!-- map9_t512_20261107:v15_9x9_pbrs_decay -->
## v15_9x9_pbrs_decay

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=joint_independent，gamma=default，reward=decay，保留比例=1.0。

最佳固定面板U224：平均97.250%，aggressive 92.50%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v15_9x9_pbrs_decay_curve.png)


<!-- map9_t512_20261107:v16_9x9_sparse -->
## v16_9x9_sparse

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=joint_independent，gamma=default，reward=sparse，保留比例=1.0。

最佳固定面板U224：平均99.125%，aggressive 97.00%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v16_9x9_sparse_curve.png)


<!-- map9_t512_20261107:v20_9x9_gamma1_pbrs_decay -->
## v20_9x9_gamma1_pbrs_decay

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=joint_independent，gamma=fixed1，reward=decay，保留比例=1.0。

最佳固定面板U40：平均4.500%，aggressive 0.00%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v20_9x9_gamma1_pbrs_decay_curve.png)


<!-- map9_t512_20261107:v21_9x9_gamma1_sparse -->
## v21_9x9_gamma1_sparse

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=joint_independent，gamma=fixed1，reward=sparse，保留比例=1.0。

最佳固定面板U220：平均97.250%，aggressive 93.50%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v21_9x9_gamma1_sparse_curve.png)


<!-- map9_t512_20261107:v17_9x9_top25 -->
## v17_9x9_top25

9×9快速消融＋课程，T512/E1/M256、双座current，seed2026102901，224轮；2048真实对局/卡，physical=8192。网络=joint_independent，gamma=default，reward=constant，保留比例=0.25。

最佳固定面板U224：平均88.625%，aggressive 72.00%。这是训练面板选模结果；独立评估见[实验报告](map9_t512_ablation_20261107.md)。

![curve](assets/map9_t512_ablation_20261107/v17_9x9_top25_curve.png)

<!-- sparse_features_20261109 -->

## v24–v29: early_shared / sparse / gamma=1

[Configurations and changes](sparse_features_20261109.md); [live/final results](sparse_features_20261109_results.md).

| Run | Updates | Latest original4 | Latest aggressive | Latest all7 | Steady player/s | Peak allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| v24_9x9_sparse_baseline | 224/224 | 90.3% | 93.0% | 85.3% | 441,906 | 21.02/22.37 |
| v25_9x9_sparse_hl128 | 224/224 | 98.5% | 95.5% | 91.6% | 447,148 | 21.02/22.28 |
| v26_9x9_sparse_binary_ce | 224/224 | 96.8% | 94.5% | 91.4% | 444,812 | 21.02/22.37 |
| v27_9x9_sparse_compact | 224/224 | 95.2% | 96.5% | 89.4% | 437,300 | 18.65/19.90 |
| v28_9x9_sparse_temporal | 224/224 | 92.9% | 95.5% | 86.5% | 425,359 | 19.35/20.69 |
| v29_9x9_sparse_dedup | 224/224 | 80.9% | 90.0% | 74.1% | 434,269 | 19.35/20.55 |

![Curves](assets/sparse_features_20261109/learning_curves.png)

<!-- /sparse_features_20261109 -->

<!-- late_decline_audit_20261110 -->

## 2026-11-10：最近一周后期衰退的胜／平／负拆分

结论：近期 v12、v14、v24、v28、v29 在目标地图评估上的明显衰退，均以平局增加为主，但不是所有性能衰退都属于这种情况。11 月 3 日地图课程在旧布局上的退化主要是败局增加；同日全尺寸 packed run 的平局减少、败局增加。单看总胜率不能区分这些现象，也不能由平局增加直接推导训练根因。

检查范围为 11 月 3–10 日的本地实验档案：34 份现代 run 的 `evaluation_curve.json`，另加 11 月 3 日课程官方图、课程旧图、直接官方训练及 full_packed 的四组评估曲线。课程官方图和旧图是同一 run 的两个评估分布，不计成两个 run；旧布局训练后补测官方图的 control 训练早于本窗口，未计入。短暂初始化或尚未学到有效策略就停止的尝试不当成“高胜率后崩溃”。

主表统一采用原四 baseline、每 checkpoint 800 局、采样动作、各自固定地图与交换座位。峰值选可用训练后半程内最高胜率，比较最后一个完整评估点；下降至少 3 个百分点作为人工复核筛选条件，不是统计显著性标准。所有主表配对均逐局核对评估配置、地图种子、初始地图 digest、双方 controller seeds 一致，原始局数与汇总完全相符。平局原因均为 `tick_limit`，现代报告无评估错误。

| Run／评估分布 | 更新区间 | 胜率 | 平局数 | 败局数 | 净变化解释 |
|---|---:|---:|---:|---:|---|
| v12 post_attention | 160→224 | 99.125%→87.125% | 1→74 | 6→29 | 少赢96局；平局+73，败局+23 |
| v14 gamma 0.99→1 | 180→224 | 92.000%→35.500% | 17→471 | 47→45 | 少赢452局；平局+454，败局−2 |
| v24 sparse baseline | 140→224 | 97.375%→90.250% | 7→65 | 14→13 | 少赢57局；平局+58，败局−1 |
| v28 temporal | 180→224 | 97.625%→92.875% | 6→48 | 13→9 | 少赢38局；平局+42，败局−4 |
| v29 temporal+dedup | 160→224 | 94.125%→80.875% | 27→139 | 20→14 | 少赢106局；平局+112，败局−6 |
| 11/03 地图课程：旧布局面板 | 120→224 | 98.500%→92.250% | 1→9 | 11→53 | 少赢50局；败局+42，占净回落84% |

这里是总数的净变化，不代表每个旧胜局只会转为某一种结果。逐局配对确认：v12 有74局胜→平、28局胜→负，同时5局负→胜、1局平→胜；v14 有450局胜→平、29局胜→负，同时也存在负→胜等反向变化。旧布局回归面板则有49局胜→负、9局胜→平，7局负→胜、1局平→胜。

v24 本表按四 baseline 峰值选 U140；此前七对手选模采用 U180，因此与此前“778胜／11平／11负”口径不同，并非数据变更。v29 四 baseline 峰值 U160，而七对手峰值为 U180。

反例与边界：

- 11/03 课程 run 的官方目标面板最终仍为98.75%，旧布局面板却从98.50%降至92.25%。这支持分布迁移／遗忘的解释；不能称为该模型在所有地图上都崩溃。
- 11/03 full_packed 从 U100 的13胜／311平／476负，变为最后完整基线评估 U180 的6胜／198平／596负。平局减少113局，败局增加120局；属于一直低胜率、后期生存表现变差。训练后来停止，不能把 U180 标成训练最终 checkpoint 或 U224。
- 11/04 full-map T512 的 U110→112 为160胜／278平／362负→78胜／342平／380负，平局是主要净变化，但只有停止前最后一个回落点，不能确认持续性后期坍塌。
- 同批 patch3 的 U220→224 为79.375%→77.500%，平局35→34、败局130→146，是败局增加。幅度仅1.875个百分点且只跨最后4轮，应作为末轮波动观察，不与 v14 的56.5个百分点坍塌等同。
- v13／v20 曾有很高平局率，但从未形成高胜率阶段；不能混同为“先学好、后退化”。

独立种子补测也复现主要现象。下表每点1600局，使用既有 best/final heldout；新 sparse study 的 best 按七对手选取，与主表四对手峰值可不同。

| Run | 1024 tick：胜／平／负，best→final | 2000 tick：final 胜／平／负 |
|---|---|---|
| v12 | 1583／0／17 → 1367／170／63 | 1368／168／64 |
| v14 | 1499／16／85 → 597／913／90 | 600／908／92 |
| v24 | 1538／40／22 → 1442／124／34 | 1449／117／34 |
| v28 | 1575／12／13 → 1478／94／28 | 1487／85／28 |
| v29 | 1502／63／35 → 1302／261／37 | 1317／244／39 |

延长到2000 tick并未恢复高胜率，不能只解释为1024 tick评估过短。独立面板上 v24／v28 的败局也增加，说明“全由平局造成”的字面结论过强；平局仍然贡献了大部分净胜局损失。

七对手面板结论一致：各点1400局，从各自七对手峰值到U224，v24 少赢79局，其中平局+72、败局+7；v28 少赢55局，平局+72、败局−17；v29 少赢168局，平局+179、败局−11。v27 末轮从92.357%降到89.357%，平局+30、败局+12，但仅跨U220→224。v25／v26 的七对手回落分别仅0.071／0.571个百分点，不能据此称为明显崩溃。

这次审计仅定位结果类型，未证明优化根因。v12 使用 PBRS、gamma 最终0.999；v14 使用 PBRS、gamma最终1；v24／v28／v29 使用 sparse、gamma=1。因此平局型衰退既不是 sparse 独有，也不能全部归咎于 gamma=1。后续值得检查达成终局的行为、截断 bootstrap 与 critic 偏差，但需要行为和训练指标的对应证据。

![最近一周衰退run的胜平负曲线](assets/late_decline_20261110/wdl_curves.png)

可复现分析：`outputs/analysis/late_decline_audit_20261110_actual01/analyze.py`；完整38条评估曲线汇总 `all_panels.csv`，逐局转移矩阵、配置配对检查及补测数据在 `audit.json`。全部为离线读取既有结果；未启动训练或新对战。

<!-- /late_decline_audit_20261110 -->

<!-- terrain_distance_20261110 -->

## v30–v32：temporal＋HL128，时限与路径消融

[配置与改动](terrain_distance_20261110.md)；[动态结果与曲线](terrain_distance_20261110_results.md)。

| Run | PPO轮 | 原时限 | 原时限七对手 | 统一2048四基线 | 统一2048七对手 | 玩家样本/s | 峰值allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| v30_9x9_temporal_hl128 | 5/224 | 2048 | — | — | — | 170,653 | 19.35/20.56 |
| v31_9x9_temporal_hl128_4096 | 0/224 | 4096 | — | — | — | — | 0.00/0.00 |
| v32_9x9_temporal_hl128_paths | 0/224 | 2048 | — | — | — | — | 0.00/0.00 |

![统一2048 tick曲线](assets/terrain_distance_20261110/curves_cap2048.png)

<!-- /terrain_distance_20261110 -->
