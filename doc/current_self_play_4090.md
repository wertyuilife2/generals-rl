# 双座 current self-play：实现与 RTX 4090 对照

日期：2026-11-05。本实现位于独立 worktree `outputs/worktrees/current-selfplay-dual`，分支 `perf/current-selfplay-dual`；原`codex_dev_clean`的代码与配置未改，测试产物保存在主工作区忽略的`outputs/`目录。

本页记录`60e5d33`的独立Actor／Critic基准；后续共享主干、可选特权encoder和priv11布局见[当前RL说明](../rl/README.md)。下述历史测量保持其原始网络配置。

## 实现范围

基线为 `123730107ac1364f7f195a266034316f57939a28` 加任务开始时工作区的未提交改动，包括stride=3支持及地图课程。逐文件 SHA256、原始 diff 与完整源码冻结在主工作区的 `outputs/analysis/current_selfplay_dual_4090_20261105/`。对比使用相同网络与 MDP，实际patch_size均为1。

- `task.train_both_sides=true` 仅允许 current self-play。一个真实对局向 RSL 提供两条固定身份轨迹，排列为 `(game0,seat0),(game0,seat1),…`；原生 CPU/CUDA 后端仍只执行一次联合动作。
- 同一actor一次处理双方观测，内部仍按inference chunk控制内存。双方各自的H4、K11、mask、old logprob、reward、value与GAE独立；每条PPO样本只使用本座动作的概率，不把联合动作概率当作单座logprob。不会将对手/critic的隐藏信息提供给actor。
- K11 仍只保存 `真实对局数 × 2座位`，不为两条训练轨迹再各复制两座知识。每对玩家同时 reset，timeout 保存各自 reset 前的终局观察做 bootstrap。
- student-only 采样直接复用上游 `PPO.act`，去掉 CTS 的空 teacher 分支及整批索引/拼接。现有 CTS timeout 与 GAE 实现继续复用，不改算法公式。
- H4 push/reset 采用替换存储；训练观察可引用不可变视图，去掉每次构造观察的大块复制，仍保证旧 transition 与 terminal observation 不被 reset 改写。
- vendor runner 增加4行可选释放：`release_terminal_observations=true`时，在timeout bootstrap和每步日志均消费完毕后从extras移除终局观察引用，避免把旧H4保留至PPO更新。vendor默认false，Generals默认true；基准采用相同时序，修改记录在[PATCHES.md](../rl/vendor/rsl_rl/PATCHES.md)。弱引用测试在update边界检查实际释放，不只检查训练返回后的状态。
- 参数独立的 actor/critic 逐个反向，避免同时保留两套激活。逻辑 batch 的梯度全部累积后才统一裁剪/Adam；检测到共享/外部可训练参数时保留原联合反向。没有减少 PPO epoch 或更新样本。
- pure current 不再创建 `history_pool/`、冻结模型实例或逐轮发布权重。评测继续独立按 update 导出已有 checkpoint；默认训练 checkpoint 保留2个，评测保留最近2个及top3候选，保留报告而非所有权重。
- 历史 self-play 和原单座 current 作为对照仍可使用：历史模式设置 `opponent="self_play", train_both_sides=false`；单座 current 设置 `train_both_sides=false`。

## 计数与默认预算

`task.num_envs` 是每 rank 的真实对局数，`task.num_envs×2` 才是双座模式提供给 RSL 的轨迹数。TensorBoard/summary/benchmark 分别报告玩家 transitions 与棋盘 ticks，完成 episode 的计数按真实对局计算，地图课程按棋盘 ticks 推进。

双座来自同一对局，存在相关性；样本条数翻倍不等于独立对局数或训练速度翻倍。训练`Episode/win`同时包含赢家与输家，自然终局平均通常为0.5，不能用来衡量棋力；继续使用独立baseline评测的胜率/Elo。

worktree 默认8192局×2座、T32，每轮524288条玩家样本；E4、64个逻辑minibatch、每batch8192，与旧默认16384单座的玩家样本及Adam预算相同。八卡224轮仍为939524096条fresh玩家样本，但只需469762048个真实棋盘tick。物理microbatch512、推理chunk1024用于4090内存控制；H200上的最佳执行尺寸未重新标定。模型仍是参数独立的ViT1M actor/critic，没有改成共享Transformer、patch3或分类价值头。

H4的reset存储实现随`policy/`源码一起固定，因此本次策略源码哈希会变化。旧run的严格`--resume`继续要求其原有固定源码；若迁移旧权重到本worktree的新训练配方，用`--warm-start`，保留兼容的网络权重并重新建立optimizer。双座版本自身的checkpoint恢复已通过回归，不绕过源码校验。

独立worktree的共享venv没有重新editable安装。运行时必须将该worktree四个包和vendor放到`PYTHONPATH`前部，避免导入主工作区：

```bash
cd /home/yuhangwang/code/generals-rl/outputs/worktrees/current-selfplay-dual
PYTHONPATH=rl/src:env/src:baselines/src:arena/src:rl/vendor/rsl_rl \
GENERALS_NVCC=/home/yuhangwang/miniforge3/envs/generals-rl/bin/nvcc \
/home/yuhangwang/code/generals-rl/.venv/bin/python -m generals_rl.train \
  --config rl/configs/current-dual-4090.json \
  --iterations 2 --output outputs/rl/current-dual-smoke-new
```

## 正确性验证

吞吐测试时的完整RL CPU回归：412 passed、15 skipped、3157 subtests passed（153秒）；CPU回归显式隐藏CUDA，GPU专项另测。包含双rank的双座训练/全局梯度一致性、resume、部分预算、rank-zero文件归属及终局观察释放；旧历史模式继续通过。

双座专项使用真实引擎和相同联合动作，将两条训练轨迹分别与两个独立单座task逐项比较，检查局部/特权观察、PBRS与胜负奖励、终局和部分reset。另验证timeout的reset前value bootstrap、GAE不会串到另一座/另一局、无池checkpoint恢复。microbatch既有测试比较完整loss、梯度、Adam、学习率、梯度裁剪次数及RNG。

CUDA专项7项通过：CPU/CUDA双座轨迹与不可变观察、masked BF16输出/梯度、编译反向及冻结推理、两种patch尺寸的GPU导出一致性。

最终v3 runner在4090真实短训完成4轮，128真实对局、T8、E4、BF16/compile；horizon3专门覆盖频繁timeout/reset。核验8192条玩家transitions、4096个棋盘tick、1280个完成对局（均timeout）、有限loss/权重、无history_pool、只保留`checkpoint.pt/model_3.pt`两个文件。TensorBoard最后一条players/boards计数相符；见[gpu_smoke_v3_verification.json](../outputs/analysis/current_selfplay_dual_4090_20261105/gpu_smoke_v3_verification.json)。原v1短训及记录另行保留。

该实际双座checkpoint也已导出为actor-only策略包，通过Arena `validate_bundle`及固定源码校验，见[gpu_smoke_v3_export_verification.json](../outputs/analysis/current_selfplay_dual_4090_20261105/gpu_smoke_v3_export_verification.json)。独立评测不依赖训练快照池。

基准源码的CPU日志为`cpu_full_terminal_release.log`；此前基准工具修复后另有17项相关回归通过。未进行长期棋力验收，不能从吞吐结果推断胜率。

## 4090测试协议与结果

硬件为RTX4090 24GB；PyTorch2.13.0+cu130。官方18–23随机长方形地图、23×23画布、ViT1M、patch1、masked BF16/compile、H4/K11、T32、E4、逻辑batch8192。双方都保持相同的奖励、网络、gamma/学习率课程、存储codec和历史去重。

4090正式对照选用可完成完整PPO的8192单座轨迹，并且两边统一physical minibatch512/inference chunk1024；未将原H200默认16384局、physical4096直接当作本卡正式结果，也不宣称这次测试确定了原实现的全部容量上限。

统一地图池512，只缩短地图生成时间；默认65536池的完整启动成本未纳入此测量，额外地图池存储也不包含在表中峰值。CUDA在每个阶段边界同步，正式重测均预热1轮、测2个完整PPO轮；首次编译不计正式吞吐。双方采用与真实训练入口一致的`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。采样、GAE和完整E4都计时，周期评测、checkpoint/日志I/O和快照发布不计时，因此去池的额外I/O收益没有算进吞吐。未设置吞吐达标门槛，`gate.passed`不表示棋力验收。GPU遥测保存在`gpu_telemetry.csv`。

更重要的是，发现基准脚本局部变量`observations`在估算存储后仍保留完整初始化观察，旧H4数据在整个测量中不释放；该多余引用在真实`train()`入口不存在。大规模测试因此额外占用数GiB，并出现OOM。最终对比两边都修复此测量引用（`del observations, codec`），基线的训练实现不改。先前v1的容量失败不再作为真实训练容量上限，也不混入正式速度和显存统计。v2修复了测量引用；v3另启用前述真实runner的终局引用释放，它是本次实现优化的一部分。

| 方案 | 真实对局 | 训练轨迹 | 每轮fresh玩家样本 | 每epoch逻辑minibatch | physical |
|---|---:|---:|---:|---:|---:|
| 优化前current单座 | 8192 | 8192 | 262144 | 32 | 512 |
| 优化后，等玩家样本 | 4096 | 8192 | 262144 | 32 | 512 |
| 优化后，等真实对局 | 8192 | 16384 | 524288 | 64 | 512 |

正式原文件：[基线](../outputs/analysis/current_selfplay_dual_4090_20261105/baseline_current_v2_01/report.json)、[等玩家样本](../outputs/analysis/current_selfplay_dual_4090_20261105/optimized_equal_players_v3_01/report.json)、[等真实对局](../outputs/analysis/current_selfplay_dual_4090_20261105/optimized_equal_games_v3_01/report.json)。测量源码固定为`baseline_source_v2/optimized_source_v3/`，逐文件hash保存在同名manifest；未覆盖原始冻结源。v2的无终局释放对照另行保留。v1的512与1024 microbatch探索速度近乎相同，1024显存压力更高，因此保留512；其数据仅用于该执行尺寸选择，不作最终速度对照。

相同玩家样本预算的最终结果如下。两者每轮均为262144个fresh玩家样本、128次Adam更新；每项排除一轮预热、测两轮，不是纯env.step吞吐。

| 方案 | 玩家样本/s | 棋盘tick/s | 采样秒/轮 | GAE秒/轮 | PPO更新秒/轮 | 完整秒/轮 | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 单座current，8192局 | 1878.84 | 1878.84 | 14.281 | 0.135 | 125.108 | 139.524 | 19.091 | 19.484 |
| 双座current，4096局 | 1953.52 | 976.76 | 9.358 | 0.134 | 124.698 | 134.190 | 13.535 | 15.297 |

完整训练吞吐提升**3.97%**，采样耗时减少**34.47%**，峰值allocated显存减少**29.10%**（约5.56GiB）。PPO更新仍占最终耗时的**92.9%**，其耗时基本不变；双座主要节省了每条有效玩家样本所需的模拟、对手推理和观察维护。不会因为每局多收一方样本就获得两倍完整训练吞吐。

终局引用释放的v3与v2等样本对照吞吐几乎相同（1953.52与1952.96），峰值allocated也相同；这个负结果保留。它释放了update期间的旧H4引用，但本组峰值发生在其他阶段，不能将理论释放字节数再从实测峰值中扣除。

同8192真实对局的双座容量测试也完成，退出码0：

| 真实对局／轨迹 | fresh玩家样本/轮 | Adam次数/轮 | 玩家样本/s | 棋盘tick/s | 采样秒/轮 | PPO更新秒/轮 | 完整秒/轮 | 峰值allocated／reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192／16384 | 524288 | 256 | 1954.57 | 977.28 | 18.626 | 249.343 | 268.238 | 21.190／22.482 |

它每轮是基线的两倍玩家样本及两倍Adam次数，整体吞吐比基线高4.03%，并没有优于4096局组。期间有多次CUDA allocator映射OOM重试，虽然完整测量成功，也不宜把这作为4090长期训练的推荐尺寸。推荐使用`current-dual-4090.json`的4096真实对局，在几乎相同吞吐下保留更多显存余量；65536地图池还会额外占用内存，正式性能测试统一使用512。

同规模未释放终局引用的v2控制为1880.53玩家样本/s、allocated21.188GiB／reserved22.500GiB；本次v3大规模吞吐高约3.94%，但短程峰值依旧相近。释放选项有实际引用生命周期测试支持，不能把这两轮计时当作所有长训阶段的固定收益保证。

本次没有降低参数量、PPO epoch、逻辑batch或精度来制造加速。主要结论是：同样本预算的采样和常驻内存得到改善，完整训练仍受既定ViT的PPO计算限制；若后续追求大幅吞吐提升，应针对Transformer更新做profile，并单独验证任何网络或训练预算变化的棋力影响。

## 复现与移植

原始JSON与源码冻结均在上述analysis目录，`queue_v2_results.json/queue_v3_results.json`保留退出码。可用同目录脚本重新测量，输出必须使用新目录：

```bash
PERF_AUDIT=/home/yuhangwang/code/generals-rl/outputs/analysis/current_selfplay_dual_4090_20261105
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/yuhangwang/code/generals-rl/.venv/bin/python "$PERF_AUDIT/run_benchmark.py" \
  optimized_source_v3 --config "$PERF_AUDIT/optimized_equal_players_v3_config.json" \
  --output "$PERF_AUDIT/optimized-recheck-new" --warmup 1 --iterations 2 --minimum-tps 0
```

测基线时改用`baseline_source_v2`与`baseline_current_8k_m512.json`，并选择另一个新输出目录。脚本显式选择冻结副本的PYTHONPATH和已有nvcc；不会重新安装共享venv。

`optimization.patch`记录吞吐测试时相对于dirty baseline的优化，`optimization_manifest.json`记录修改文件和哈希；补丁已在临时基线副本实际应用，并逐文件验证。原工作区370个源文件与原有tracked diff均保持不变，见`main_working_tree_guard.json`。

## 提交前审查与清理

自对弈优化单独提交在`perf/current-selfplay-dual`。已有stride=3、地图课程、env校验及logger修改继续保留为worktree未提交内容，不混入此提交；地图课程的棋盘tick计数适配也保留在worktree，随其课程代码一起提交。独立候选由原HEAD加本次优化构造并完整验证，确认不依赖这些未提交功能。

清理内容：移除去池后不可达的current summary分支，复用已解析的对手配置；双座缓存命中后直接返回，复用固定索引并去掉重复的player_ids分配；内联一次性知识编码包装；删除测试中的未使用导入/变量及current模式无效的快照参数。4090预设保留显式microbatch和推理chunk限制，避免以后调整全局默认时增加本卡显存压力。

修复双座`stochastic=false`原先只在rank0绑定时失败的问题：所有rank在创建环境和进入learner collective之前统一拒绝。真实双rank回归同时检查双方拒绝、没有创建输出目录，且进程组仍可同步。

清理后完整CPU回归：worktree **412 passed、15 skipped**，独立提交候选 **398 passed、12 skipped**，均有**3157 subtests passed**。两份代码的CPU/CUDA双座一致性测试分别通过。独立候选四轮4090短训（128局、T8、E4、BF16/compile、horizon3）通过，产生8192玩家样本/4096棋盘tick，1280局timeout，保留2个checkpoint且无history_pool。

本轮仅验证清理后的正确性，未重跑完整吞吐对照；上表仍对应不可变的`optimized_source_v3`。审查日志、独立候选及提交补丁保存在`outputs/analysis/current_selfplay_review_20261105/`。
