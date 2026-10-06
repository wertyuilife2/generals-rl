# Generals MoE CTS baseline

基于纯净 rsl_rl 源码和少量适配补丁，实现可修改 MDP 的历史池 self-play 训练。
CPU/GPU 使用同一组 observation、action 和 reward；训练、对手池、评估和录像工具分别组织。

## 安装

在已安装 PyTorch 的 Python 环境中，从仓库根目录运行：

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r rl/requirements-train.txt
.venv/bin/python -m pip install -e ./env -e ./rl/vendor/rsl_rl -e ./rl --no-deps --no-build-isolation
```

GPU 模拟另外需要 CUDA 编译工具链；让 `nvcc` 位于 PATH，或用 `GENERALS_NVCC` 指定编译器路径。

## 训练与评估

```bash
# CPU smoke：8 个环境，2 次更新
.venv/bin/python -m generals_rl.train --config rl/configs/smoke.json \
  --iterations 2 --output outputs/moe_cts/smoke

# CPU baseline，默认历史池 self-play
.venv/bin/python -m generals_rl.train --config rl/configs/baseline.json \
  --iterations 1000 --output outputs/moe_cts/baseline

# GPU：16384 个环境，每环境精确采样 1M tick
.venv/bin/python -m generals_rl.train --config rl/configs/gpu.json \
  --steps-per-env 1000000 --output outputs/moe_cts/gpu_run

# 当前学生 PPO：v17 网络/MDP，默认开启 categorical 自适应 LR 与 PPO 监控
.venv/bin/python -m generals_rl.train --config rl/configs/self_play_ppo.json --replace-policy \
  --steps-per-env 1000000 --output outputs/moe_cts/self_play_ppo

# v9：随机初始化的轻量 CNN，纯历史模型 self-play；每轮最多 3 小时
.venv/bin/python -m generals_rl.train --config rl/configs/v9.json --replace-policy \
  --steps-per-env 1000000 --output outputs/moe_cts/iterations/v9/train

# v10：相同从零自对弈方案，第二层卷积 dilation=3，直接感受野扩至 9×9
.venv/bin/python -m generals_rl.train --config rl/configs/v10.json --replace-policy \
  --steps-per-env 1000000 --output outputs/moe_cts/iterations/v10/train

# v11：回到两层普通卷积；局部己将相对坐标，省去不参与 PPO 的单帧存储
.venv/bin/python -m generals_rl.train --config rl/configs/v11.json --replace-policy \
  --steps-per-env 1000000 --output outputs/moe_cts/iterations/v11/train

# v12：仅将 v11 并行环境增至 24576；786432 samples/update，继续从零纯自对弈
.venv/bin/python -m generals_rl.train --config rl/configs/v12.json --replace-policy \
  --steps-per-env 1000000 --output outputs/moe_cts/iterations/v12/train

# v13：回到16384环境；历史对手用argmax，learner仍随机采样，保持从零
.venv/bin/python -m generals_rl.train --config rl/configs/v13.json --replace-policy \
  --steps-per-env 1000000 --output outputs/moe_cts/iterations/v13/train

# v15：回到v11随机历史对手与动作；记住局部视野曾见的敌将，从零训练
.venv/bin/python -m generals_rl.train --config rl/configs/v15.json --replace-policy \
  --steps-per-env 1000000 --output outputs/moe_cts/iterations/v15/train

# 在新目录续训：模型、两个 optimizer 和可用历史池恢复，对局和 RNG 重新开始
.venv/bin/python -m generals_rl.train --resume outputs/moe_cts/baseline/checkpoint.pt \
  --iterations 1000 --output outputs/moe_cts/resumed

# 对原版 aggressive/expansion/defensive/random AI 的独立评估
.venv/bin/generals-evaluate outputs/moe_cts/baseline/checkpoint.pt \
  --maps-per-ai 50 --output outputs/moe_cts/baseline/builtin_evaluation.json
```

默认值由 `TaskConfig`、`runner_config`、`SelfPlayConfig` 和 `TrainingEvalConfig` 定义。
JSON 配置只指定需要关注或覆盖的参数；其余设置通过 `--config` 修改。
训练输出目录必须是新目录，`--iterations` 与 `--steps-per-env` 二选一。

训练记录保存在 TensorBoard，结束时生成 `summary.json`；`run_config.json` 保存本次完整配置。
速度分别报告采样加更新的 transition/s、包括评估等待在内的全流程 transition/s。

学生 PPO 默认 `schedule="adaptive", desired_kl=0.02, ppo_monitor=true`。自适应 LR 使用每个
minibatch 的已采样动作 log-prob 估计 KL，沿用 rsl_rl 的调节规则；监控另在最多8192条固定
rollout观测上每epoch计算精确 KL，按1024条分块forward，不保存完整rollout的旧分布。
TensorBoard 的 `PPO/` 下记录 approximate/exact KL、clip fraction、entropy、梯度范数、实际 LR、
每epoch的 WAIT/MOVE gate 与加权条件 MOVE KL，以及小批观测缓存大小。
精确 KL 衡量更新开始到epoch结束的变化；近似 KL 相对 rollout 的行为策略。
大网络严格FP32对照可显式设顶层 `matmul_precision="highest", cudnn_allow_tf32=false`，
分别控制矩阵与卷积TF32；默认不覆盖cuDNN设置。CUDA评估复用配置，报告/trace记录实际精度。
在 `runner.algorithm` 设置 `schedule="fixed", desired_kl=null` 关闭 LR 调节，
设置 `ppo_monitor=false` 关闭监控；`ppo_probe_samples` 和 `ppo_probe_batch_size` 控制监控开销。
旧 `v1`–`v18` 配置保持原实验的 fixed LR，不会因新默认值改变学习率。

- 周期训练 checkpoint：`save_interval=2000`；最终模型统一为 `checkpoint.pt`，包含两个 optimizer。
- 历史对手池：每 50 次更新发布权重快照，结束时补充最终模型；最多保留初始模型和最近 7 个候选。
  已分配到对局的旧模型保持引用至该局结束，不因从池中淘汰而中途更换对手。
- 定期评估默认开启：每发布 10 个周期 learner 快照评估一次，初始随机模型不计入周期。
- 每次评估默认每 AI 50 张独立地图、每图交换两座位，共 400 局，时限 2000 tick。
- 所有评估默认从局部 student 的 masked categorical 分布采样，完成整个固定赛程；超时单列为 draw，并计入总胜率分母。
- 每局有独立采样随机流，默认由评估seed和game_id派生；报告记录模式、协议及每局采样seed，支持挑选单局复现录像。
  独立评估可用 `--sampling-seed` 固定地图并更换动作种子；`--greedy` 或 `evaluation.stochastic=false` 显式选择greedy诊断。
  TensorBoard `Eval/stochastic=1` 标识采样评估，历史greedy结果保留原模式标签。
- CPU 评估子进程持有快照副本，结果写入 `evaluations/index.json`、各快照的 `report.json` 和 TensorBoard `Eval/`。
- 大网络可显式设 `evaluation.device="cuda:0"`，在一次PPO更新结束后暂停训练并同步评估；
  `evaluation.timeout_seconds` 控制超时，同时受训练剩余墙钟预算约束。worker退出并回收后再收集下一轮。
  GPU评估默认保存每AI首个真实胜/负的联合动作trace，缺失类别明确报告；录像按记录动作在CPU/Core重放，
  不重新推理模型。设置 `evaluation.keep_snapshots` 后，trace随当前best及最近K个完成评测有界保留。
- 可用 `--no-eval` 关闭定期评估。TensorBoard 启动：`tensorboard --logdir outputs/moe_cts`。

## MDP 与模块边界

| 模块 | 职责 |
| --- | --- |
| `mdp/state.py` | CPU/GPU 共用张量状态视图 |
| `mdp/observations.py`、`mdp/combat.py` | 局部/特权观测、每局历史与可选战斗特征 |
| `mdp/knowledge.py`、`mdp/layout.py` | 各玩家合法观测的持久知识、地图/历史/动作维度契约 |
| `mdp/actions.py` | 动作编码、合法 mask |
| `mdp/rewards.py` | 奖励项与权重 |
| `task.py` | 联合 tick、终止/截断、重置、座位轮换；支持注入上述组件和对手 |
| `cts.py` | RSL VecEnv/TensorDict 适配与算法配置 |
| `student_ppo.py`、`ppo_monitor.py` | 全学生 PPO、categorical 自适应 LR、默认开启的更新监控；学生获得任务梯度 |
| `information_ppo.py`、`backbones.py`、`map_ppo.py` | 信息输入消融、上游成熟网络、可变地图的policy/value head |
| `curriculum.py` | rollout边界的可选gamma课程与tick penalty衰减 |
| `map_sampling.py` | 回合重置时从独立随机地图池抽图 |
| `opponents.py`、`self_play.py` | 冻结历史 student、采样及局内对手生命周期 |
| `self_play_training.py` | CTS 模型工厂和快照发布 |
| `evaluate_builtin.py`、`training_evaluation.py` | 完整赛程对战与训练期间的评估调度 |
| `search.py` | 算法独立的 PolicyValue 协议和 CPU 整 tick 分支模拟 |

默认棋盘 9×9、2 玩家。局部观测仅显示本人领土及八邻域，隐藏地图特征清零；
army/territory/alive 总数公开，隐藏对手 city 总数。单帧 985 维，student 使用 4 帧历史，
teacher/critic 使用完整棋盘；部署和历史对手推理不接收 critic。

动作共 649 个：`WAIT=0`；`MOVE=1+source*8+direction*2+mode`，方向上/右/下/左，
模式全部留一/一半。mask 与原引擎一致，允许零兵移动；终局仅允许 WAIT。
`task.productive_actions=true` 过滤预计在下一次增长后仍为零兵的移动；
训练、历史对手和评估共用该 mask；每次更新与整套评估记录等待、零兵、立即折返和实际移动兵力指标。
v14 的等价 HALF 去重没有改善开发胜率，已撤回；复现该实验请使用 commit `670abe0` 或其冻结源码。

v15 的已见敌将记忆未改善开发胜率，已撤回；复现该实验请使用 commit `1fac98c` 或其冻结源码。

`task.combat_features=true` 在 9×9、1v1 中用两张战斗特征替换重复的相对身份平面，维度保持 985。
特征分别是可见敌军四邻域中下一次增长后最大留一攻击兵力，以及自身守军/一次增援与威胁的差值。
隐藏兵力在计算前清零；这些特征表示潜在动作，不是对联合动作的预测。训练、历史对手和评估共用 encoder 工厂。
以下迁移命令用于复现 v4 的历史实验。v9 起的方案比较从随机初始化训练，无需迁移：

```bash
.venv/bin/python rl/tools/migrate_combat_observations.py \
  --checkpoint outputs/moe_cts/iterations/v3/train/evaluations/best.pt \
  --config outputs/moe_cts/iterations/v3/train/run_config.json \
  --output outputs/moe_cts/iterations/v4/migrated
.venv/bin/python -m generals_rl.train --config rl/configs/v4.json \
  --warm-start outputs/moe_cts/iterations/v4/migrated/checkpoint.pt \
  --steps-per-env 1000000 --output outputs/moe_cts/iterations/v4/train
```

迁移将旧身份输入权重折叠到等价 self/enemy 平面，新特征列初始化为零，旧函数在浮点舍入范围内保留。
其余 MDP 设定及模型布局必须兼容；最终候选选择和训练记录见训练日志。

`task.king_threat_distance=true` 用可见可移动敌军到己方主将的最近 Manhattan 距离 /16
替换冗余的 seat-P1 标量，仍保留 seat-P0，单帧维度985。没有可见威胁时取1；
它不考虑墙和行军路径，也不强制防守动作。开启前使用同一迁移工具的
`--king-threat-distance` 选项：原 seat-P1 权重折叠到 seat-P0 和 bias，新距离列为零。
只恢复权重，新建 optimizer 和历史池；浮点舍入（尤其 TF32）可能造成微小差异。

v8/v9 使用 `generals_rl.spatial_ppo.LightSpatialPPOPolicy`：actor 使用四帧局部地图、最新公共特征和动作 mask，
两层16通道卷积输出原649动作；critic 使用独立卷积网络处理特权观测。其结构与 dense 权重不兼容，
v9 直接随机初始化。下面的蒸馏命令仅用于复现 v8 的历史实验；新的方案比较禁止蒸馏、warm start 和旧模型初始化。

v11 在 actor 首卷积前增加两张己将相对 dx/dy 平面（坐标差除以8），由最新局部观测的
己方所有权与将军标记推导，不读取隐藏敌将位置；critic、MDP 与 rollout 维度不变。
`own_king_geometry` 默认关闭；v11 另以 `runner.storage_obs_keys` 去掉 CNN 不读取的
`single_obs` 存储，实时观测保持完整，PPO/GAE 与默认完整存储逐项精确一致。

```bash
.venv/bin/python rl/tools/distill_spatial.py --light-cnn --device cuda:0 \
  --checkpoint outputs/moe_cts/iterations/v8/distill_source/checkpoint.pt \
  --config outputs/moe_cts/iterations/v8/distill_source/run_config.json \
  --teacher-pool outputs/moe_cts/iterations/v8/distill_source/history_pool \
  --output outputs/moe_cts/iterations/v8/distillation
.venv/bin/python -m generals_rl.train --config rl/configs/v8.json --replace-policy \
  --warm-start outputs/moe_cts/iterations/v8/distillation/checkpoint.pt \
  --steps-per-env 1000000 --output outputs/moe_cts/iterations/v8/train
```

`--replace-policy` 用完整配置替换 policy 子树，避免递归合并遗留 dense 专属参数。
蒸馏只保存两个 CNN 权重文件、不保存观测数据集；正式训练仍按配置限制历史池、评估和 checkpoint 数量。

逐 tick reward：`0.05*Δterritory + 0.2*Δcities + 0.001*Δarmy + win - loss - 0.001`。
自然结束和时限截断分开处理；截断保留 reset 前的下一状态供 value bootstrap。
CPU 默认 episode 时限 512 tick；GPU 配置为 1024 tick，gamma 从 0.99 覆盖为 0.995。
v5及后续配置使用 `potential_gamma=.995`：将统计差分替换为
`gamma*Phi(next)-Phi(before)`，自然终止令下一潜能归零，超时保留潜能并bootstrap。
`critic_potential` 与 `value_potential_offset` 配套启用，让critic输出基础value减Phi；
训练gamma必须与reward匹配。

当前信息消融配置为 `v19`–`v22`：actor持久知识与critic读取actor历史/知识形成2×2。
`task.persistent_knowledge=true` 时知识按玩家/episode隔离，只更新合法视野，owner/city表示最后观测值并附年龄；
知识单独存储，不按history重复堆叠。成熟ResNet/ViT的三档与动态地图合约见
[架构设计](../doc/rl_architecture_design.md)，接受新地图形状不等于已验证行为泛化。

课程默认关闭。设顶层 `curriculum.tick_decay=true`，在TensorBoard迭代0..1000将原tick penalty乘以1→0，
资源PBRS和终局奖励系数保持原值。gamma课程显式设置 `curriculum.gamma_final` 和
`curriculum.gamma_end_iteration`；初始值取PPO配置，课程同步修改PPO与PBRS gamma。
同一rollout的采样、GAE和PPO更新保持一个设置，随后推进下一轮；`Curriculum/`记录实际值，resume按绝对迭代恢复。
这些开关是后续实验准备，不自动加入历史或正在运行的版本。

训练默认 self-play，每局开始均匀抽取一个冻结历史 student，整局不换模型。
候选淘汰后，正在进行的对局继续持有旧模型；不同座位各自维护历史，对手采样使用独立 RNG。
独立 MDP 调试可用 random/wait 对手。

训练对手只使用历史 student；原版 baseline AI 仅用于独立评估和录像。
v6–v8 的混合对手训练实现已移除，旧 `baseline_league` 配置会明确报错。
复现这些历史版本时使用对应 Git commit 或该版本已冻结的 `source/`，勿直接运行旧配置。

MoE CTS 默认 teacher/student lane 比例 0.75/0.25，32 tick rollout、4 epoch。
先做 PPO，再做 student latent distillation 和 expert load balance；网络沿用 vendor 的 dense MoE。
CPU 默认 4 minibatch，GPU 配置 64；两阶段复用同一批采样数据及索引计划。
`rl/configs/v3.json` 使用 student-only PPO：所有 lane 由局部 student 采样，
一个 Adam 更新 student/actor/critic，privileged critic 保留；取消 latent 蒸馏。
该适配支持 fixed LR，且不保存未使用的旧动作概率，16k×32 额外节省约 1.27 GiB。

`task.map_pool_size=65536` 可启用地图重采样；各回合结束时换图，正在进行的对局保持原地图。
生成和采样使用由 task.seed 分流的独立 RNG；0 为关闭。
模块集中适配 native backend 的初始状态缓存，重置保留底层指针。

后续 MCTS 可复用 MDP 组件和分支模拟器，策略先验可通过 `action_probabilities()` 获取。
当前尚未实现 MCTS 的 selection/backup 或搜索目标训练；局部视野搜索还需要自行定义 belief。

## 录像工具

`rl/tools/recordings.py` 的 `MatchRecorder` 可注入 `battle_batch(..., recorder=...)`，
保存 action trace 和压缩逐 tick 状态，并用 CPU CoreEngine 独立重放核验。
视频渲染需要 Pillow 和系统 ffmpeg；工具不参与训练：

```bash
.venv/bin/python rl/tools/render_recording.py MATCH.frames.json.gz MATCH.mp4
```

## 来源与验证

原始 vendor commit：`38c6cad003a293c50196e5d62a799ad59c8fb400`。
来源 commit：`28b4516d22617b11aeaf8ead63cc00b0c0bcd1bd` 的 `source/rsl_rl/` 子树。
完整原始源码和许可证保留；改动说明见 [PATCHES.md](vendor/rsl_rl/PATCHES.md)，差异由 Git 记录。
历史模型、录像、实验报告和测试日志存放在被 Git 忽略的 `outputs/`。

```bash
.venv/bin/python -m unittest discover -s rl/tests -v
.venv/bin/python -m unittest discover -s rl/tools/tests -v
.venv/bin/python -m pytest rl/vendor/rsl_rl/tests -q
```

测试覆盖 CPU/GPU MDP 一致性、局部信息隔离、timeout bootstrap、历史池冻结与恢复、
真实两阶段更新、精确采样预算、固定赛程评估与子进程 RNG 隔离。
Vendor 回归对照旧缓存流程，验证 Normal/Categorical 的损失、参数、两个 Adam 状态和 RNG 一致。
基线支持单设备训练。

## 训练版本迭代

`codex_dev` 上的 v1/v2/… 配置、结果与曲线记录在 [训练日志](../doc/rl_train_log.md)。
v9 起每个方案均从随机权重、新 optimizer、新历史池训练，不传 `--warm-start` 或 `--resume`；
所有版本使用相同训练种子和选模赛程作开发对比，最终候选另用未使用的随机种子复核。
`--warm-start PATH` 只加载完整 checkpoint 或快照中的模型权重，从新 optimizer、迭代 0 和新历史池开始；
配置使用本次 JSON 覆盖默认值。`--resume` 则继承原运行配置并恢复 optimizer 和历史池，二者互斥。
可用 `matmul_precision=high` 启用本轮 float32 矩阵乘法加速，默认精度沿用进程设置。
`training_control` 可设置时限、评估平台停止和完整 checkpoint 保留数量；开启早停时，采样预算为上限，实际步数和停止原因写入摘要。
在该 run 输出目录创建 `STOP` 文件，可在当前完整更新结束后保存 checkpoint、完成已排队评估并写入摘要，再停止该轮训练。
`evaluation.keep_snapshots` 限制已完成评估的权重数量，另保留按四 AI 最低胜率优先挑选的 `evaluations/best.pt`。
TensorBoard 包含四 AI 与最低胜率、吞吐、显存、进程峰值内存及评估队列长度。

```bash
.venv/bin/python rl/tools/analyze_iteration.py outputs/moe_cts/iterations/v1/train \
  --output outputs/moe_cts/iterations/v1/analysis --demo --render
```

分析工具保存曲线、摘要及每 AI 一败一胜的录像（若存在），录像通过独立 Core 逐 tick 重放核验。
