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

- 周期训练 checkpoint：`save_interval=2000`；最终模型统一为 `checkpoint.pt`，包含两个 optimizer。
- 历史对手池：每 50 次更新发布权重快照，结束时补充最终模型；最多保留初始模型和最近 7 个候选。
- 定期评估默认开启：每发布 10 个周期 learner 快照评估一次，初始随机模型不计入周期。
- 每次评估默认每 AI 50 张独立地图、每图交换两座位，共 400 局，时限 2000 tick。
- 评估使用局部 student 的 argmax，完成整个固定赛程；超时单列为 draw，并计入总胜率分母。
- CPU 评估子进程持有快照副本，结果写入 `evaluations/index.json`、各快照的 `report.json` 和 TensorBoard `Eval/`。
- 可用 `--no-eval` 关闭定期评估。TensorBoard 启动：`tensorboard --logdir outputs/moe_cts`。

## MDP 与模块边界

| 模块 | 职责 |
| --- | --- |
| `mdp/state.py` | CPU/GPU 共用张量状态视图 |
| `mdp/observations.py`、`mdp/combat.py` | 局部/特权观测、每局历史与可选战斗特征 |
| `mdp/actions.py` | 动作编码、合法 mask |
| `mdp/rewards.py` | 奖励项与权重 |
| `task.py` | 联合 tick、终止/截断、重置、座位轮换；支持注入上述组件和对手 |
| `cts.py` | RSL VecEnv/TensorDict 适配与算法配置 |
| `student_ppo.py` | 可选的全学生 PPO；复用原版 PPO 更新，学生获得任务梯度 |
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
`task.productive_actions=true` 可过滤预计在下一次增长后仍为零兵的移动，
训练、历史对手和评估共用该 mask；每次更新与整套评估记录等待、零兵、立即折返和实际移动兵力指标。

`task.combat_features=true` 在 9×9、1v1 中用两张战斗特征替换重复的相对身份平面，维度保持 985。
特征分别是可见敌军四邻域中下一次增长后最大留一攻击兵力，以及自身守军/一次增援与威胁的差值。
隐藏兵力在计算前清零；这些特征表示潜在动作，不是对联合动作的预测。训练、历史对手和评估共用 encoder 工厂。
旧权重应先迁移并以新 optimizer/历史池 warm start，禁止直接 resume：

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

v8 可选 `generals_rl.spatial_ppo.LightSpatialPPOPolicy`：actor 使用四帧局部地图、最新公共特征和动作 mask，
两层16通道卷积输出原649动作；critic 使用独立卷积网络处理特权观测。其结构与 dense 权重不兼容，
可先用 `rl/tools/distill_spatial.py --light-cnn` 从冻结 student 和历史池进行流式 KL/value 初始化，
再从新 optimizer 和新 CNN 历史池开始 PPO。蒸馏不使用 baseline overlay，固定 probe 包含在训练数据中。

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

训练默认 self-play，每局开始均匀抽取一个冻结历史 student，整局不换模型。
候选淘汰后，正在进行的对局继续持有旧模型；不同座位各自维护历史，对手采样使用独立 RNG。
独立 MDP 调试可用 random/wait 对手。

`baseline_league` 可在最多2%的固定环境中混入原版 aggressive，默认关闭；其余使用历史 self-play。
例如 `{"baseline_envs":128,"seed":19000043,"native_candidates":true}` 用于16,384环境。
原AI保持每局记忆、随机数和结果反馈；只有候选打分可选用等价C++加速（需要g++），不修改评估AI。
`batched_observations=true` 可批量构造原版local Observation，只从GPU读取选中的环境行；
默认关闭。观测类型、整数公共统计、不可变数组和原AI决策/反馈接口保持一致。
TensorBoard `League/` 分别记录两类对手的胜率、决策数及host开销；训练内胜率不代替独立评估。

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
