# Generals MoE CTS baseline

基于纯净 rsl_rl 源码和少量适配补丁，实现可修改 MDP 的历史池 self-play 训练。
CPU/GPU 使用同一组 observation、action 和 reward；训练、对手池、评估和录像工具分别组织。

## 安装

在已安装 PyTorch 的 Python 环境中，从仓库根目录运行：

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r rl/requirements-train.txt
.venv/bin/python -m pip install -e ./env -e ./baselines -e ./arena -e ./rl/vendor/rsl_rl -e ./rl --no-deps --no-build-isolation
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
三个 JSON 配置只指定需要关注或覆盖的参数；其余设置通过 `--config` 修改。
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
| `mdp/observations.py` | 局部/特权观测、每局历史 |
| `mdp/actions.py` | 动作编码、合法 mask |
| `mdp/rewards.py` | 奖励项与权重 |
| `task.py` | 联合 tick、终止/截断、重置、座位轮换；支持注入上述组件和对手 |
| `arena_adapter.py`、`inference.py` | 局部 student 推理与独立 arena 的 Controller 适配 |
| `cts.py` | RSL VecEnv/TensorDict 适配与算法配置 |
| `opponents.py`、`self_play.py` | 冻结历史 student、采样及局内对手生命周期 |
| `self_play_training.py` | CTS 模型工厂和快照发布 |
| `evaluate_builtin.py`、`training_evaluation.py` | 完整赛程对战与训练期间的评估调度 |
| `search.py` | 算法独立的 PolicyValue 协议和 CPU 整 tick 分支模拟 |

默认棋盘 9×9、2 玩家。局部观测仅显示本人领土及八邻域，隐藏地图特征清零；
army/territory/alive 总数公开，隐藏对手 city 总数。单帧 985 维，student 使用 4 帧历史，
teacher/critic 使用完整棋盘；部署和历史对手推理不接收 critic。

动作共 649 个：`WAIT=0`；`MOVE=1+source*8+direction*2+mode`，方向上/右/下/左，
模式全部留一/一半。mask 与原引擎一致，允许零兵移动；终局仅允许 WAIT。

逐 tick reward：`0.05*Δterritory + 0.2*Δcities + 0.001*Δarmy + win - loss - 0.001`。
自然结束和时限截断分开处理；截断保留 reset 前的下一状态供 value bootstrap。
CPU 默认 episode 时限 512 tick；GPU 配置为 1024 tick，gamma 从 0.99 覆盖为 0.995。

训练默认 self-play，每局开始均匀抽取一个冻结历史 student，整局不换模型。
候选淘汰后，正在进行的对局继续持有旧模型；不同座位各自维护历史，对手采样使用独立 RNG。
独立 MDP 调试可用 random/wait 对手。

MoE CTS 默认 teacher/student lane 比例 0.75/0.25，32 tick rollout、4 epoch。
先做 PPO，再做 student latent distillation 和 expert load balance；网络沿用 vendor 的 dense MoE。
CPU 默认 4 minibatch，GPU 配置 64；两阶段复用同一批采样数据及索引计划。

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

不同 checkpoint、baseline 和其他算法的统一 Elo 比较使用顶层 [arena](../arena/README.md)。
在参赛配置中设置 `kind: "rl"` 和 checkpoint 路径；每个版本使用独立名称。
Arena 的参考比赛可在 macOS 上执行，无需 Linux 原生 CPU 后端。
