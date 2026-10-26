# Generals PPO self-play

默认训练使用 F03 消融选出的 **ViT 1M dense、H4/K11、recent8 历史对手池与 gamma 课程**，配置随包保存在 [`generals_rl/default.json`](src/generals_rl/default.json)。直接运行训练入口即可使用，不需要 F03 调度工具。训练规则由 `env/` 提供，四个基础 AI 只参与评估。

正式结果见 [H200 scaling：22 项训练](../doc/h200_scaling.md) 和 [F03 顺序消融](../doc/f03_ablation.md)；研究历史见 [Auto research 记录](../doc/auto_research.md)。

## 安装与运行

在仓库根目录激活已有 PyTorch/CUDA 的 `generals-rl` conda 环境：

```bash
conda activate generals-rl
python -m pip install -r rl/requirements-train.txt
python -m pip install -e ./env -e ./baselines -e ./arena -e ./rl/vendor/rsl_rl -e ./rl --no-deps --no-build-isolation
python -m generals_rl.train
```

`requirements-train.txt` 是本地测试环境的依赖固定清单；已有兼容依赖的环境可跳过第一条 pip 命令。
训练使用仓库内的 RSL 分支，不能用同名 PyPI 包替代该 editable 安装。H200 完整实训使用 PyTorch 2.8.0+cu129。

默认从头初始化，训练 **224 轮 PPO**，输出写入新的 `outputs/rl/<时间戳>/`。`--output PATH` 可指定新目录；`--iterations` 或 `--steps-per-env` 可覆盖预算，二者互斥。单卡执行沿用每 rank 配置，环境总数和全局 minibatch 小于八卡实验。

按正式实验的八卡规模运行：

```bash
torchrun --standalone --nproc_per_node=8 -m generals_rl.train --output outputs/rl/f03-selected
```

配置默认使用 CUDA 和原生 BF16，需要相应设备与 `nvcc` 编译工具链。下表的全局量以八个 rank 为准。

| 项目 | 默认值 |
| --- | --- |
| 游戏与输入 | 9×9、2 人，局部 H4 combat 历史＋K11 持久知识；独立特权 critic |
| 模型 | ViT 1M dense，actor/critic 参数独立 |
| 采样 | 每 rank 16,384 个环境，全局 N=131,072；T=32 |
| PPO | E=4，每 epoch 64 个逻辑 minibatch；全局 B=65,536，每 rank B=8,192 |
| 预算 | 224 轮，八卡合计 939,524,096 个 fresh learner transitions |
| 历史对手 | 最近 8 个候选快照，逐局抽样；每轮发布，`retain_seed=false` |
| 学习率 | actor/critic 共享，adaptive 范围 `[1e-5, 1e-3]`，desired KL=.01 |
| 课程 | gamma 与 PBRS gamma 从 .99 线性到 .999，iteration=112 到达终值；tick 惩罚在 iteration=125 衰减至 0 |
| 数值执行 | learner／历史对手推理与 PPO 使用 BF16，`compile_forward=true`；概率、GAE、loss 归约及主参数/Adam 状态保持 FP32 |

iteration 从 0 起算。每次 PPO 轮包含一次采样、GAE 和完整 E4；重复使用的数据不重复计为 fresh 样本，对手样本也不计入 learner transitions。物理 microbatch 与推理 chunk 用于控制激活内存，不改变逻辑更新预算。

Actor 只接收该座位的局部观测、历史、知识和公开统计；critic 的完整棋盘不传给 actor。H4 是四帧近期输入，recent8 是八个历史对手，二者不同。知识和历史按座位及 episode 隔离，初始随机对手随候选池更新淘汰，进行中对局继续使用其原对手。
默认 actor/critic 的运行统计 observation normalization 均关闭；各字段的固定缩放由 `policy/` 中的特征构造实现。

## 配置覆盖与恢复

`--config FILE.json` 接受完整训练配方或稀疏覆盖。包含 `task`、`runner.num_steps_per_env` 以及 policy／algorithm 类名的完整配方独立加载，避免混入新默认的 AMP、学习率或课程设置。稀疏配置覆盖包内默认；同一 policy class 递归合并，更换 `runner.policy.class_name` 时替换整个 policy 子树。若需要清除同 class 的旧字段，使用 `--replace-policy` 并提供完整 `runner.policy`。

[`v32.json`](configs/v32.json) 与 [`v32-opt.json`](configs/v32-opt.json) 是已迁移到当前配置字段的 LightCNN 对照，需显式选择：

```bash
python -m generals_rl.train --config rl/configs/v32.json --output outputs/rl/v32-control
python -m generals_rl.train --config rl/configs/v32-opt.json --output outputs/rl/v32-opt-control
python -m generals_rl.train --resume outputs/rl/f03-selected/checkpoint.pt --iterations 224 --output outputs/rl/resumed
```

`--resume` 继承 checkpoint 旁的原 `run_config.json`，恢复模型、optimizer、学习率和可用历史池；对局和随机流重新开始。`--iterations` 表示本次额外执行的 PPO 轮数，上例从已完成224轮的模型恢复后将训练到448轮。恢复不会改用新的包内默认配置。当前主线只接受现有配置字段与类路径；旧历史对手池的模块路径签名不再兼容，继续旧实验使用其归档源码。`--warm-start PATH` 在权重结构匹配时只加载权重，使用本次配置、新 optimizer 和新池，不能与 resume 同用，也不提供旧类路径或不匹配权重的转换。完整有效配置保存在新输出目录的 `run_config.json`，训练统计为 `summary.json` 与 TensorBoard。

CPU 开发需显式覆盖两处 BF16、关闭网络编译并缩小运行规模。例如保存为 `cpu.json`：

```json
{
  "task": {"backend": "cpu", "device": "cpu", "num_envs": 16, "map_pool_size": 0},
  "runner": {
    "policy": {"amp_bfloat16": false, "compile_forward": false, "inference_chunk_size": 16},
    "algorithm": {"amp_bfloat16": false, "num_mini_batches": 2, "physical_minibatch_size": 128}
  },
  "evaluation": {"enabled": false}
}
```

```bash
python -m generals_rl.train --config cpu.json --iterations 1 --output outputs/rl/cpu-smoke
```

仅设置 `--backend cpu --device cpu` 不会自动修改精度；上述 CPU 配置用于开发检查，不作为八卡训练的性能或质量复现。历史已退休的实验变体使用原冻结源码和运行配置。

## 代码职责

| 路径（相对 `src/generals_rl/`） | 职责 |
| --- | --- |
| `policy/` | actor、特征、历史／知识、动作映射和推理运行时；训练共用，导出时随模型快照 |
| `actor_critic.py` 及各 `*_ppo.py` 策略模块 | 训练网络、独立 critic 与网络配方 |
| `mdp/`、`task.py`、`cts.py` | 奖励、训练任务、环境和 RSL 接口 |
| `self_play.py`、`self_play_training.py`、`opponents.py` | 历史池、发布与对手采样 |
| `student_ppo.py`、`train.py`、`distributed_training.py` | PPO 更新、训练入口与多卡同步 |
| `export.py`、`training_evaluation.py` | 策略导出、Arena 评估队列、候选保留及 best 选择 |

对战执行、赛程与评分位于 Arena 的 `matches/` 和 `competitions/`。RL 不再维护独立对战评估器，旧入口和旧类路径没有兼容别名。

## 模型导出与 Arena 评估

正常训练无需单独的发布步骤：

1. 启动训练时把 `generals_rl/policy/` 和部署配置快照到 run 目录，checkpoint 记录其内容哈希。
2. 按 `runner.save_interval` 配置的周期原子保存 checkpoint；按 `evaluation.every_n_updates` 配置的周期保存、导出 actor 策略包并入队 Arena。最后一轮始终保存，启用评估时必评。
3. Arena 按固定地图配置、对手版本、预算和种子批量对战，从相同初始分开始逐局评分，返回 Elo、CI、胜平负及耗时。
4. 训练结束排空队列，再用共同的新种子和加大预算确认周期 Elo 最高的候选，写出 `best_model.json`；候选数和预算倍数分别由 `evaluation.final_top_k`（默认3）与 `final_games_multiplier`（默认2）控制。
5. 共享 best 指向的完整 `bundle/`，合作者直接 `generals-arena commit 目录` 或 `run`。

默认关闭基于 Elo 的平台期早停（`training_control.plateau_evaluations=null`），
`min_improvement` 默认按 10 Elo 判断有效进步。需要早停时显式设置连续无有效进步的评估次数；
该开关与 `evaluation.enabled` 独立。

`evaluations/index.json` 保存持久队列；`evaluations/candidates/` 保存候选的完整可恢复 checkpoint、
训练配置和 actor-only bundle；`evaluations/jobs/` 保存每个任务的 profile、Arena 报告和 worker 日志。
best 指针不复制权重，包含模型代号、已完成更新数、
`policy_id/profile_id`、Elo/CI、报告和可恢复 checkpoint 路径。best 恢复模型/optimizer，
对局和随机流重置；candidate 不另存历史对手池，只恢复原 run 中仍可获得的池内容。
`policy_id` 对权重、部署配置和源码一起取内容身份；不同版本无需手工维护冻结策略目录。
常规 checkpoint 由 `training_control.keep_checkpoints` 控制（默认2，含最终 `checkpoint.pt`）；
`evaluation.keep_snapshots` 单独控制最近评估候选的保留数量，完整恢复 checkpoint 与对应 bundle 一起保留或删除；best、最终复评候选及排队／运行中的候选另外保护。该项默认 `null`，即保留所有评估候选，不受常规 checkpoint 数量限制。评估任务的报告、profile 和日志继续保留。
从较早 checkpoint 恢复时，会按 SHA 找回该候选的完整权重，重新导出已清理的 bundle 并核对原 `policy_id`；无法恢复权重的历史结果只用于曲线，不参加 best 或最终复评选择。
评估 profile 中的对手文件路径相对于 profile 所在目录解析；复制完整 run 后，分析工具按新的 run 目录定位报告和 checkpoint。

手动导出任意新 checkpoint（checkpoint 中已有来源元数据）：

```bash
python -m generals_rl.export outputs/rl/my-run/checkpoint.pt \
  --output outputs/models/nova-224 --codename nova-ViT1M
# 默认模型只支持9×9、1v1；首次创建与它兼容的联赛
python -m generals_arena league init outputs/arena/models --config arena/configs/leagues/legacy_9x9_1v1.json
python -m generals_arena commit outputs/models/nova-224 --league outputs/arena/models --device cuda:0 --workers 1
```

已有联赛跳过 `init`；其他地图／人数模型需要匹配自己的联赛配置。`commit/run` 的环境执行仍是 CPU 单局，
`--device` 只指定包的推理设备；需要 GPU 环境和跨对局批推理时使用下方 `evaluate` 或 Arena 的 `tournament`。

包内只有 `manifest.json`、`actor.pt`、`model_config.json` 和 `policy/`，不含 critic、optimizer 或训练依赖。
Arena 仅解析通用 manifest；features/history/knowledge/mask、网络和动作转换均由 RL 的共享 policy 实现。
训练和部署共用 actor 与特征实现，部署才切换 eval/冻结参数。执行包需要 env/Arena、PyTorch、NumPy、timm，
不需要安装 RL、RSL 或 TensorDict。显式配置的 AMP、matmul/TF32/cuDNN 设置随包保存，
执行后恢复进程全局设置，以免不同包相互影响。权重和 `outputs/` 不纳入 Git，直接共享整个目录即可。

旧 checkpoint 需提供与当前导出器兼容的权重、原配方及完成轮数；不能仅凭路径或命名判断兼容。
例如已验证可导出的历史 S4 final U224（不是同 run 的 best U196），保留原 stochastic 评估模式：

```bash
python -m generals_rl.export /path/to/S4/final.pt \
  --config /path/to/S4/run_config.json --updates 224 \
  --codename v0.0.1-ViT1M --mode stochastic --output outputs/models/v0.0.1-ViT1M-224
```

默认 `inference.mode="stochastic"` 保留训练配方的采样评估；可显式设为 `greedy`，该选择属于模型包身份。
新 run 的代号来自 `model.codename`，未设置时自动用 `run_id-输出目录名`，resume 继承原代号。同代号下 `completed_ppo_updates` 是一基完成次数，
原始 `iteration=updates-1`；新 seed/新实验应使用新代号。自动代号含输出目录名，手动指定代号时需自行保持唯一。
`--codename` 可更名，但会改变导出包身份。元数据在 `model_config.json`，Arena 不解释。
源代码或 features/network/inference 配置改变时，旧 checkpoint 不能伪装成原方案 resume；
应使用原源码继续，或显式 warm-start 新训练。新训练的 source snapshot 要和 checkpoint 一起保留。

单独评估导出包：

```python
import json
from generals import MapConfig
from generals_arena.competitions.evaluation import make_profile
profile = make_profile(MapConfig(9, 9, 2, .1, .05), maps_per_opponent=100, max_ticks=2000)
# 可配置其他 package / baseline 对手；1v1 自动加入 WAIT 对照。
# 独立 C++ 策略包只支持 CPU 隔离执行，不能用于下方 GPU 批评估。
with open("profile.json", "w") as f:
    json.dump(profile, f)
```

```bash
python -m generals_arena evaluate --bundle outputs/models/nova-224 --profile profile.json \
  --output outputs/eval/nova.json --backend gpu --device cuda:0 --batch-size 256
```

训练评估固定对手策略版本、初始条件和赛程，逐局使用与公开联赛相同的 `gor-relative-codeforces-v9`，所有参赛策略的原始分数正常更新。
1v1 从原始分 0 开始，实际安排 WAIT 对局，展示时减去 WAIT 原始分；3–8 人 FFA 从 1000 开始，独立保持全池均分 1000。
置信区间按独立地图和模式对应的概率模型计算。profile 内的最终分数受阵容、赛程和预算影响，不能直接等同公开池的分数；
best 仅比较相同 profile 和固定部署条件，profile 本身不绑定候选设备/batch。评分更改后的 schema 2 profile 应用于新 run；旧 run 用冻结源码继续。
当前赛程按对手分组，Elo 对分组顺序与 WAIT 对局结果敏感；基础AI面板的最高 Elo 只是该评估协议的最优候选，不能保证在RL模型直接对战中更强。需要同时查看各对手胜平负和确认评估的近似区间。
Arena 将训练评估、固定 round-robin 和持久 league 作为不同赛程模式，共用比赛执行、结果、录像和评分规则。
RL 只管理 checkpoint、候选队列和 best 选择，固定对手评估继续调用 `generals_arena.competitions.evaluation`；通用比赛入口见 [Arena 文档](../arena/README.md)。
`--record-replays` 保存可由 Arena 逐 tick 验证的录像；
`generals-arena evaluate` 是接受 bundle/profile 的 CLI。纯 env.step 性能指标与含决策的批评估 wall throughput 分开报告。

训练默认与训练卡共用 `cuda:0`，在 PPO 更新间同步批评估；CPU 或显式独立单卡评估可异步。
设置 `evaluation.cuda_visible_devices="7"` 时训练须通过 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` 排除该卡，
worker 中仍用 `cuda:0`。当前批评估只使用一个 device，不保留旧多卡分片执行器；可增大单卡 batch。
`evaluation.profile` 可指向 JSON 文件或配置对象，`every_n_updates/maps_per_opponent/backend/batch_size`
控制频率、预算及执行；显式 `profile` 内的地图／对手／局数覆盖默认 profile 的生成参数。推理模式由顶层 `inference.mode` 指定。
`batch_size=0` 表示按当前兼容赛程组全量并行，不自动探测显存或搜索最优 batch。profile 中的 package 对手须各自设置推理 `device`，否则使用 CPU。
随机推理的 lane RNG 绑定 seed 和计数，
与分批、重排及其他局结束无关。GPU 不同 batch 的矩阵乘法可能造成微小浮点差异，不能保证整局动作逐位一致。

## Benchmark 与验证

`rl/tools/benchmark_training.py` 测量完整 rollout＋GAE＋PPO 的 fresh learner transitions/s，包含双方决策，排除评估、存盘和独立 profile。全流程 wall throughput 另报；它与仅计 `env.step(action)` 的环境基准不同。

```bash
python rl/tools/benchmark_training.py --help
python -m pytest -q rl/tests rl/tools/tests
```

保留当前模型／MDP、自对弈、分布式、学习率恢复、评估和 Core 重放回归；vendor 框架及许可证保留。`search.py` 提供未来搜索的整 tick 分支接口，尚未实现完整 MCTS。实验结果和冻结产物留在报告及原归档，正式训练不依赖旧 F03 阶段调度、门禁或作业监督脚本。
