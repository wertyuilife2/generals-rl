# Generals PPO self-play

基于仓库内 RSL 分支的离散 PPO。当前支持官方规则 1v1、双座纯自对弈，以及历史对手池消融。
默认配置见 [default.json](src/generals_rl/default.json)，设计与可选实验见
[训练设计](../doc/rl_training_design.md)，实验结果见[训练日志](../doc/rl_train_log.md)。

## 安装与启动

在已有 PyTorch/CUDA 的环境中，从仓库根目录执行：

```bash
conda activate generals-rl
python -m pip install -r rl/requirements-train.txt
python -m pip install -e ./env -e ./baselines -e ./arena -e ./rl/vendor/rsl_rl -e ./rl --no-deps --no-build-isolation
python -m generals_rl.train --output outputs/rl/my-run
# 八卡；每 rank 使用同一份配置中的环境数与物理批大小。
torchrun --standalone --nproc_per_node=8 -m generals_rl.train --output outputs/rl/my-ddp-run
```

RSL 使用本仓库源码。GPU 环境需要 `nvcc`；可用 `GENERALS_NVCC=/path/to/nvcc` 指定已安装的编译器。
独立 checkout 使用共享 Python 环境时，设置
`PYTHONPATH=rl/src:env/src:baselines/src:arena/src:rl/vendor/rsl_rl` 选择该 checkout 的源码。

## 默认配置

采用2026-11-10路径消融的 `v32_9x9_temporal_hl128_paths`，见[配置](../doc/terrain_distance_20261110.md)及[结果](../doc/terrain_distance_20261110_results.md)。这是训练配方；新run从头初始化，不自动加载该实验的权重。

| 项目 | 默认值 |
|---|---|
| 地图 | 固定9×9尺寸、随机地图，5级易图→官方图课程；每层图池65536张 |
| Episode | 训练与评估均为 **2048 tick**，对9×9及全尺寸一致 |
| 地图生成／时间特征 | 初始池磁盘缓存、2个CPU生成worker，固定池不更新；clock尺度2048、记忆age尺度2000，与episode上限分开配置 |
| 输入／网络 | `temporal_paths` H8/K14、40通道；ViT1M、`early_shared`、特权critic、patch1、分次前向/反向 |
| 采样 | 每卡2048个真实对局、4096条玩家轨迹；T512 |
| 更新 | E1/M256；每卡逻辑batch8192、物理batch8192；推理chunk2048 |
| 预算 | 224轮PPO；八卡合计3,758,096,384条玩家transition |
| 训练对手 | live current；双方轨迹均用于PPO |
| 学习率 | categorical adaptive LR，范围 `[1e-5, 1e-3]`，desired KL=.01 |
| 奖励／价值 | 纯终局±1，其余0；gamma=1、lambda=.95；HL-Gauss128，sigma=.04、value loss系数=.01、关闭value clipping |
| 地图课程 | 全局board ticks达到71,303,168开始推进，306,184,192结束；无奖励/gamma课程 |
| 数值／存储 | BF16前向、FP32参数/GAE/loss；历史去重、FP16特征传输、`PathObservationCodec`保存行动时路径 |
| 评估 | 默认开启，每20次更新及最终模型；原四baseline＋Kestrel v7、Razor v8、Aegis v2，采样动作 |
| 保留 | 常规checkpoint保留2个；评估保留最近2个，并保护best、top3复评候选和在途任务 |

`task.num_envs` 表示真实对局数。双座模式的玩家transition是board tick数的两倍；
**地图课程按全局board ticks推进**，PPO样本数按玩家transition统计。
`runner.num_steps_per_env` 是rollout长度，与episode上限分别配置。权重在一个rollout内固定，局内可以跨多次更新。

4090可从 [current-dual-4090.json](configs/current-dual-4090.json) 开始：保留环境数、T/E/M和逻辑batch，仅将物理batch降至512、推理chunk降至1024。
地图课程按全局board ticks固定阈值；减少GPU数会使课程在更多PPO轮后结束。正式八卡实验的评估脚本额外检查U112并另做best/final独立种子复评；通用入口保留每20轮＋最终评估和top3复评机制，不依赖实验机的绝对路径。
全尺寸地图需同时覆盖 `task.width=task.height=23, task.random_maps=true` 和 `runner.policy.map_width=runner.policy.map_height=23`，并重新评估显存与课程预算；当前默认质量结果只覆盖9×9。
持续换图可设置 `task.map_refresh_reuse>0`；常驻池大小与累计新图数量分开控制，具体参数、缓存目录、复现边界及实测见[地图池设计](../doc/rl_rollout_and_map_bank_20261111.md)。固定池仍是默认，以便和已完成实验比较。
历史网络预设 [v32](configs/v32.json)／[v32-opt](configs/v32-opt.json) 使用当前默认2048上限；
它们是早期LightCNN，与本轮路径v32不同。复现已完成实验请使用其归档 `run_config.json` 与冻结源码。

## 配置覆盖与恢复

`--config FILE.json` 接受完整训练配方或稀疏覆盖。包含 `task`、`runner.num_steps_per_env` 以及 policy／algorithm 类名的完整配方独立加载，避免混入新默认的 AMP、学习率或课程设置。稀疏配置覆盖包内默认；同一 policy class 递归合并，更换 `runner.policy.class_name` 时替换整个 policy 子树。若需要清除同 class 的旧字段，使用 `--replace-policy` 并提供完整 `runner.policy`。

[`v32.json`](configs/v32.json) 与 [`v32-opt.json`](configs/v32-opt.json) 是已迁移到当前配置字段的 LightCNN 对照，需显式选择：

```bash
python -m generals_rl.train --config rl/configs/v32.json --output outputs/rl/v32-control
python -m generals_rl.train --config rl/configs/v32-opt.json --output outputs/rl/v32-opt-control
python -m generals_rl.train --resume outputs/rl/my-run/checkpoint.pt --iterations 224 --output outputs/rl/resumed
```

`--resume` 继承 checkpoint 旁的原 `run_config.json`，恢复模型、optimizer、学习率和可用历史池；对局和随机流重新开始。`--iterations` 表示本次额外执行的 PPO 轮数，上例从已完成224轮的模型恢复后将训练到448轮。恢复不会改用新的包内默认配置。当前主线只接受现有配置字段与类路径；旧历史对手池的模块路径签名不再兼容，继续旧实验使用其归档源码。`--warm-start PATH` 在权重结构匹配时只加载权重，使用本次配置、新 optimizer 和新池，不能与 resume 同用，也不提供旧类路径或不匹配权重的转换。完整有效配置保存在新输出目录的 `run_config.json`，训练统计为 `summary.json` 与 TensorBoard。

CPU 开发需显式覆盖两处 BF16、关闭网络编译并缩小运行规模。例如保存为 `cpu.json`：

```json
{
  "task": {"backend": "cpu", "device": "cpu", "num_envs": 16, "map_pool_size": 0, "map_curriculum": null},
  "runner": {
    "num_steps_per_env": 4,
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

## 可选训练方案

| 配置 | 用途 |
|---|---|
| `task.map_curriculum` | 易图→官方图的分层图池；阈值 `warmup_transitions/end_transitions` 的单位均为全局board ticks |
| `curriculum.potential_decay` | 线性衰减territory/cities/army shaping；终点缺省沿用tick课程终点 |
| `runner.policy.critic_fusion` | `late`、`early_shared`、`post_attention`、`joint_independent` |
| `runner.policy.value_distribution` | `scalar`、128-bin `hl_gauss`、两端点 `binary_ce` |
| `task.feature_set` | 默认 `temporal_paths`；另有 `stacked`、`compact`、`temporal`、`temporal_dedup` |
| `runner.algorithm.advantage_top_fraction` | 默认1；小于1时按标准化advantage绝对值筛样 |
| `runner.policy.patch_size` | 默认1；3使用非重叠patch并恢复逐格动作logit |

特征方案统一派生H/K及内置codec；修改 `feature_set` 后可重新解析已有配置。
显式 `storage_codec=null` 或自定义codec保持原选择；`storage_history=null/{}` 可关闭历史去重。
分类critic要求纯终局±1奖励、固定折扣，并关闭clipped value loss；GAE与timeout bootstrap始终使用解码标量。
具体特征、网络限制和损失语义见[训练设计](../doc/rl_training_design.md)。

历史对手模式设 `opponent="self_play", train_both_sides=false`，并配置 `self_play.pool_size/snapshot_interval`；当前紧凑/时序配方只支持双座，切换历史或单座消融需同时使用 `task.feature_set="stacked"`。
单座当前对手消融设 `opponent="current_self_play", train_both_sides=false`。

## 导出与评估

```bash
python -m generals_rl.export outputs/rl/my-run/checkpoint.pt \
  --output outputs/models/my-model --codename my-model
```

导出包包含actor权重、部署配置与冻结 `policy/` 源码，不含critic和optimizer。
Arena推理使用同一套特征、记忆、掩码和动作转换。模型包应整体保存；权重不纳入Git。

默认周期评估由Arena原生CPU批量链执行，模型在 `evaluation.device` 上推理。
每对手100张地图并交换座位，七对手共1400局；另加200局WAIT诊断，均值排除WAIT。
显式 `evaluation.profile` 的地图、上限、局数与对手优先于默认生成参数。

创建独立评估profile：

```python
import json
from pathlib import Path
from generals import MapConfig
from generals_arena.competitions.evaluation import make_profile
from generals_arena.policies.references import training_opponents

profile = make_profile(MapConfig(9, 9, 2), maps_per_opponent=100, max_ticks=2048,
    opponents=training_opponents(Path("outputs/eval/opponents")))
Path("profile.json").write_text(json.dumps(profile))
```

```bash
python -m generals_arena evaluate --bundle outputs/models/my-model --profile profile.json \
  --output outputs/eval/my-model.json --backend cpu --device cuda:0 --batch-size 128
```

全尺寸评估使用 `MapConfig(players=2)`，且模型需支持相应地图尺寸。默认 `inference.mode="stochastic"`；推理模式、特征和源码均属于模型身份。
更换评估面板或规则后新建run，保留历史报告原始口径。比赛、录像与联赛接口见 [Arena README](../arena/README.md)。
共享训练GPU时，周期评估在PPO更新之间同步运行；独立GPU可配置 `evaluation.cuda_visible_devices`，
并从训练可见卡中排除。默认allocator为 `expandable_segments:True`，显式环境变量优先。

## 指标与验证

TensorBoard位于run输出目录，记录PPO更新、episode胜负/截断、吞吐、显存和七对手胜率。
`Arena/baseline4_mean_win_rate` 保留原四基线口径；`Arena/mean_win_rate` 排除WAIT。
共享梯度探针记录主rank首个物理批、归约和裁剪前的梯度，`PPO/shared_grad_probe_samples` 给出采样量；
这些指标及分类critic首批诊断不是全rollout或全卡平均。

```bash
python -m pytest -q rl/tests rl/tools/tests
python rl/tools/benchmark_training.py --help
```

代码职责：`policy/` 管理可导出的actor与MDP输入；`task.py`／`mdp/` 管理对局与奖励；
`map_sampling.py`／`map_curriculum.py` 管理初始地图；`student_ppo.py` 管理更新，`ppo_monitor.py` 管理诊断；
`training_evaluation.py` 管理候选及Arena评估队列。vendor修改记录见 [PATCHES.md](vendor/rsl_rl/PATCHES.md)。
