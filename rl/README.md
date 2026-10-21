# Generals PPO self-play

默认训练使用 F03 消融选出的 **ViT 1M dense、H4/K11、recent8 历史对手池与 gamma 课程**，配置随包保存在 [`generals_rl/default.json`](src/generals_rl/default.json)。直接运行训练入口即可使用，不需要 F03 调度工具。训练规则由 `env/` 提供，四个基础 AI 只参与评估。

正式结果见 [H200 scaling：22 项训练](../doc/h200_scaling.md) 和 [F03 顺序消融](../doc/f03_ablation.md)；研究历史见 [Auto research 记录](../doc/auto_research.md)。

## 安装与运行

在仓库根目录激活已有 PyTorch/CUDA 的 `generals-rl` conda 环境：

```bash
python -m pip install -r rl/requirements-train.txt
python -m pip install -e ./env -e ./baselines -e ./arena -e ./rl/vendor/rsl_rl -e ./rl --no-deps --no-build-isolation
python -m generals_rl.train
```

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

## 配置覆盖与恢复

`--config FILE.json` 接受完整训练配方或稀疏覆盖。包含 `task`、`runner.num_steps_per_env` 以及 policy／algorithm 类名的完整配方独立加载，避免混入新默认的 AMP、学习率或课程设置。稀疏配置覆盖包内默认；同一 policy class 递归合并，更换 `runner.policy.class_name` 时替换整个 policy 子树。若需要清除同 class 的旧字段，使用 `--replace-policy` 并提供完整 `runner.policy`。

[`v32.json`](configs/v32.json) 与 [`v32-opt.json`](configs/v32-opt.json) 是历史 LightCNN 对照，需显式选择：

```bash
python -m generals_rl.train --config rl/configs/v32.json --output outputs/rl/v32-control
python -m generals_rl.train --config rl/configs/v32-opt.json --output outputs/rl/v32-opt-control
python -m generals_rl.train --resume outputs/rl/f03-selected/checkpoint.pt --iterations 224 --output outputs/rl/resumed
```

`--resume` 继承 checkpoint 旁的原 `run_config.json`，恢复模型、optimizer、学习率和可用历史池；对局和随机流重新开始。恢复不会改用新的包内默认配置。`--warm-start PATH` 只加载权重，使用本次配置、新 optimizer 和新池，不能与 resume 同用。完整有效配置保存在新输出目录的 `run_config.json`，训练统计为 `summary.json` 与 TensorBoard。

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

## 批量评估与录像

```bash
python -m generals_rl.evaluate_builtin outputs/rl/f03-selected/checkpoint.pt --maps-per-ai 100 --output outputs/rl/builtin.json
python -m generals_rl.evaluate_builtin outputs/rl/f03-selected/checkpoint.pt --devices cuda:0 cuda:1 --output outputs/rl/builtin-multigpu.json
python -m generals_rl.evaluate_duel --help
```

训练和 F03 评估使用 masked categorical 采样、固定地图与双座赛程。基础 AI 每局使用独立随机流；模型 duel 将随机流绑定地图与绝对座位，报告配对地图 bootstrap 得分区间。tick 上限截断计 draw；worker 执行超时是任务错误。Greedy 分数与采样分数分开。

四个基础 AI 统一使用 native C++ batch。评估支持 CPU/GPU VecEnv、CUDA Graph、数组动作 trace 和 `--devices` 多卡分片；旧 `baseline_workers` 只兼容读取，不控制执行。Razor 等自定义对手接口保留。

周期评估保存在 `evaluations/index.json`、各快照 `report.json` 和 TensorBoard。`evaluation.device="cuda:0"` 默认暂停训练后同步评估；CPU 评估异步。可用 `evaluation.cuda_visible_devices` 分配独立 GPU，例如训练仅暴露 GPU0–6，评估设置 `"7"`；worker 内仍用 `cuda:0`。默认检查训练/评估 GPU 隔离，多卡评估可设置 `"6,7"` 并将训练限制到 GPU0–5。异步结果按快照队列处理，训练结束排空评估，其他 rank 使用 CPU/Gloo 等待。

动作 trace 可由 `rl/tools/analyze_iteration.py`、`recordings.py` 和 `render_recording.py` 独立重放、核验和渲染，不重新推理模型。[Arena](../arena/README.md) 已支持 v32、v32-opt 和 F03 的模型、history 与 knowledge，但当前是单局 greedy 执行；正式 RL 批量评估继续使用上述采样协议。后续统一对战入口、回放与 Elo 时需统一地图、采样和超时规则。

## Benchmark 与验证

`rl/tools/benchmark_training.py` 测量完整 rollout＋GAE＋PPO 的 fresh learner transitions/s，包含双方决策，排除评估、存盘和独立 profile。全流程 wall throughput 另报；它与仅计 `env.step(action)` 的环境基准不同。

```bash
python rl/tools/benchmark_training.py --help
python -m pytest -q rl/tests rl/tools/tests
```

保留当前模型／MDP、自对弈、分布式、学习率恢复、评估和 Core 重放回归；vendor 框架及许可证保留。`search.py` 提供未来搜索的整 tick 分支接口，尚未实现完整 MCTS。实验结果和冻结产物留在报告及原归档，正式训练不依赖旧 F03 阶段调度、门禁或作业监督脚本。
