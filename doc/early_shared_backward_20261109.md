# early_shared 分开前向与反向：显存和吞吐验证

2026-11-09。针对有特权信息的 early_shared，在保持逻辑 batch 和更新规则的条件下，默认依次执行两路前向、反向。8×H200 实测预热后峰值 allocated 从 34.03 降至 20.66 GiB（−39.3%），完整 PPO 吞吐下降约 1.9%。这是一项显存优化，没有重新验证长期训练胜率。

## 执行方式

一个物理 microbatch 内依次进行：

1. Actor 前向，计算 surrogate 和 entropy loss，反向并释放其激活。
2. Critic 前向，计算 value loss，反向；共享参数的梯度累加到第一步结果上。
3. 累积完该逻辑 minibatch 的全部 microbatch 后，执行一次跨卡梯度同步、一次全局梯度裁剪和一次 Adam step。

两路之间不清空梯度、不裁剪、不更新权重。early_shared 原本就需要对局部信息和局部＋特权信息分别执行 Transformer，因此没有增加 Transformer 前向次数；局部 embedding 重算一次。共享参数不意味着必须同时保留两路激活。

策略通过 `separate_actor_critic_backward` 属性声明适用性，StudentPPO 复用现有分开反向分支。本次生产代码只修改 [scaled_ppo.py](../rl/src/generals_rl/scaled_ppo.py) 和 [student_ppo.py](../rl/src/generals_rl/student_ppo.py)，没有新增训练超参数或模型参数。

- 有特权、`critic_fusion=early_shared`、`batch_shared_forward=false`：默认分开前向和反向。
- 无特权 early_shared、late、post_attention：继续复用共享激活并联合反向。
- 显式 `batch_shared_forward=true`：保留既有合并前向和联合反向行为。
- Rollout 继续使用联合前向。TensorBoard 的既有指标 `PPO/separate_actor_critic_backward` 记录 PPO 实际路径。

## 八卡实测

同一台 8×H200，PyTorch 2.8.0+cu129；9×9、每卡 2048 真实对局/4096 玩家轨迹、双座 current self-play、T512/E1/M256、逻辑与物理 minibatch 都为每卡 8192、推理 chunk2048、BF16＋torch.compile。每轮全卡 16,777,216 条 fresh 玩家样本。

两版加载同一份 v11 U224 的完整 Actor/Critic 权重，以同样的种子、地图设置和新建 Adam 状态开始。使用独立进程，先测原实现，再测优化实现；每版预热 2 轮，再测 5 轮，所有测量轮均保留。完整流程包含环境、双座采样、超时 bootstrap、GAE、PPO、Adam、自适应学习率、PPO 监控和跨卡同步，排除 eval、训练 logger 和模型写盘。地图课程保持初始等级，gamma=0.99。

时间为各轮八卡最慢时间的中位数，显存取八卡最大值；各阶段的最慢 rank 可不同，阶段中位数之和不必等于总耗时中位数。

| 指标 | 原联合反向 | 分开前向/反向 | 变化 |
|---|---:|---:|---:|
| 预热后峰值 allocated/卡 | 34.03 GiB | 20.66 GiB | −39.3% |
| 预热后峰值 reserved/卡 | 35.97 GiB | 23.56 GiB | −34.5% |
| 采样时间/轮 | 13.82 s | 13.74 s | — |
| GAE 时间/轮 | 0.58 s | 0.61 s | — |
| PPO 更新时间/轮 | 22.62 s | 23.33 s | +3.1% |
| 完整时间/轮 | 36.44 s | 37.15 s | +2.0% |
| 全卡 fresh 玩家样本/s | 460,454 | 451,612 | −1.9% |

包括初始化和预热，进程全程最大 allocated 分别为 34.34/20.71 GiB，reserved 为 36.00/23.56 GiB。测量阶段完整单轮时间范围分别为 36.30–42.77 s、37.05–38.48 s；百分之二量级的时间差属于本次短测观察，不承诺其他设备/地图下的吞吐不变。两版正式计时均每轮 256 个逻辑 minibatch，物理 batch 和样本预算完全相同。

这组是新的受控性能测量，不应把其显存数值与历史 v11 整轮训练记录混用。减少双路激活是主要实现变化，但没有对 0.71 s/轮的时间差逐项做硬件归因。

## 正确性与精度

本机回归 **123 passed、37 subtests passed**。新增检查覆盖 patch1/3、CPU FP32/CUDA BF16、两路共享梯度、非整除 microbatch（4＋2）、多次 Adam 更新；同时验证原 checkpoint 恢复、Actor 导出、其他融合模式和分布式 PPO。

H200 使用 v11 U224 完整权重及同一批 128 个真实观测，比较原联合反向和新分开反向：

| 精度/执行 | 前向 mean KL | 前向 max KL | 最大 value 差 | 梯度相对 L2 差 |
|---|---:|---:|---:|---:|
| FP32 eager，matmul precision=high | 0 | 0 | 0 | 0.00115% |
| BF16 eager | 0 | 0 | 0 | 0.16838% |
| BF16 compile | 0.00000327 | 0.00016003 | 0.017578 | 0.15908% |

这里的 KL 是同一权重在两种执行方式下的分布差异，不是 PPO 更新的 KL。分开反向改变了共享 embedding 梯度的累加/低精度转换顺序，单独编译两路也可能改变算子融合，因此不保证逐位一致或相同训练轨迹。CPU 梯度及完整 Adam 更新检查用于验证数学目标一致；BF16 额外检查更新后的策略/value 差异。

八卡两种方式均完成 7 轮真实 PPO 更新，最终所有参数向量相对 rank0 的最大差值均为 **0**，没有非有限 loss 或 OOM。测量进程已结束，八张卡已释放。

## 复现材料

材料目录：outputs/analysis/early_shared_backward_20261109（`../outputs/analysis/early_shared_backward_20261109`）。包括冻结源码和哈希、修改前文件、本次独立补丁、固定配置/观测/权重、benchmark.py（`../outputs/analysis/early_shared_backward_20261109/benchmark.py`）、原始 JSON/log、comparison.json（`../outputs/analysis/early_shared_backward_20261109/comparison.json`） 和回归日志。

在该目录下，将 `source/rl/src`、`source/rl/vendor/rsl_rl`、`source/env/src`、`source/arena/src` 加入 `PYTHONPATH` 后，用 `torchrun --standalone --nproc_per_node=8 benchmark.py ppo --variant joint --output <新文件>` 和 `--variant sequential` 对照。默认环境数、T、batch 和轮数就是上述口径。`python benchmark.py parity --output <新文件>` 可复现完整权重的数值检查。
