# Generals MoE CTS 补丁说明

原始源码来自 `/home/yuhangwang/code/go2_rl_robotlab` 的纯净 commit
`28b4516d22617b11aeaf8ead63cc00b0c0bcd1bd`，子树 `source/rsl_rl/`，版本 3.3.0。
本仓库原始导入 commit 为 `38c6cad003a293c50196e5d62a799ad59c8fb400`；两个子树的 Git 树哈希一致。
未复制来源工作区的未提交修改。完整上游源码、测试及许可证保留。

仅修改以下四个运行时源文件：

| 文件 | 修改与用途 |
| --- | --- |
| `modules/actor_critic_moe_cts.py` | 增加 masked Categorical、离散动作 ID、log-prob/entropy 和无采样的动作先验；Normal 默认行为保留。Categorical 的 `action_std` 仅返回 `[N,1]` 的全 1 logger 占位值。 |
| `algorithms/moe_cts.py` | 增加 Categorical KL；可靠地分配非空 teacher/student 环境组；使用 reset 前终局下一状态处理 timeout bootstrap；删除全 epoch batch 缓存，两个训练阶段复用固定随机索引。 |
| `storage/rollout_storage_cts.py` | 分离动作与分布形状/dtype，保留 bool mask；按 env-major 索引逐批读取 rollout；支持有效时间前缀。新增默认开启的 `store_action_std` / `store_action_mean`；关闭时对应 `sigma` / `mu` 为 `None`，不分配、写入或取样缓存。Categorical 关闭标准差，StudentPPO 另关闭旧动作分布均值。 |
| `runners/on_policy_runner_cts.py` | 使用已有 callable resolver；配置离散动作、分布形状、算法的 teacher 数量工厂和标准差缓存开关；无日志目录时跳过周期保存。第二优化器可为 `None`，保存与恢复时兼容单优化器算法。 |

默认 MoE-CTS 的 PPO/GAE、两个 optimizer、student PPO encoder 停止梯度、latent MSE、expert load balance 和网络结构沿用上游。
Categorical KL 不使用标准差；logger 的 `Policy/mean_noise_std` 为兼容常数，不能用于分析离散探索。
Normal 模式仍完整缓存标准差。16384 环境 × 32 tick × 649 动作时，移除的 float32 缓存理论大小为 1.268 GiB。

新增 `tests/test_moe_cts_streaming.py` 对照原 env-major flatten 和全 epoch 缓存，检查两轮更新的
损失、参数、两个 Adam 状态和 RNG 完全一致；Categorical 对照保留旧标准差缓存，新路径不分配。
测试同时覆盖短 rollout 和未使用尾部含 NaN 的情况。
Generals 集成测试位于 `rl/tests/`，self-play、评估及录像接入均位于 vendor 外。

`generals_rl.student_ppo` 是 vendor 外的学生 PPO 对照：沿用未修改的 `PPO.update`，
复用已修复的 CTS timeout/GAE 与全学生流式存储，解除学生 actor 编码的停止梯度，
冻结 teacher，以一个 Adam 更新学生/actor/critic，不进行 latent 蒸馏。
使用 fixed learning rate、`desired_kl=None` 避开原 PPO 的 Gaussian KL；critic 可读 privileged
状态，学生 latent 保持 detach，部署策略仍只读 local obs。`rl/tests/test_student_ppo.py`
验证完整 Adam/RNG 与原 PPO update 精确一致，并验证真实 self-play、warm start、resume 和子进程 eval。
StudentPPO 设置 `requires_old_distribution=False`，runner 据此关闭 `mu` 缓存；其 fixed LR
更新不使用旧动作均值。默认 CTS 保留该缓存。16384 环境 × 32 tick × 649 动作时额外节省
1.268 GiB float32 存储；同一 rollout 的有缓存/无缓存 loss、参数、Adam 和 RNG 逐项精确对照。

使用 Git 查看原始源码及全部改动：

```bash
git show 38c6cad:rl/vendor/rsl_rl/rsl_rl/algorithms/moe_cts.py
git diff 38c6cad -- rl/vendor/rsl_rl
```
