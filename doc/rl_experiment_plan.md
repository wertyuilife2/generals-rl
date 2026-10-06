# v19 起：信息、容量与长期信用分配实验

用户需求来源：2026-10-07 附件 pasted-text-1.txt，以及 doc/temp_plan.md。用户补充优先于旧方案中的实验顺序与奖励衰减建议。所有正式训练从随机权重、新 Adam、新历史池开始，不加载旧权重；工作分支 codex_dev，main 不修改。

## 实验顺序与完整范围

1. v19：重跑 v17 的 MDP、轻量网络和历史池，采用已启用的 categorical adaptive LR 和可复现采样评估，建立当前协议下的从零对照。旧 v17 的 fixed LR / greedy 成绩仅作历史背景。
2. v20：仅给 critic 增加 actor H4 信息，actor 与 MDP 保持对照；v21：仅给 actor 增加持久事实记忆；v22：记忆与 history-state critic 组合，补齐 2×2 消融。版本顺序可根据实际结果调整，但必须保留四个条件的结果。
3. 解耦观测布局、地图 geometry 和动作 head；选择有成熟开源代码依据的 ResNet 与 ViT，记录上游版本、许可和适配。两种架构都支持可变地图大小，提供约 1M / 10M / 100M actor 档位。公平比较固定 critic、MDP、PPO、自对弈和评估协议；比较同样本量的学习曲线及同墙钟预算的强度、吞吐、显存。100M 必要时减 N 增 T，保持 N×T=524288；调整实际 microbatch/梯度累积以适配单卡。
4. 在信息与容量结果上，先单独比较固定 gamma=.995 / .999，再测试 .995→.999 课程；分别检验 rollout 与 lambda，保持分段 PPO 和整局固定历史对手。所有 potential_gamma 与实际 return gamma 对齐。
5. 最后检验 tick penalty 在 TensorBoard 训练迭代 0..1000 线性衰减 1→0。领地/城市/兵力 potential-based shaping 与胜负奖励不衰减。课程完成后留足目标阶段预算。
6. 最后做 H4→H8 消融，处理额外观测缓存成本；不提前把 H8 混入前面实验。

## 每轮执行与验收

- 先做有意义的单元/契约验证和 CPU/GPU smoke；正式时间 1..3h，至少 1h 后评估曲线无显著上升才停，最多 3h。时间包含启动，因此报告同时保留纯 collection+update compute 与完整 wall。
- 纯历史模型 self-play：不加入 baseline 训练对手；历史对手整局固定；模型池固定 8，checkpoint 固定 3，eval 模型 3+best。
- 所有 eval 默认采样；四 baseline 固定选模种子、双座；选定 best 后另做开发地图与采样种子评估。明确选模与开发成绩，旧 greedy 结果不当作新消融对照。
- 每个正式版本先 commit 代码/配置，再冻结运行时源码与哈希，训练/异步 eval 使用冻结源，防止后续改代码污染运行。最终结果、吞吐、内存、PPO KL/clip/entropy/value 与行为分析写入 doc/rl_train_log.md，曲线保存 doc/rl_train_figures/；选最好模型记录真实失败与胜局 demo。
- 逐阶段依据结果选择下一轮基座；不堆砌特定 baseline hack，不开展大规模超参数搜索。强度目标是可推广的 Generals 策略，这四种 AI 是回归检查，不是能力上限证明。

## 当前进度

v19 对照正式训练已启动（PID2961698，冻结源 commit898c838），至少1h、最多3h。第一点 update99 sampled selector 已完成。v20 critic-only 配置和适配器已通过18项CPU合同验证，待对照结束后正式训练。持久记忆、完整上游ResNetV2/ViT及可变几何、microbatch/精度合同已实现并分别验证；GPU BF16、完整池显存和大模型训练中评测仍需门禁，尚未进行其正式训练。完整目标尚未完成。
