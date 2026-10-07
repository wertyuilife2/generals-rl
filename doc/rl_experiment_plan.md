# v19 起：信息、容量与长期信用分配实验

用户需求来源：2026-10-07 附件 pasted-text-1.txt，以及 doc/temp_plan.md。用户补充优先于旧方案中的实验顺序与奖励衰减建议。所有正式训练从随机权重、新 Adam、新历史池开始，不加载旧权重；工作分支 codex_dev，main 不修改。

## 实验顺序与完整范围

1. v19：重跑 v17 的 MDP、轻量网络和历史池，采用已启用的 categorical adaptive LR 和可复现采样评估，建立当前协议下的从零对照。旧 v17 的 fixed LR / greedy 成绩仅作历史背景。
2. v20：仅给 critic 增加 actor H4 信息，actor 与 MDP 保持对照；v21：仅给 actor 增加持久事实记忆；v22：记忆与 history-state critic 组合，补齐 2×2 消融。正式训练顺序为v19→v20→v21→v22，保留四个条件的结果。
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

v19对照已完成1847更新、968359936转移、121.95分钟正式训练，并正常排空18次采样selector评测；停止为人工曲线平台判断后的`requested_stop`，不是自动patience12触发。最终best1199采样selector为93.75/97.5/99.25/99.5%，复用开发seed1001042为90.5/97/99.5/99.5%（400局/AI）。该开发集此前使用过，未作为heldout；最终八局录像、独立Core trace核验、失败分析、PPO/资源趋势及全部18点曲线已记录在训练日志。高baseline成绩不结束完整任务，也不构成可变地图泛化或世界级强度的证明。

v20 critic-only正式训练完成2034更新、1066401792转移、153.510分钟，人工平台STOP后正常exit0并排空20次采样selector。最终best1399 selector93.75/95.75/99.5/99.75%，复用开发集93/96.25/99.5/99.5%；与v19最佳模型新执行1600局采样cross-play胜率54.375%（800配对图CI52..56.75%）。相同最佳selector aggressive仍均93.75%，开发改善CI跨零；wall吞吐下降12.51%，因此未证实critic-only的稳定样本/计算效率收益。完整曲线、969tick八录像、信息观察与单训练seed/不同预算限制已记录训练日志。

v21仅actor增加持久事实（d55a835）正式训练已正常结束：929更新、487063552转移、84.20分钟，best299 selector91/97.5/99.25/99.5%，复用开发集89.75/97.75/99.25/99.5%。全部九个selector、117505次开发决策的知识核验、751tick八局录像与Core重放、PPO/资源和完整曲线已记录并提交ad6d8f6。知识保留正确不等于网络有效使用；开发差异与单训练seed不足以证明记忆的稳定收益。

v22组合条件（af92831）已正常结束：实际session31517 terminal845145 exit0，958完整更新、502267904转移、118.529分钟；九个selector全部completed0，best499为88.5/96.25/100/98.75%。best后的四点约57分钟未持续提高，按完整更新边界STOP，非自动patience12。最终best与中期499同SHA，明确复用actual14838的1600局开发（89.75/97.5/99.5/99%）及59696/26293八录像/Core证据，不假称新终局执行。终局捕获、完整sample/wall及PPO图均已actual CPU exit0并核验，四条件同样本/同wall比较只作单seed开发证据。容量共同采用H4+K11与S+H4+K11，是遵循已接受的信息语义的工程选择，不是tiny胜率或效率赢家；所有六档信息/critic保持一致。

成熟上游ResNetV2/ViT、可变地图布局/geometry/action head、microbatch、GPU评测/真实trace和可选gamma/tick课程模块已实现。116 Git blobs/25模块父子合同以及两种1M严格FP32/p512完整门禁已通过。随后ResNet与ViT的BF16/p512完整门禁也实际exit0（16380/66af8c、64661/74d5af）：NT524288/256Adam、24历史驻留、1600局CUDA采样workers1→4均完成，完整对局与选中joint-action记录一致；update190.050/150.043秒、workers4完整eval阶段404.623/365.667秒。共同选定BF16/p512、pub2、每4updates评测、timeout900、reserve1300、min1h/max3h、plateau4；p1024仅短诊断且ViT空闲不足1GB，未选择。接下来提交v23/v24配置、从各自commit冻结并从零训练，再执行10M/100M四个完整门禁及四轮正式训练；然后gamma/rollout/lambda、tick0..1000课程、最后H4/H8。所有容量门禁均非正式学习曲线。tiny→MapPPO同时扩大critic且改变计算设置，不能把该系统比较全归因actor容量；六档内部固定critic结构和共同协议，不声称critic初始张量一致。
