# early_shared + sparse：分类 critic 与精简观测消融

2026-11-09。当前实现位于 `codex_dev_clean` 工作区；本轮冻结源代码、SHA256 清单、配置及相对启动前工作区的补丁保存在 `outputs/experiments/sparse_features_20261109_actual01/`。不覆盖旧实验。

[动态结果与曲线](sparse_features_20261109_results.md)。这份文档解释设计与比较口径；实际测量以结果文档为准。

## 公共条件与六组实验

每组单种子 `2026102901`、8×H200、每卡 2,048 个真实对局／4,096 条玩家轨迹、T512/E1/M256、物理与逻辑 minibatch 都为每卡 8,192；224 轮 PPO，共 3,758,096,384 条新玩家样本。没有 advantage 筛选，也没有历史池或训练 baseline 对手。

全部 9×9，使用与旧 v11 相同的地图课程和地图生成缓存，课程按真实 board ticks 在 71,303,168–306,184,192 间推进。episode 上限 2,000；胜负 ±1，其余奖励从第一步起全部为零；gamma=1、lambda=.95，奖励/gamma 课程关闭。训练中途仍用 rollout 边界更新。

early_shared：width128/depth4/heads4，patch1，一格一个 token，critic 使用特权地图 embedding 加局部 embedding，再经过同一套 Transformer。PPO 使用已验证的 actor/critic 分次反向，共享梯度累加后只做一次 clip/Adam；不增大模型、token 或 batch 预算。BF16 + compile，inference chunk=2,048。自适应 categorical KL 学习率仍开启，初始 1e-4，范围 [1e-5,1e-3]，desired KL=.01。

| Run | Critic | Actor 神经网络空间输入（对齐后） | 额外改动 |
|---|---|---:|---|
| v24_9x9_sparse_baseline | 原标量 MSE | 68 通道 | 以上公共配置，无额外新特征 |
| v25_9x9_sparse_hl128 | 128-bin 线性 HL-Gauss | 68 | bins∈[-1,1]，sigma=.04 |
| v26_9x9_sparse_binary_ce | 两端点 soft CE | 68 | 单 logit，端点 -1/+1 |
| v27_9x9_sparse_compact | 原标量 MSE | 28 | A：精简当前图、短期差分、记忆；FP16 transport |
| v28_9x9_sparse_temporal | 原标量 MSE | 40 | B：A + 8帧差分、敌兵记忆、公开统计 EMA |
| v29_9x9_sparse_dedup | 原标量 MSE | 40 | C：与 B 相同，只去除 2 兵源格的重复半兵动作 |

v24–v26 的 actor 初始化及采样 RNG 起点一致，critic 头的参数数量允许不同。A/B 的 embedding 输入维度不同，不把它们称为相同初始化。三个特征实验均保持标量 critic，避免把两类改进混杂。A 同时改变表征和存储精度，是“方案级”消融；不能把 A 的胜率变化单独归因于某一个通道。B 对比 A 验证时间特征组合，C 对比 B 单独验证动作去重。

## 与 AverageJoe 的观测和动作比较

论文为 [Superhuman AI for Generals.io](https://arxiv.org/html/2606.23348v1)，公开实现固定在 [AverageJoe 5041e635](https://github.com/strakam/AverageJoe/tree/5041e635dcf6c87b8b844cb2e244ad3e0abe40d7)。本地源码副本在 `outputs/paper_reviews/Rb79jWnVDX/AverageJoe/`。此前完整调研见 [论文比较](averagejoe_paper_comparison.md)，其中旧 v32 网络参数不代表本轮网络。

| 项目 | AverageJoe 源码 | 本轮基线 | 本轮吸收或保留 |
|---|---|---|---|
| 地图历史 | 当前/事实24通道 + 7步己/敌兵差分14通道 | H4×13 + K11 + 己王XY2，65有效通道补至68 | 当前/事实只用一份，选择性保留动态差分 |
| 连续量 | 兵力等主要 /50，age log | log-army、log-risk、signed-log-margin；规范化 clock/public | 保留我们较稳定的量纲和战斗特征 |
| 全局记忆 | 敌军总兵与领地各512步，两个MLP token | 仅最新public13 | B/C 用16/64/256 tick半衰期的EMA残差，6个标量，不增加token |
| 隐藏敌军 | 最后看到正数敌兵与age；重见空格仍可能保留旧记录 | last-seen归属/地形/结构/age，无兵数 | B/C 存最近可见敌兵量；重见己方/空格/中立格会清除敌兵通道 |
| 动作 | 每格9槽，包括每格PASS，flat categorical | 全图单WAIT + 每格8移动，WAIT/MOVE分解 | 保留单WAIT及条件MOVE，避免重复PASS影响先验 |
| 可变地图 | padding到24，3×3 patch，固定位置表 | 有效格mask、动态画布、patch1 | 继续用有效格/token mask，不引入额外padding信息 |
| critic输入 | 与actor同局部信息，分类value头 | 全state + actor局部历史/记忆 | 本轮保留early_shared特权critic，单独消融输出头 |

作者具体特征与状态更新来自 [networks/common.py](https://github.com/strakam/AverageJoe/blob/5041e635dcf6c87b8b844cb2e244ad3e0abe40d7/networks/common.py)，动作/头来自 [transformer.py](https://github.com/strakam/AverageJoe/blob/5041e635dcf6c87b8b844cb2e244ad3e0abe40d7/networks/transformer.py)。本轮不同时引入 patch3、EMA 模型评估、top25% 筛选或论文 timeout workaround。

### A：compact

最新局部图取12个通道：visible、mountain、city、general、self、enemy、empty、neutral、log_army、enemy_attack_risk、combat_margin、fog_obstacle。删除可由其余类别推断的 plain；保留官方的山/城混合迷雾提示，不泄露迷雾内类别。

3组相邻帧的己方／可见敌方 log-army 差分，共6个通道。差分可以反映移动、生产、战损或显隐变化，不能把敌兵消失直接解释为死亡。当前 visibility 与记忆一起提供判断依据。

历史知识只向主干提供8个通道：explored、mountain、last_seen_city、last_seen_enemy_general、last_seen_empty、last_seen_neutral、last_seen_enemy_owner、last_seen_age。已探索但不属这些类别的归属可推断，当前己王始终可见；所有记忆仍按官方规则在重新看见时修正。加己王相对XY，共28通道。公开 clock、计分板和 king-threat-distance 仍走 context，不重复铺满地图。

### B：temporal

在 A 上把原始历史窗口扩为8帧，因此为7组己/敌兵差分，共14通道。再加最近可见敌兵量的 log 通道，共37个有效空间通道，补至40。

另维护敌方公开总兵力与领地的三个 EMA，半衰期16/64/256 tick，输入“当前值−EMA”的6个残差。总兵力沿用 log1p/10，领地按有效格数归一化。统计只用公开 scoreboard；按实际 tick 差更新，重复查询同一 tick 不重复衰减。每局重置，各座/各lane独立，搜索 clone 同样独立。相较复制512步两条队列，6个数更省存储；损失了精确长序列次序，这是待验证的取舍。

### C：temporal_dedup

B 的观测、网络及初始化完全相同。源格只有2兵时，full传1兵，half也传1兵，屏蔽重复的half；3兵及以上仍保留两种模式。原生编码继续是 `WAIT=0; MOVE=1+source*8+direction*2+mode`，无需更改环境或动作解码。mask 仅用己方源兵数，源格相邻目标本来就在己方视野内。

这不会减少649个输出槽的张量大小，主要检验有效动作支持及探索熵；不承诺单靠去重提高计算吞吐。

## 存储与计算：避免伪节省

原 rollout 已按时间去重 H4 历史，不能把“四帧减少到一份”误算为buffer节省4倍。新方案继续存原始帧一次，在 minibatch/前向时构造差分，H8也按相同机制去重。内部传输的原始 schema 保留13通道，模块化的 `policy/features.py` 决定网络真正使用什么。

A/B/C 的二值量继续 bit-pack；连续量在 rollout 行动前量化到FP16再还原FP32，编码器、特权encoder和知识通道保持同一量化口径。buffer保存FP16，gather后得到行动时完全相同的FP32值。尤其 combat risk/margin 是连续特征，不能当身份二值位压缩。实验基线及两个分类critic继续使用原FP32 codec。

token数和Transformer主体不变，因此本轮主要节省输入embedding和buffer；attention开销不会按通道数同比下降。B还增加动态记忆更新与更长窗口，必须以完整rollout+PPO测量判断净收益，不能仅报告推理时间。

## 分类 critic 接入 PPO 的四个边界

1. **GAE单位不变**。HL 用 `sum(softmax(logits)*centers)`，binary 用 `2*sigmoid(logit)-1` 得到原奖励单位的标量。lambda=.95、真正terminal吸收、reset前timeout bootstrap保持原实现。gamma=1不意味着未结束对局的timeout应标为平局。
2. **CE替代价值回归**。分类组关闭 value clipping，actor ratio clipping不变；target为detach后的GAE return。MSE组保持原clipped MSE，正式组value_loss_coef分别为MSE=1、HL=.01、binary=4。初始系数均为1的pilot发现HL共享梯度远强、binary偏弱，因此按初始共享梯度量级校准固定系数后复检；没有用胜率选系数。训练全程仍监控梯度，不能把该校准解释为全程梯度完全一致。线性支持严格匹配当前纯±1奖励，训练前检查reward；每次GAE后检查目标范围及finite，超范围报错，只有数值尾差允许clamp。
3. **检查共享梯度**。每轮首个物理batch记录未DDP归约/未clip的actor与已乘value系数的critic共享主干grad norm、比值与cosine。不额外前向或反向，只复制一份小主干梯度。该probe是每rank局部诊断，不是全量梯度统计。另记录标量value误差、target越界比例、HL编码偏差和端点概率质量。既有KL、clip fraction、entropy、explained variance、自适应LR继续记录。
4. **buffer只存标量**。rollout的values/returns仍 `[T,N,1]`；bin logits和soft label只在当前物理minibatch生成，避免 `[T,N,128]` 扩张。

HL采用作者 [train/ppo.py](https://github.com/strakam/AverageJoe/blob/5041e635dcf6c87b8b844cb2e244ad3e0abe40d7/train/ppo.py) 的128个等距中心和半bin边界CDF积分后归一化，sigma=.04。**边界截断使 ±1 标签期望向内偏约0.027**，本轮保留论文算法并显式记录，不谎称encoding均值严格等于return。binary soft target为 `(return+1)/2`，预测仍能取[-1,1]内连续值；存在平局或bootstrap误差时不能直接等同实测胜率。

## 评估、预检与产物

每组先以真实2048局/GPU、T512/E1/M256跑3轮pilot，从头初始化，pilot不计入正式224轮。检查有效optimizer步数、有限参数与loss、显存、吞吐，随后用每组导出的actor跑七对手短局smoke。

正式每20轮及U112/U224评估；固定seed271828，每对手100地图×双座，共1400主评估局，附带WAIT诊断不计均值。原四基线 aggressive/expansion/defensive/random，加固定源码包 kestrel-v7/razor-v8/aegis-v2。所有评估采样而非greedy；原生CPU批量环境＋CUDA actor，按8卡分片同步执行。1024tick评估上限延续旧曲线口径，不能与2000tick结果混用。

最佳快照按七对手平均胜率选（四基线均值、aggressive作同分次序）；训练结束后对best/final各使用独立seed94722775、每对手200地图双座，分别评估1024/2000上限。既报告baseline4，也报告heuristic3和all7，不用更容易的四基线均值代替七对手结果。

只保留最近训练checkpoint、少量近期eval快照及最佳快照；每个版本冻结配置/源码/七对手，TensorBoard保留必要诊断。绘图同时按PPO轮数、新玩家样本数和墙钟时间；另报告排除eval的稳态训练吞吐与allocated/reserved显存。

历史v11为PBRS+gamma课程，不是本轮匹配基线；旧独立网络sparse+gamma1也不是同架构。它们可作背景参考，因果比较使用本轮v24。

## 已完成的预检（正式胜率结果另见动态报告）

每组在8×H200、正式环境数与batch下从头运行3轮；吞吐取后2轮，峰值覆盖所有8个rank及编译过程。

| 方案 | value系数 | 玩家样本/s | rollout秒/轮 | PPO秒/轮 | allocated GiB | reserved GiB | 观测buffer GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| v24_9x9_sparse_baseline | 1 | 458,595 | 12.67 | 23.91 | 21.02 | 22.15 | 5.619 |
| v25_9x9_sparse_hl128 | 0.01 | 458,226 | 12.75 | 23.87 | 21.02 | 22.24 | 5.619 |
| v26_9x9_sparse_binary_ce | 4 | 458,082 | 12.73 | 23.90 | 21.02 | 22.15 | 5.619 |
| v27_9x9_sparse_compact | 1 | 459,684 | 12.70 | 23.80 | 18.65 | 19.84 | 3.297 |
| v28_9x9_sparse_temporal | 1 | 443,986 | 13.53 | 24.26 | 19.35 | 20.53 | 3.647 |
| v29_9x9_sparse_dedup | 1 | 446,236 | 13.44 | 24.16 | 19.35 | 20.51 | 3.647 |

这里的观测buffer包括按时间去重的历史和codec字段，不包括GAE/动作等标量buffer。只有2个稳定更新的吞吐样本，正式报告使用长程稳定窗口，不能据此断言1%差距有意义。

A的观测buffer约减少41%，但完整训练峰值只减少约11%；B/C保留更长历史和额外知识后，峰值仍比基线低约8%。token数不变，预检中没有观察到明显整体提速。

分类critic校准前后的首个物理batch共享梯度比（critic/actor）：

| 头 | coefficient=1 | 正式系数 | 校准后实测 |
|---|---:|---:|---:|
| MSE | .0937 | 1 | .0937 |
| HL128 | 6.8024 | .01 | .0680 |
| Binary CE | .0217 | 4 | .0867 |

HL后续两轮比值约.1982/.0214，说明固定系数不保证全程同尺度；继续用TensorBoard观察。三组critic以外的actor初始权重一致；B/C actor和critic初始权重也完全一致。

六组导出actor均完成七对手+WAIT、每对手2张地图×双座、64tick的原生批量smoke，分类头校准后再复检。短局smoke只检验协议/运行正确性，不作为训练胜率。

正式队列于2026-11-09启动，先v24，之后v25–v29；完整224轮的胜率与曲线由动态结果报告记录。
