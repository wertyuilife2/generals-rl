# Auto research：从早期 PPO 到 v32-opt

整理日期：2026-10-21。本文合并历史训练日志、旧实验计划及架构设计，保存研究路线、关键结果、正式方案与证据边界。旧曲线图片和逐版本 JSON 不再纳入源码历史；原文、配置、图像及完整 Git 历史由本次维护的仓库外备份保存，已有 `outputs/`、冻结 release、模型和收据不删除。本文是历史记录，不表示任何旧训练仍在运行。

与外部工作的特征、网络和训练方案对照见 [AverageJoe 调研](averagejoe_paper_comparison.md)，其本地比较对象固定为 2026-10-16 的调研快照。

当前默认已改为 F03 消融最终继承的 ViT1M dense／H4／recent8／缩窄共享LR／gamma课程配方，配置随包保存在 `generals_rl/default.json`。完整结果分别见 [H200 scaling 报告](h200_scaling.md) 与 [F03 消融报告](f03_ablation.md)。下文的 v32／v32-opt 结论保留其历史口径，不代表当前默认仍为 LightCNN。

## 历史阶段：v32-opt

**该阶段选定的 v32-opt 是严格 current-policy self-play、desired KL=.01，而不是中途的 history8 回退。** 在原 v32 的网络容量、输入和 PPO 预算下，完整训练计算吞吐为129,114.81 fresh learner transitions/s，约为 v32 的1.826倍；final 与 best 均通过原配对非劣和直接对战门槛。仍保留 v32 与 v32-opt 配置作为历史对照。

| 项目 | v32-opt |
| --- | --- |
| 游戏与采样 | 9×9、1v1、GPU VecEnv；16,384环境×32 tick；episode最多1,024 tick；65,536地图池 |
| Actor | 本人局部 H4、K11持久知识、主城相对坐标；16通道两层CNN |
| Critic | 独立权重；当前特权状态＋同一份本人H4/K11；输出塑形return的value |
| 动作 | WAIT＋每格8种MOVE；WAIT/MOVE factorization；productive action mask |
| 更新 | StudentPPO；无 teacher 采样或蒸馏；4 epochs×64 minibatches，8,192样本/minibatch |
| 信用分配 | gamma/PBRS gamma=.995，lambda=.95；tick−.001在update0..1000线性归零 |
| 自对弈 | live learner；双方合法观察隔离；对手样本不重复计入 learner transitions |
| 学习率 | 初始1e−4，categorical adaptive、desired KL=.01；PPO监控开启 |
| 执行优化 | 历史/知识/坐标一次拼装、输入对齐64/72通道、共享history解析、projection0、cuDNN benchmark |
| 参数 | Actor＋critic共26,074个FP32参数；未靠削减输入/容量换吞吐 |
| 配置与使用 | [v32-opt.json](../rl/configs/v32-opt.json)、[v32.json](../rl/configs/v32.json)、[RL说明](../rl/README.md) |

## 研究路线与工程决定

v1–v8含续训、warm-start、旧网络迁移和少量baseline训练对手，评估也有greedy历史协议；这些结果不能直接当作后续从零 sampled 实验的公平对照。v9起每个方案均随机初始化、新Adam、新历史池；选模与开发地图曾反复复用，单训练seed结果不能推出跨seed因果收益。

| 阶段 | 检验内容 | 结果和后续处理 |
| --- | --- | --- |
| v1–v3 | MoE CTS参考、productive mask、student-only PPO、地图重采样 | 正式路线转为全部student采样/更新；保留公共runner/storage适配，不再支持早期dense/MoE模型加载 |
| v4–v8 | 可见战斗特征、proper PBRS、少量baseline训练对手、主城威胁距离、轻量CNN | 保留局部combat/threat、PBRS和LightCNN；baseline只用于评估，旧权重迁移/蒸馏及混合对手路线退役 |
| v9–v11 | 从零纯历史自对弈、dilation3、己将相对几何 | 保留普通3×3卷积和own-king geometry；扩张卷积未成为正式方案 |
| v12–v16 | 扩大环境数、greedy历史对手、等价HALF去重、已见敌将记忆、降低重复更新 | 未形成稳定的最终收益；恢复随机历史对手和原动作协议，记忆由后来的K11统一表达 |
| v17–v18 | WAIT/MOVE分解、将军局部context | 保留WAIT/MOVE；v18后期退化促成categorical adaptive KL及更新监控，旧context分支退役 |
| v19–v22 | Actor K11与critic读取H4/K11的2×2信息消融 | 信息正确性成立，未证明tiny网络稳定胜率或计算效率收益；按信息语义保留H4/K11＋组合critic作为后续共同协议 |
| v23–v26 | 成熟ResNet/ViT的1M与10M容量实验 | 当时有限wall预算内更新数很少，不能据此否定容量上限；v26按用户要求中断，不计完整条件 |
| v27–v29 | 固定gamma .995/.999与.995→.999课程 | 两固定组best开发aggressive均90.25%，课程82%；选择较快的固定.995，不宣称全局最优 |
| v30 | N4096/T128对照N16384/T32，保持NT | 训练wall吞吐低18.28%，best开发aggressive88.25%；恢复T32。环境数和GAE上下文同时变，不是纯执行优化 |
| v31 | lambda .99对照.95 | best开发aggressive85.75%，相对v27−4.5pp，配对区间[−8.5,−.75]；保留.95 |
| v32 | 仅tick惩罚在update0..1000归零 | 早期学习更快，最佳开发点较高；零tick后未持续改善，仍保留为候选工程方案 |
| v33–v36 | 匹配小批量H4/H8 | v34意外中断、v35用户暂停，均排除完整比较；v36 H8明显退化，保留H4，H8能力保留给后续F03消融 |
| v32-opt | 输入/推理/更新优化；current与history8；降低current KL目标 | KL.02 current失败、history8 best中途接纳；最终KL.01 current完整通过，成为正式入口 |

旧analytic critic value-potential offset、projection32 LightCNN、standalone actor-budget MapPPO及旧backbone presets均不属于正式支持范围。它们的退休不改变PBRS、当前Scaled架构、geometry、MCTS接口或独立规则引擎。新训练的 `retain_seed=false` 默认允许淘汰初始随机模型；缺字段旧manifest仍按true解释，不能用新默认改写历史实验。

## v32 的完成结果

v32采用H4/K11、组合critic、LightCNN、纯历史池8和KL=.02。当时初始随机策略永久保留，其余7位滚动更新，每50次update发布，整局固定随机对手。采样预算为N16,384/T32、gamma=.995、lambda=.95，4 epochs×64 minibatches。

历史奖励为 `r_terminal + .995×Φ(s_next) − Φ(s) − .001×max(0,1−u/1000)`，其中 `Φ=.05×领地+.2×城市+.001×兵力`，胜/死奖励为+1/−1。真正终止时下一状态势函数为0；1,024tick时间截断保留势函数，并用reset前终态bootstrap，不人工判负。课程只衰减tick惩罚，PBRS与终局奖励始终保留；u999仍有−.000001，u1000起才为零。

| 训练指标 | 实测 |
| --- | ---: |
| 完成更新 | 1,206 |
| fresh learner transitions | 632,291,328 |
| 完整wall | 9,004.021秒（150.067分钟） |
| 采样＋更新吞吐 | 70,716.28 transitions/s |
| 全程wall吞吐 | 70,223.04 transitions/s |
| Torch peak allocated/reserved | 15.26401/16.32617 GiB |
| selector | 12次全部完成；best为u699 |

历史CPU selector每100次PPO更新评估一次：stochastic、seed/sampling271828，每AI200张地图换座共400局，四AI合计1,600局，FP32/B128、max2,000tick，观测时钟仍按1,024tick归一化。按 `(四AI最低胜率, aggressive胜率, overall胜率)` 字典序选模，因此best不一定是overall最高点。u从0编号；best/u699即第700次更新后的snapshot14，累计367,001,600 transitions、约86.43分钟，此时tick惩罚仍为−.000301。

CPU selector的overall由u99的77.5625%升至u199的94.875%，之后进入平台；u699为96.375%，最后u1199为96.3125%。aggressive在u199后为87%–91.25%；u699→1199差−1pp，地图簇配对95%区间[−4.75,+2.75]，不足以确认末期退化。最后一个selector点u1199不是最终u1205权重的独立评测。

另一套CUDA开发协议采用seed/sampling1001042、FP32/high/TF32、B128、相同1,600局规模；best/u699的aggressive/expansion/defensive/random胜率为93.25%/96.75%/100%/99.75%，overall97.4375%，不能与CPU selector数字混用。开发种子曾在其他版本复用，不是未见holdout。相对v27固定tick控制，aggressive+3pp和overall+.75pp的配对区间均含0；v27仅训练930轮，零tick阶段没有等预算对照。u1000..1199确实完成200轮零tick，但没有持续的末期收益，所以不能把最佳模型归因于零tick阶段。地图簇bootstrap区间仅反映给定模型的评测不确定性，不涵盖训练seed差异和多重选模影响。

末100轮entropy均值1.07453、exact KL .025882、clip fraction .135777、critic EV .916213，未见数值异常或动作熵崩塌；高EV不能代替对战强度，末期KL尖峰仍需监控。PPO监控约占更新耗时2.17%。

完整小批量H4/H8比较使用N8192/T32。v33 H4 best开发为86.5%/94.5%/98.75%/99%，v36 H8为37.5%/56.75%/83.25%/77.25%；aggressive差−49pp，地图簇95%区间[−54.5,−43.5]。H8 actor和critic同时改变、参数从26,074增至39,898，只有单训练seed，不能推广为所有网络长记忆无效。v36计算吞吐70,039.68/s但评估排空后的wall仅35,966.54/s，说明必须区分计算与全流程成本。

v32的冻结来源为原提交 `817cb47e9ed5cabfeb3adf8064e3946b4d56fc05`，best SHA256为 `a1c3c940acdfd6bc92d2e1a383c140b78502a583641506de7867780f31cdaefc`。原始产物保留在 `outputs/moe_cts/iterations/v32/`：`train/summary.json`记录训练预算，`final_best/development_report.json`记录CUDA开发评测，`diagnostic/final_tick_curriculum_comparison.json`记录v27配对比较；`frozen_config.json`与`source_manifest.json`记录历史配置/源码，当前默认配置不替代这些冻结输入。

## v32-opt 优化和最终验收

输入组织优化将同一逻辑的actor61/critic71通道对齐为64/72，packed解析供两分支复用；critic没有额外压缩投影。固定推理与更新batch后使用cuDNN算法选择。改变对手为current减少历史分组和模型调用，但每tick双方仍需决策，且PPO更新占据大量时间，不能按分组减少比例推算端到端收益。

在一次固定联合动作的H8/N8192对照中，历史池→两个整批current actor，采样吞吐243,264→约308,000/s（+26.5%），完整采样＋GAE＋PPO仅71,403→76,047/s（+6.5%）；合并成一个更大的双座batch为75,046/s。该测量排除评估与I/O，只隔离执行成本，不证明current自对弈的学习质量。

| 完整训练，同1,206更新/632,291,328新样本 | 采样＋更新 transitions/s | 全程wall transitions/s | 效果验收 |
| --- | ---: | ---: | --- |
| v32 | 70,716.28 | 70,223.04 | 对照 |
| current / KL .02 | 130,317.38 | 119,115.76 | 原NI10项全失败；final/best直接对v32仅8.75%/6.6875% |
| history8 / KL .02 | 112,532.93 | 109,459.28 | best对v32 54.75%，best NI通过；final NI及直接对战失败，按用户调整先接纳best |
| **current / KL .01** | **129,114.81** | **126,918.08** | **原final/best配对NI10/10、直接对战及后段趋势全部通过** |

最终current .01从随机权重、新Adam训练，原生exit0。trainer wall4,981.885秒，collection1,233.059秒、update3,664.065秒，GPU峰值allocated/reserved20.841/22.182 GB。各轮实际轨迹和局长不同，表中整体倍率不作为单一组件的因果加速比。

测试前冻结final/u1205与selector best/u1099。每端点1,000地图/AI×4AI×2seat＝8,000局，CPU FP32/B128、stochastic、seed/sampling271828、max_ticks2,000：

| checkpoint | aggressive | expansion | defensive | random | overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| v32 final/u1205 | 89.65% | 96.40% | 98.90% | 99.50% | 96.1125% |
| **v32-opt current final/u1205** | **90.80%** | **96.65%** | **99.25%** | **99.90%** | **96.6500%** |
| v32 selector best/u699 | 88.65% | 97.15% | 99.40% | 98.95% | 96.0375% |
| **v32-opt current best/u1099** | **93.05%** | **97.45%** | **99.40%** | **99.70%** | **97.4000%** |

地图簇配对bootstrap20,000次，原每AI3pp/overall1.5pp非劣容差10/10通过。final总体差+.5375pp、单侧95%下界+.125pp；best差+1.3625pp、下界+.95pp。测试扩展复用selector种子，部分地图/赛程重叠，不能称完全独立holdout；另保留map_index200–999的6,400局诊断，同样10/10通过，此事后诊断不替换原gate。

直接模型对战每端点400地图换座共800局，CPU FP32/B128、相同map/sampling种子、max2,000tick：final对v32 final为530W/270L/0D，得分66.25%、95%区间[63.0%,69.375%]；best对v32 best为571W/229L/0D，71.375%、[68.0%,74.625%]。单侧下界63.5%/68.5%，均超过原45%门槛，原生exit0。

后半程六次selector overall为96.5625%–98.1875%，最后u1199为97.125%。u1000..1199全部真实零tick，exact KL均值.01021、局长199.85、截断均值.253%；旧current .02为.03450/471.47/32.786%。降低live更新幅度、缓解对手非平稳性与本单变量结果一致，但没有多训练seed因果证明。

final SHA256为 `0e7c3da6e83fe0e56662fd0db5169f7c565c7831e103b5d5645a9ee9316c8d00`；best为 `122ef6fa9066cc4cf3d35c4eea46ff02b3c28bd5c026708451fc422e9b5513ab`。完整证据位于 `outputs/analysis/v32_opt/pure_current_kl001_actual01/attempt02/`：`completed_training_review_actual01.json`、`acceptance/paired_noninferiority.json`、`crossplay_actual01/{final,best}/report.json`及`accepted_release_actual01/local_quality_gate.json`。失败current及history8对照也在原输出中保留，不改写为成功。

## H200 与 F03 的保留能力

多卡采用同步PPO：每rank独立模拟与随机流、初始权重同步，按全局样本归一化advantage/调节KL，logical minibatch梯度平均后统一clip/Adam；rank0负责快照与文件，其他rank只读刷新池。支持LightCNN及Scaled的1M/10M/100M **actor＋critic总预算**，BF16只作用于网络计算；physical microbatch/inference chunk不改变logical预算。

F03不是直接读取 `v32-opt.json`。它的冻结基线为 `ScaledInformationSpatialPPOPolicy`、ViT1M、H4/K11、独立组合critic、projection0、BF16和编译，阶段配置从冻结 `F03_run_config.json` 生成。七阶段当时比较了以下候选：

| 阶段 | 候选 |
| --- | --- |
| S1 | live current与recent8 episode-frozen历史池 |
| S2 | actor自适应LR上下界、独立固定critic LR及组合 |
| S3 | desired KL .01/.02 |
| S4 | 固定gamma .99/.999及.99→.999课程，PPO/PBRS同步 |
| S5 | ViT10M dense/pyramid |
| S6 | 匹配ViT1M与ResNet1M dense/pyramid |
| S7 | actor与critic H4→H8 |

2026-10-21只读核验的H200状态文件显示S7/completed、48个run completed、current_job=null；该信息只描述当时该实验文件，不代表扫描整机或重新验证全部模型。S1原自动记录为live，实际经单独有效裁决继承recent8；后续S2选择lr_bounded、S3保留KL.01、S4选择gamma_course、S5–S7保留该配置。当前源码已退役七阶段调度、专用诊断和自动晋级工具；批量端点评估器继续保留。实验工具源码归档在 `outputs/archive/f03_ablation_tools/20261021/353812d.tar.gz`，原冻结release、receipt、配置、权重及结果保持原样。

### 当前Scaled架构与匹配结论

骨干复用timm 1.0.30（Apache-2.0）中的ResNetV2/GN/WS与ViT/Pre-LN/LayerScale/SDPA，无ImageNet预训练权重。ViT一格一token、ResNet保留地块输出；dense/pyramid与位置插值在当前适配器中实现。1M指两条独立encoder连同heads总参数：ViT1,081,258，ResNet1,057,130（当前projection0），不能混用已退休actor-budget presets。

H4/9×9的actor为48历史＋11知识＋2几何平面，对齐64；critic为当前特权12＋局部历史48＋知识11，对齐72，最新public经独立context旁路。没有running observation normalization；LayerNorm/GroupNorm仍是骨干内部运算。K11是当前记忆，不堆叠四份、不保存历史兵力；critic没有特权H4。两个encoder及head不共享权重。

2026-10-20审查的配对实验使用三个seed（20261013/14/15）、各224轮、8rank/globalN131,072/T32、939,524,096个fresh样本、recent8冻结对手；同seed配置仅run_id/backbone不同，执行编译方式差异仍需计入性能解释：

| 模型 | Best对F03得分均值±样本std | Final对F03得分 | Final基础AI最弱/总体 | 记录训练wall吞吐 |
| --- | ---: | ---: | ---: | ---: |
| ViT1M dense | 61.6250%±2.8174pp | 70.2500%±2.3419pp | 98.8333%/99.5833% | 143,172 transitions/s |
| ResNet1M dense | 4.0000%±1.9244pp | 4.0000%±1.9244pp | 40.6667%/66.1667% | 45,686 transitions/s |

F03得分为胜1/平.5；每best/final分别800局F03＋800局基础AI。ResNet三个best均为最终224轮，不能怪选模太早；它仍在改善，结果只证明当前配方固定预算的差距，不证明更长训练上限。wall包含初始化、评估收尾和同步，排除后续端点评估，不是env.step或纯采样吞吐。

ResNet呈更高MOVE KL/clip/entropy和更多中段timeout，LR长期卡下界、完全只WAIT、只因纯current等解释不被数据支持。8通道窄stem＋16块深卷积、GN/初始化、边界处理和内容相关远距离交互是待检验假设，没有逐层/算子profile不能声称确定根因或分摊3.13倍wall差。后续优先在同一冻结obs上比较KL及critic校准，再独立试加宽stem/减少block、小残差初始化、空间mixing和编译优化；不要一次改变多因素或通过朴素共享encoder泄漏特权信息。

完整审查原文 `doc/network_1m_dense_review_20261020.md` 作为本地WIP保留、不加入历史；统计原始导出与复现脚本在 `outputs/analysis/network_review_1m_dense_20261020/`。论文比较文档独立保留，因为它不是重复实验日志。

## 评估协议、已知限制与源码清理边界

历史4个baseline仅为回归对手，高胜率不证明世界冠军强度、未见地图尺寸泛化或对所有rule-based策略的优势。训练自对弈胜率、baseline胜率、模型对战draw-adjusted score不能混为一个指标；selector与开发复用、单seed和不同训练预算均须保留说明。典型失败涉及更早的扩张、合兵和回援，不宜只将最后WAIT归为根因；记忆正确也不等于模型有效使用。v32八例录像重放核对了1,257个决策tick和局部H4，未独立重建K11，不能据此断言模型忘记了已探索敌王；defensive输局来自另设的397胜/3负诊断，未混入主开发集。

CPU与GPU比赛均调用现有离线env；arena提供对战/回放/Elo，批量RL评估器提供当前网络加载、history/knowledge和GPU分片。原neko C++基础AI合并曾在64局/6,062 joint ticks上核对动作和状态digest一致；该小规模CPU评估总耗时约1.219倍改善，不能外推H200正式任务。新的批量native controllers还去除了旧Python worker池；`baseline_workers`仅保留输入兼容。官方迷雾/其他规则对齐须单独评估，不能将后来在线规则更新默认为训练后端已改变。

当前保留v32/v32-opt和Scaled H4的checkpoint协议、训练评估和公共基础；固定critic LR与Scaled H8专用分支退役，旧实验使用其冻结源码复现。此次整理不修改env规则、不部署H200、不删除既有outputs。未来MCTS保留算法独立`PolicyValue`和CPU整tick分支模拟器；selection/backup、belief及搜索训练尚未实现，搜索不得将特权全图误当局部观测。

源码测试随保留功能组织：规则一致性、FOW隔离、当前模型/输入、PPO更新/恢复、历史池、分布式/评估协议和搜索分支仍应覆盖；旧模型迁移/蒸馏、旧配置和废弃架构分支专属测试成组移除。原始提交列表、清理候选审计及维护过程文档只作为本地资料，不再成为额外提交的研究记录。
