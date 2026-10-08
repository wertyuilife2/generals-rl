# v19 起：信息、容量与长期信用分配实验

用户需求来源：2026-10-07 附件 pasted-text-1.txt，以及 doc/temp_plan.md。用户补充优先于旧方案中的实验顺序与奖励衰减建议。所有正式训练从随机权重、新 Adam、新历史池开始，不加载旧权重；工作分支 codex_dev，main 不修改。

## 当前执行路线（按用户最新调整）

2026-10-07 用户要求暂缓耗时过长的 ResNet/ViT 架构探索。v23–v25已完成结果保留，v26在首轮后按请求中断，100M门禁及正式训练暂不继续；这些未完成项目不作为架构结论。下面的容量记录保留历史安排，当前执行改为小网络的固定gamma→gamma课程→rollout/lambda→tick衰减→H8。

以v22 H4K11 / SH4K11和InformationSpatialPPOPolicy为共同基座，三种gamma实验与终局开发、八录像及完整曲线均完成。v27固定.995：930更新/115.005分钟；v28固定.999：1133更新/139.982分钟；v29 .995→.999：1435更新/177.879分钟。三组开发aggressive为90.25%/90.25%/82.00%，overall为96.6875%/96.6875%/93.0625%；实际相同采样协议下，课程相对两个固定控制aggressive−8.25pp，配对地图CI均不跨0。固定两组best开发没有明确差异，而.995早期学习更快，因此**后续采用固定gamma.995**。这是单训练seed的工程选择，不声称最优gamma或两模型等效。保留全部结果和真实预算限制，详见rl_train_log.md。

v29真实u1000..1199共200轮γ.999/tick_scale1，104,857,600转移、1492.456秒，已通过TB验收；best999的实际γ=.99899601936，但后续目标阶段确实完成。14次selector全部completed0；原87039真实exit0、observer也exit0，正常平台STOP，不重启。源码新增return/raw advantage/critic EV为监控，不修改目标或RNG。课程后期aggressive/expansion回落，末段高EV并不证明终局价值准确。失败样本显示可见home威胁下分配/回援不足，知识保留正确不等于策略有效使用。

**v30完整结束并选择原rollout基座。** N4096/T128实际820更新、429,916,160转移、124.086分钟；八次selector全部completed0，best399的selector88/97/98.25/98.5%，独立CUDA开发88.25/97.25/99.25/99.5%、overall96.0625%。原43364正常STOP并真实exit0，observer正常退出/评测队列排空；immutable capture、开发1600、2615tick八实际录像/Core行为、终局last100/300、完整sample/wall与v27配对报告均完成。后六公共采样点aggressive CI均含0；开发aggressive相对v27−2pp CI[−5.75,+1.75]，overall−.625pp[−1.875,+.625]。全程wall吞吐57,744.4，较v27低18.28%，中段低20.86%，成本主要collection；峰值allocated13.5899GiB较v27少1.7033GiB。选择恢复v27 N16384/T32、NT524288/γ.995/λ.95为后续工程基座，单个λ.99候选接下来从零训练；不宣称纯T因果、训练随机性显著、长rollout本质无效或最佳开发同强度。失败录像显示较早部署/回援与长对局资源问题，最后一tickWAIT已不能补救。详细结果与三截图见rl_train_log.md。

当前v31仅λ.95→.99，已依据完整v30恢复v27 N16384/T32，其余配置固定；a2d2447配置/116Git blobs冻结后正式从零运行original60387，actual N/T/γ.995/λ.99、新历史池及compiler确认。前四个1600局采样selector99/199/299/399均completed0，aggressive34.25→75.50→78.50→85.25%，overall67.125→91.125→91.3125→94.4375%；第四点其余AI95.75/98.75/98.00%，同样本v27 aggressive84.5%、overall94.8125%，本点已追近控制，仍是中途结果。相同全局采样控制v27 λ.95的首点aggressive差+0.5pp CI跨0，第二点−6.25pp CI[−10.75,−1.75]、overall−2.0625pp[−3.6875,−.4375]；同点训练wall接近，显示本阶段学习较慢，未说明最终收益或跨训练seed显著性。数值早期0..29有效、wall速度差仅−.217%；R/A/EV为v27未记录的新指标，不补造比较。当前趋势锚点399/85.25%，后续计数重置，仍在学习且未到最低1h，原训练继续；完整终局开发/八录像/失败/全samplewall报告后才选择tick课程基座。在所选gamma.995时，TD残差几何权重1/(1−gamma×lambda)由约18.26增至66.89tick；这只是GAE递推权重，不是bootstrap学习时间上限，不预言胜率。比较raw return/advantage、critic EV、KL/clip/自适应LR与实际吞吐，不做多值搜索。

再做用户指定TB0..1000的tick−.001乘linear1→0，其他PBRS资源/胜负保持固定；必须真实完成u1000..1199至少200轮零tick及目标阶段评测，3h预算优先且不足如实报告。若选定设置在预算内无法完成，则先做保持配对的较小NT constant-tick控制再做衰减，不偷偷缩短1000日程。H8最后，按实际rollout缓存与显存验证可执行性；若需减NT，同样补fresh H4控制。剩余容量探索暂缓，整个迭代目标仍继续。

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

容量正式执行更新：v23配置2c3a691、v24配对配置e817a78均已独立提交；各自116运行文件和当前3个Map/CUDA分析工具从各自commit冻结，配置SHA绑定并回读。v23 original session12099/PID3436959已从零启动，自身实际seed0的328张量与训练seed20261013独立新模型逐位一致、pool resume_source=null，CPU初始化审查64696/aa0f29 exit0。v24在单卡上等待v23真实终局，之后从零运行。v23首个iteration3正式1600局采样panel已completed/returncode0，胜率1/8/7.75/13.25%；仅4轮更新、2097152转移，actual TB完整wall1264.937秒含402.299秒同步eval，不足以判断容量效果或平台，训练继续。后续tick课程必须保留用户指定TB0..1000日程并留足最终阶段训练，不能在慢网络只完成少量更新时声称课程已经验证；根据容量结果选择预算内可完成日程的基座。

v23已真正终局：original12099/62f8ca exit0、20更新、10485760转移、640tick/env、103.85分钟，既定evaluation_plateau停止；5次selector均completed0，best19为2/8/26.25/17.5%，复用开发1001042为2.5/7.75/24.75/14.75%。最终best不可变捕获、CUDA开发23960/becc7a exit0、CPU原动作Core八录像88415/397460 exit0（4857tick）、完整sample/wall截图与终局PPO/历史池/失败分析已记录。compute2451.59/s、wall1682.89/s，更新占C+U89.29%；640tick/env与少量reset使历史池周转有限，最终seed0占81.55%活动lane但不能推成整轮样本比例。当前时间预算内未学好baseline，不证明大网能力上限；v24按冻结配对配置从零运行。两种1M正式训练都实际结束后再做实际性能诊断，随后推进10M/100M完整门禁及正式训练。gamma/rollout/lambda、tick0..1000课程和最后H8范围不变。

v24已实际从零启动original66320/PID3532285，own e817a78冻结源/配置预检通过，GPU先空闲；自己的初始246张量与新构造逐位一致，CPU84206/301ed9 exit0，116文件/12模块均自身冻结源。首轮524288转移、实际246Adam state均step256，physical512/logical8192/4096chunks/BF16符合配置（b9647a exit0）；C15.012/U150.514秒，compute约3167/s。尚无baseline panel，首个在iteration3；原训练继续，不复用v23胜率或初始状态合同。

v24首个iteration3采样selector现已completed0：1600局，aggressive/expansion/defensive/random为0.25/5.75/3.75/5.25%，post-sync wall17.01分钟、已含349.798秒worker。同2097152转移下早期胜率低于v23、完整wall少19.31%，仅一点评估不证明架构优劣。9ae146 CPU聚合/TB/SHA读回已完成，原66320仍live；继续执行预设共同1–3h/平台规则，之后完成v24终局及1M pair量化比较和实际性能诊断。四个更大容量门禁/正式训练、gamma/rollout/lambda、tick0..1000与最后H8仍待完成。

容量配对量化入口已准备（ignored `full_capacity_preparation/capacity_pair_preparation/compare_terminal_pair.py`，SHA `51091be53c237259af8cf0ed163e82bd4d08ed50eff64f40b50d715e11304b94`）：仅接受两份真实终局review，ResNet/ViT同档位、同critic参数及共同运行实现，配置仅run_id/backbone可不同；复用已有按地图簇配对统计，比较确切共同transitions的实际panel。wall比较只列截止分钟前的真实panel和时差，不插值假造共同点。CPU help/compile通过，尚未执行完整输入load或v23/v24终局pair，不能算已完成容量结论；最终可按实际输入修正必要缺口。工具不修改正在运行的源码。

v24已完成四个采样selector（iteration3/7/11/15，均1600局completed0），最新1.25/14/8.75/11%，post-sync完整wall67.67分钟、8388608转移。第四点CPU聚合/TB/身份审查97be2f exit0，root68e89b回读；原66320仍live，未达到至少五个panel的平台停止条件。其余三项成绩仍低但上升，aggressive未持续改善；保持本轮冻结协议，终局后再做开发/录像/完整曲线。准备性能trace的CPU分析不占GPU；实际profiler仍须等两轮1M正式训练都退出，四个更大容量实验和后续课程/H8尚未完成。


v24正式训练及1M配对已完成：original66320/22f0eb exit0，20更新/10485760转移/84.70分钟，5个完整采样selector，best19为2/12/8.75/11.5%，复用开发为2.5/11.75/12/9.5%。终局review、八录像2528tick/真实Core核验、失败分析、完整sample/wall图已记录训练日志。相同样本的ViT compute约1.290×、wall约1.226×，expansion较高、defensive/random较低，aggressive均2%；单seed浅历史池不构成架构上限。真实终局CPU pair274b77/exit0及地图簇CI完成，没有合成wall点。root空闲GPU开始ResNet1M性能诊断original63882，之后ViT；诊断不是新容量门禁或正式训练。四个10M/100M门禁/正式训练、gamma/rollout/lambda、tick0..1000课程和最后H8仍待完成。


两家1M profiler均actual exit0，各完整256Adam；CPU trace分析完成。固定batch512 actor p50为ResNet7.702ms、ViT3.884ms，不能外推self-play速度。kernel归因缺口约60–70%，同步等待不直接等于可省成本。最小mask验证候选ef2b38c仅改变3个adapter、30个CPU checks通过；root已从该commit冻结116文件。先做无profiler短窗口GPU ABBA验证实际收益，再按实测配对门禁选择10M/100M执行设置；其后的全部课程和H8范围不变。


两家1M无profiler mask ABBA已actual exit0：ResNet逻辑批均值0.739740→0.726952秒（−1.729%），ViT0.584546→0.570156秒（−2.462%），完整model/gradient/Adam/LR/loss/monitor/RNG逐位一致。短8批诊断不是完整容量/正式学习；保留ef2b38c最小补丁，10M ResNet完整门禁original39678已启动，随后按相同设置执行ViT。全部更大容量正式训练及后续gamma/rollout/lambda、TB0..1000课程与最后H8仍待完成。


10M ResNet完整容量门禁original39678/f121a9 actual exit0（NT524288、256Adam/clip/observer、24驻留、save/额外publication、1600采样对局），完整wall16.265m；ViT10M同设置original88259/fb17d4已启动。v25/v26草案仍在ignored preparation，未选择或启动正式训练；等完整配对成本后共同确定评测cadence/timeout/reserve，分别commit/freeze并fresh训练1..3h。后续范围不变。


两种10M完整门禁均actual exit0（Res39678/f121a9、ViT88259/a28934），CPU实际paired selection3f99fd完成：BF16/p256、pub2/eval每2updates、timeout900/reserve1680、min1h/max3h，共同配置仅runid/backbone不同。目标五点+余量9132s；明确约36.6%评测占比取舍和随机门禁吞吐限制。接下来各自commit/freeze，fresh v25/v26正式训练，容量验证不代替学习曲线；100M和后续课程/H8仍待完成。
