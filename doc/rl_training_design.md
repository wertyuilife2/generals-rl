# RL训练设计

运行接口和当前默认值见 [RL README](../rl/README.md)。本页说明可复用的实现边界；已完成实验的配置、预算和结果以其报告为准。

## MDP与可变地图

Actor读取自己的局部地图、历史、记忆与公开比分，critic可额外读取当前完整棋盘。真实地图填充到训练画布；独立布尔 `valid_cells` 控制合法动作、统计归一化、attention与池化。无效格不作为可学习的地图通道，也不堆叠到历史。训练和导出共用构造器，将动作转换回真实地图宽度。

动作保持环境原生编码：WAIT=0，其余为来源格×四方向×全量/半量。mask排除非己方来源、兵力不足、越界、山脉和填充目标。

| 特征方案 | 历史／记忆 | 神经网络空间输入 | 附加公开统计 |
|---|---|---|---|
| stacked | H4/K11 | 4帧局部13通道＋K11＋将军相对坐标，按4通道对齐为68 | 当前比分／时钟 |
| compact | H4/K11 | 当前局部类别、相邻帧己/敌兵力差、精简记忆，共28通道 | 当前比分／时钟 |
| temporal | H8/K12 | 延长兵力差历史、最近见到的敌军兵力，共40通道 | 敌军兵力/领土相对16/64/256 tick EMA的差值 |
| temporal_paths | H8/K14 | temporal＋己王／已发现敌王的地形距离，仍对齐40通道 | 同temporal |

记忆按座位和episode隔离，只用可见地块与公开比分更新；旧敌军兵力不代表当前隐藏状态。新观测可更新地形、建筑和将军身份。局部combat风险与margin仍为连续特征，不按二值存储。

新默认显式设置 `clock_scale_ticks=2048`，经过时间为 `clamp(tick/clock_scale_ticks, 0, 1)`，不随episode上限自动缩放；规则相位 `%2/%50` 保持原样。`knowledge_age_scale=2000` 独立控制记忆年龄并截断到1。两项尺度随actor导出。旧完整配置或policy spec缺少clock字段时保留原来的 `tick/max_episode_steps`，保证旧模型兼容。固定时钟超过尺度后饱和，不再区分更长的绝对经过时间；较长时限训练应明确选择尺度，不能将其当作剩余时间。

路径只依赖本座地形记忆和公开的山/城混合提示，未知提示保守阻断、已知城市可通行；不计算战斗成本。
GPU按地形连通性／目标变化更新缓存；路径在行动时存入rollout，不用后续探索到的信息重算旧观测。
新增通道填在原temporal的padding位置，保持既有输入索引与网络参数量。编码、测试和实验口径见
[路径与时限消融](terrain_distance_20261110.md)。

特征配方由 `policy/features.py` 统一定义H/K、精度和动态统计维数。配置解析负责派生task、policy与内置codec；切换方案时全部重建。内置配方固定历史长度；显式自定义codec及关闭压缩/去重的选项仍由调用方负责。

## Rollout与存储

双座当前策略自对弈中，每局每tick生成两条玩家transition，GAE与奖励按座位分开。一次rollout内参数固定；对局可跨越多轮PPO。自然终局与时限到达分别记录；当前默认在episode时限处按平局0终止，后续价值为0。可选`bootstrap`模式使用重置前终局观测的critic值；未结束对局的rollout边界仍正常bootstrap。

训练桥接层仅在`bootstrap`模式为timeout行保存重置前快照；`terminal_draw`模式无需此快照。结束的对局先重置模拟器和对应记忆，再统一编码一次局部观测、特权观测与动作mask，避免每次有对局结束就重复编码整个batch。timeout快照包含当时的历史、记忆和路径特征，历史推进及局部重置不修改已经交给rollout的张量。`GeneralsTask.step()`默认仍提供完整终局观测；只有桥接层显式使用`capture_all_terminals=False`，此时终局字段仅需bootstrap的timeout行有效；无需快照时复用下一步观测作为占位，PPO不读取该占位计算终局价值。

CUDA动作范围、mask和模拟器状态检查使用设备异步断言，CPU保留异常检查；结束行索引在step内计算一次，供episode指标和reset复用。桥接层将是否存在bootstrap快照作为Python元数据交给PPO，避免无timeout时再次从GPU读取布尔值。异步断言失败会终止CUDA执行，不能依赖捕获该异常后继续使用同一CUDA上下文。

历史去重每tick存一帧，加上rollout起点上下文和episode边界；抽样时重建原堆叠。标准codec按位压缩二值信息，保留FP32连续值。紧凑方案在行动前就量化至FP16可精确表示的值，再以FP32参与推理；存储解码恢复相同输入，避免old log-prob与更新输入因压缩不一致。

存储实现与MDP表示分别配置。旧标准存储验证见[历史去重](history_storage_20261103.md)和[位打包](packed_storage_20261103.md)。

包内默认配方显式设置`task.time_limit_mode="terminal_draw"`，在时限处接零后续价值，并令potential吸收；旧完整配置缺少该字段时，底层`TaskConfig`保留`bootstrap`默认值以兼容历史训练与恢复。Task的`terminated/truncated`仍记录模拟器自然终局／时限原因，训练桥接层独立决定是否bootstrap；两种模式都切断reset处的GAE，rollout边界仍bootstrap，最后一tick自然胜负优先。

## Actor与critic

共享ViT1M主干为128维/4层/4头；独立模式每份为96维/4层/3头。Actor动作头始终只依赖局部信息。`share_backbone` 与 `critic_privileged` 分别控制权重共享和完整状态分支。

| `critic_fusion` | 行为与约束 |
|---|---|
| late | 局部encoder输出与独立小型特权ViT输出融合；可共享或独立主干 |
| early_shared（当前训练默认） | 局部embedding与特权embedding相加，同一Transformer权重执行两路前向；要求共享dense ViT |
| post_attention | 复用局部主干输出，与特权embedding联合经过默认两层critic专属attention；要求共享dense ViT |
| joint_independent | 恢复1237301的完整状态＋局部历史／记忆联合输入critic；要求独立特权dense ViT patch1 |

`early_shared`无特权时复用一次主干结果。有特权且 `batch_shared_forward=false` 时，PPO依次完成actor前向/反向和critic前向/反向，释放各自激活；共享梯度累积后统一归约、裁剪与Adam更新。开启 `batch_shared_forward` 则沿batch维合并两路主干计算，attention序列仍隔离。其余共享融合方式复用已有激活。详见[结构实验](privileged_fusion_20261106.md)和[顺序反向验证](early_shared_backward_20261109.md)。

`patch_size=3` 使用非重叠patch，23×23补齐到24×24；动作头恢复九个格子的移动logit。包含边缘有效格的partial patch保留，完全无效patch排除。改变patch大小或网络语义需重新训练。

紧凑/时序方案目前支持 `early_shared` 双座当前自对弈。配置校验明确拒绝尚未验证的组合。

## PPO与价值头

默认保留所有样本。`advantage_top_fraction<1` 时，先使用跨卡统计标准化advantage，再在每卡按绝对值选取对应比例，actor和critic使用同一子集。epoch与minibatch数量保持不变，实际训练样本访问次数减少；rollout缓存不缩小。固定物理批之外通过梯度累积完成原逻辑batch。

| 价值头 | 网络输出与目标 | GAE使用的值 |
|---|---|---|
| scalar | 单标量，原MSE／clipped value loss | 原标量 |
| hl_gauss | 默认128个[-1,1]线性中心，sigma=.04，CDF积分后归一化软标签，soft CE | softmax分布的期望 |
| binary_ce | 单logit，软标签 `(return+1)/2`，BCE | `2*sigmoid(logit)-1` |

分类头仅支持纯终局±1奖励、固定折扣、关闭clipped value loss。rollout values、GAE、timeout bootstrap保持标量单位；计算returns后检查有限性与[-1,1]支持范围。HL的边界截断存在向内偏差，监控会记录，不静默校正。`value_loss_coef`需按不同损失尺度单独选择；分箱数量不等于优化开销比例。

`PPO/shared_*grad*`采样主rank首个物理批在归约/裁剪之前的共享主干梯度；`shared_grad_probe_samples`给出该批大小。分类价值诊断也取首个物理批。这些指标用于趋势诊断，不是全卡/全rollout平均。原exact-KL探针使用独立随机流。

## 地图和奖励课程

包内默认配方开启9×9的 `task.map_curriculum`；底层 `TaskConfig` 不隐式启用课程。每个seed生成易图和精确官方终点；中间层级插值出生距离、山脉与城市数量。`map_pool_size`是每层地图数，因此启用课程会增加初始化时间和图池内存。当前对局不换地图，进度仅影响后续reset。

地图生成与缓存由 `map_bank.py` 管理，后台预取在 `map_stream.py`，reset采样在 `map_sampling.py`。1v1出生点筛选、城市公平性、连通性，以及课程连通性修复／出生点配对／路径距离使用C++ CPU内核；Python保留配置、种子、NumPy数组和任务调度。C++继续使用NumPy原BitGenerator，保留随机数调用顺序；固定池的生成结果不因迁移或worker数变化。初始池默认启用内容寻址磁盘缓存（`map_cache_dir="auto"`），按地图配置、种子、画布、课程布局参数、生成源码摘要（包括C++）和NumPy版本区分；仅改变课程推进阈值可复用。目录优先使用 `GENERALS_RL_MAP_CACHE`，否则为XDG cache下的 `generals_rl/map_banks`。`null`禁用缓存。文件锁、原子发布、逐数组校验和防止并发重复生成和读入损坏结果；不同rank仍有不同种子的池。`map_generation_workers=2` 默认用两个spawn CPU进程并行生成，结果顺序与单进程一致；它不是OpenMP线程数。

`map_refresh_reuse=0`保留固定池，保证旧数据分布可复现。设为正整数R后，在后台持续生成新种子的chunk；每累计 `chunk实际大小 × R` 次真实对局抽图，按固定顺序替换池中的一段。首次非空reset只从初始池抽图，不计入替换预算，避免还没开始使用初始池就先替换它；总抽图指标仍包含首次reset。`map_refresh_chunk_size=1024`控制单次替换数量；`map_prefetch_chunks=8`独立控制待消费chunk数，包括运行中与已就绪项，不再由worker数决定。新图只用于后续reset，不替换正在进行的棋盘，也不写入磁盘缓存。生成落后时等待固定的下一块并记录耗时，不能因worker完成顺序不同而改变训练数据。更改池大小、R、chunk大小或首次抽图预算语义会改变数据流；worker数和预取深度只影响速度与内存。原生迁移与八rank实测见[报告](rl_native_mapgen_20261111.md)。

CUDA chunk上传使用生成结果独占的pageable数组与`non_blocking=True`，在当前流内依次上传和写入池，避免每字段一次host同步；后续reset在同一流读取。未增加跨流访问或可提前覆写的共享pinned buffer。

长跑／全尺寸训练可先验证每层8192、R=4、chunk1024、两个worker的滚动池：常驻容量固定，累计新种子持续增长。此配置尚未验证长期学习收益，暂不替换默认固定65536池。初始缓存会随配置／源码版本累积，当前无自动清理；持续生成的图不落盘。resume与既有环境一致，恢复课程计数但重新开始对局、抽图流及滚动池，不支持精确恢复地图流位置。

TensorBoard的 `MapBank/` 记录初始cache hit、加载秒数、累计抽图数、累计引入base seeds和等待新chunk的累计秒数；这些是主rank指标，不是八卡求和。`Curriculum/map_*`保留课程位置和当前局的层级比例。算法、实测和边界见[地图池与同步优化](rl_rollout_and_map_bank_20261111.md)。

历史配置键 `warmup_transitions/end_transitions` 与checkpoint的 `map_curriculum_transitions` 均计数**全局真实board ticks**。双座采样数是该值的两倍。resume恢复保存的绝对计数；warm-start新训练从零开始。`easy_spawn_reference_size=9`可在较大地图上保留易图近距离出生分布。详见[9×9课程](map_curriculum9_20261103.md)、[全尺寸课程](map_curriculum_full_20261103.md)及[压缩存储实验](map_curriculum_full_packed_20261103.md)。

gamma和奖励课程按绝对PPO iteration推进，每次rollout开始前固定设置。`potential_decay`从原始territory/cities/army系数线性衰减，终点由 `potential_end_iteration` 指定，未指定时沿用tick课程终点；胜负奖励保持原值。此时奖励随训练变化，不能声称全程满足固定potential的PBRS策略不变性。

## 实验记录

- [逐版本训练日志、曲线与后期衰退审计](rl_train_log.md)
- [T512结构、gamma和奖励消融](map9_t512_ablation_20261107.md)
- [稀疏价值头与紧凑特征设计](sparse_features_20261109.md)／[结果](sparse_features_20261109_results.md)

上述报告保留实际运行时的1024/2000 tick评估口径；2048是当前新训练和默认评估的上限，不改写历史赛果。报告中的 `outputs/` 路径指向本地原始产物，Git仅保存报告和必要曲线。
