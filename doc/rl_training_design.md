# RL训练设计

运行接口和当前默认值见 [RL README](../rl/README.md)。本页说明可复用的实现边界；已完成实验的配置、预算和结果以其报告为准。

## MDP与可变地图

Actor读取自己的局部地图、历史、记忆与公开比分，critic可额外读取当前完整棋盘。真实地图填充到训练画布；独立布尔 `valid_cells` 控制合法动作、统计归一化、attention与池化。无效格不作为可学习的地图通道，也不堆叠到历史。训练和导出共用构造器，将动作转换回真实地图宽度。

动作保持环境原生编码：WAIT=0，其余为来源格×四方向×全量/半量。mask排除非己方来源、兵力不足、越界、山脉和填充目标。`temporal_dedup`只屏蔽来源恰有2兵时与全量等价的半量动作。

| 特征方案 | 历史／记忆 | 神经网络空间输入 | 附加公开统计 |
|---|---|---|---|
| stacked | H4/K11 | 4帧局部13通道＋K11＋将军相对坐标，按4通道对齐为68 | 当前比分／时钟 |
| compact | H4/K11 | 当前局部类别、相邻帧己/敌兵力差、精简记忆，共28通道 | 当前比分／时钟 |
| temporal | H8/K12 | 延长兵力差历史、最近见到的敌军兵力，共40通道 | 敌军兵力/领土相对16/64/256 tick EMA的差值 |
| temporal_dedup | 同temporal | 同temporal | 同temporal，另启用等价动作去重 |

记忆按座位和episode隔离，只用可见地块与公开比分更新；旧敌军兵力不代表当前隐藏状态。新观测可更新地形、建筑和将军身份。局部combat风险与margin仍为连续特征，不按二值存储。

特征配方由 `policy/features.py` 统一定义H/K、精度和动态统计维数。配置解析负责派生task、policy与内置codec；切换方案时全部重建。内置配方固定历史长度；显式自定义codec及关闭压缩/去重的选项仍由调用方负责。

## Rollout与存储

双座当前策略自对弈中，每局每tick生成两条玩家transition，GAE与奖励按座位分开。一次rollout内参数固定；对局可跨越多轮PPO。自然终局与外部时限截断分别记录，timeout使用重置前终局观测的critic值bootstrap。

历史去重每tick存一帧，加上rollout起点上下文和episode边界；抽样时重建原堆叠。标准codec按位压缩二值信息，保留FP32连续值。紧凑方案在行动前就量化至FP16可精确表示的值，再以FP32参与推理；存储解码恢复相同输入，避免old log-prob与更新输入因压缩不一致。

存储实现与MDP表示分别配置。旧标准存储验证见[历史去重](history_storage_20261103.md)和[位打包](packed_storage_20261103.md)。

## Actor与critic

共享ViT1M主干为128维/4层/4头；独立模式每份为96维/4层/3头。Actor动作头始终只依赖局部信息。`share_backbone` 与 `critic_privileged` 分别控制权重共享和完整状态分支。

| `critic_fusion` | 行为与约束 |
|---|---|
| late（默认） | 局部encoder输出与独立小型特权ViT输出融合；可共享或独立主干 |
| early_shared | 局部embedding与特权embedding相加，同一Transformer权重执行两路前向；要求共享dense ViT |
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

`task.map_curriculum`默认关闭。每个seed生成易图和精确官方终点；中间层级插值出生距离、山脉与城市数量。`map_pool_size`是每层地图数，因此启用课程会增加初始化时间和图池内存。当前对局不换地图，进度仅影响后续reset。

历史配置键 `warmup_transitions/end_transitions` 与checkpoint的 `map_curriculum_transitions` 均计数**全局真实board ticks**。双座采样数是该值的两倍。resume恢复保存的绝对计数；warm-start新训练从零开始。`easy_spawn_reference_size=9`可在较大地图上保留易图近距离出生分布。详见[9×9课程](map_curriculum9_20261103.md)、[全尺寸课程](map_curriculum_full_20261103.md)及[压缩存储实验](map_curriculum_full_packed_20261103.md)。

gamma和奖励课程按绝对PPO iteration推进，每次rollout开始前固定设置。`potential_decay`从原始territory/cities/army系数线性衰减，终点由 `potential_end_iteration` 指定，未指定时沿用tick课程终点；胜负奖励保持原值。此时奖励随训练变化，不能声称全程满足固定potential的PBRS策略不变性。

## 实验记录

- [逐版本训练日志、曲线与后期衰退审计](rl_train_log.md)
- [T512结构、gamma和奖励消融](map9_t512_ablation_20261107.md)
- [稀疏价值头与紧凑特征设计](sparse_features_20261109.md)／[结果](sparse_features_20261109_results.md)

上述报告保留实际运行时的1024/2000 tick评估口径；2048是当前新训练和默认评估的上限，不改写历史赛果。报告中的 `outputs/` 路径指向本地原始产物，Git仅保存报告和必要曲线。
