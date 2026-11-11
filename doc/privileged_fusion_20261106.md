# 特权信息融合：两种待训练方案

两种网络已实现为独立选项，保留默认`critic_fusion="late"`及现有默认训练配置。按用户要求，本轮只实现和检查，没有启动两种候选的正式训练，也没有向H200添加新任务。

## 网络与接口

在`runner.policy`中设置以下覆盖；两种新模式均要求`backbone="vit"`、`spatial_mode="dense"`和`share_backbone=true`。

```json
{"critic_fusion": "early_shared"}
```

方案一：局部历史H4、K11记忆及将军坐标经原actor embedding和位置编码得到局部token；完整状态11通道经过同网格patch embedding，逐格加到局部token。Actor和Critic分别运行同一套四层Transformer，最后使用各自的投影和head。共享的是embedding与Transformer参数，主干计算两次；特权状态不进入Actor那次前向。

```json
{"critic_fusion": "post_attention", "critic_attention_layers": 2}
```

方案二：完整复用局部四层Transformer输出，保持空间token；完整状态只作轻量patch embedding，与局部token逐格拼接，经LayerNorm和线性投影残差融合，再经过两层Critic专属Transformer、有效格池化及value head。两层专属block从Actor末两层复制初始化，参数独立，之后分别训练。融合之后有空间attention，不再立即池化。

两种模式保留H4/K11、公开统计、WAIT/MOVE动作头和原actor state dict。共享参数只在`actor.encoder`注册，Adam不会重复更新；value loss可以训练共享主干，value投影/head独立。`critic_privileged=false`仍可关闭完整状态输入；patch1/3、部分边缘patch和padding mask均支持。新增选项只影响训练Critic，Actor导出格式不变。ResNet/pyramid继续使用原`late`模式。

核心实现位于`rl/src/generals_rl/scaled_ppo.py`；`policy/scaled.py`和`policy/tokens.py`只提取复用的输入、embedding及attention操作，未改动vendor。

## 固定条件与参数量

准备配置从上一批`v2_shared_priv_history`的实际`run_config.json`复制，只更改run id和Critic融合选项。保留9×9地图及相同课程、历史recent8均匀池、单座学习、H4/K11、每卡16384环境、T32、E4/M64、逻辑及物理批8192、相同reward/gamma课程。计划仍为八卡、224轮、939,524,096条fresh玩家样本、同一个训练seed。Actor固定为128维／4层／4头，与上一批共享参考相同；没有混入96维或部分共享的变化。

9×9、patch1、head128下，实际参数为：

| 结构 | Actor（含共享主干） | Critic额外参数 | 总计 |
|---|---:|---:|---:|
| 原late共享参考 | 882,313 | 133,761 | 1,016,074 |
| 方案一early_shared | 882,313 | 70,913 | 953,226 |
| 方案二post_attention | 882,313 | 501,633 | 1,383,946 |

这是固定Actor和训练预算的结构比较，不是严格等参数量或等计算量比较。在相同本地运行时和初始化seed下，三种模式的Actor权重逐项相同。参数与初始化检查保存于实验目录`capacity_audit.json`，该构造检查不能代替实际训练初始化审计。

配置与状态位于`outputs/experiments/privileged_fusion_ablation_20261106_actual01/`：

- `configs/v10_early_shared_priv_history.json`
- `configs/v11_post_attention_priv_history.json`
- `plan.json`明确记录`prepared_not_launched`，未创建自动启动队列。

采样评估沿用上一批的9×9固定面板：四baseline各100图、交换座位，共800局，BF16、1024tick；WAIT锚点单列。正式启动前仍需冻结最终源码、验证H200八卡smoke和显存余量，并沿用上一批U112/U224额外评估及八卡分片评估入口。

## 检查范围

专项检查覆盖：共享权重不重复注册；Actor初始权重及输出保持一致；改变特权观测不影响Actor；value梯度能进入共享主干及特权embedding；padding、变长token及patch边缘；CPU单次PPO更新、checkpoint恢复与Actor导出；CUDA BF16打包/掩码输出与梯度对照；patch1/3编译前后对照及反向传播。

BF16编译核融合在接近零的logit处曾出现约0.0033的绝对差异，故编译对照同时检查logit误差（绝对0.005）、value误差（绝对0.003）和实际categorical分布KL（最大绝对值小于1e-4），不把BF16误差当作位级一致。未调整训练精度或PPO算法。

专项31项通过：主检查29项保存在实验目录`validation.xml`，历史池快照和续训两项保存在`validation_history.xml`。旧ViT/ResNet/pyramid、Actor、共享PPO、microbatch、分布式CPU及导出相关回归78项通过。正式训练吞吐、H200显存和棋力尚未测得，不从这些实现检查推断胜率提升。
