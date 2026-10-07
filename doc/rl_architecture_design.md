# Generals RL：成熟 ResNet / ViT 的逐地块适配设计

日期：2026-10-07。本文对应用户要求的 1M / 10M / 100M 三档、可变地图尺寸，以及相同参数量下的计算效率和样本效率比较。训练阶段顺序仍是：v19重跑v17对照 → 事实记忆与 critic 信息输入消融 → 本文网络比较 → gamma / rollout / lambda → tick penalty 衰减 → H4/H8。已执行上游检查、参数审计、CPU合同和1M孤立GPU校准；随后两种1M真实满存储/256Adam/24模型驻留/1600局CUDA采样容量门禁均已actual exit0，详见训练日志。v23 ResNet1M与v24 ViT1M从零正式训练、五点完整曲线、开发评测与最佳模型录像均已完成；真实终局pair已执行，ViT计算更快而当前有限预算的强度未整体胜出。两轮结束后的实际profiler正在执行；10M/100M GPU门禁与正式结果尚待完成。随机同权重pool压力测量不代表成熟已训练策略分组吞吐。

## 1. 选择与来源

使用 **timm 1.0.30** 的成熟上游实现，Generals 只提供地块保持适配器、输入布局和 policy/value head。不上 ImageNet 预训练权重、不复用视觉增强或分类训练 recipe，不自行重写 residual block、attention、LayerScale 或位置插值。

| 项目 | 选择 | 理由与范围 |
|---|---|---|
| ResNet | 完整 timm `ResNetV2`，标准 ResNet-50 的 [3,4,6,3] 四阶段，GroupNorm + Weight Standardization | 官方成熟模型构造、deep stem、stage、projection、block 与初始化；无跨样本 running statistics，适合 PPO rollout / update / opponent inference 的一致性 |
| ViT | timm `VisionTransformer`，DeiT-III 风格 Pre-LN + LayerScale + GELU MLP + fused SDPA | 复用成熟 transformer block；一格一 token；上游动态尺寸位置插值；无需 xFormers |
| 任务适配 | stride=1 的 ResNet stem/tower；ViT patch_size=1；移除分类池化和分类头 | 9×9 棋盘不能经 ImageNet 默认下采样变成几格，必须保留动作 source 与地块一一对应 |

已经调查 ResNet-RS / ResNet strikes back、DeiT-III、DINOv2 / DINOv3、NaFlexViT。ResNet-RS 的成熟改良与缩放研究是很好的参考，但其分类模型默认使用 BatchNorm、大步长与额外正则；直接照搬并不适合当前 PPO。BiT/ResNetV2 是这里有意选择的成熟 ResNet 分支，**不宣称它是 2026 年图像分类榜单第一名**。DeiT-III 与 timm 持续维护的 ViT 模块提供现代稳定组件；DINO 的自监督预训练目标/teacher/augmentation 不是本任务的训练目标，也不引入本实现。NaFlex 的 mixed-resolution packing 是未来需要同一 batch 混合地图尺寸时的参考，首版用同尺寸分桶，避免堆叠两套输入协议。

上游原始资料：

- [timm v1.0.30 release](https://github.com/huggingface/pytorch-image-models/releases/tag/v1.0.30)，2026-09-22，固定 commit `0df212b369a5385b16dfe513d5143a7311ea1ddc`。
- [固定版本 ResNetV2 源码](https://github.com/huggingface/pytorch-image-models/blob/0df212b369a5385b16dfe513d5143a7311ea1ddc/timm/models/resnetv2.py)、[标准化卷积](https://github.com/huggingface/pytorch-image-models/blob/0df212b369a5385b16dfe513d5143a7311ea1ddc/timm/layers/std_conv.py)。直接实例化完整 `ResNetV2`，由上游构造 `ResNetStage`、`PreActBottleneck`、shortcut、`StdConv2d`、`GroupNormAct` 与模型初始化，不自行排列一个同宽 block tower 冒充成熟 ResNet 架构。
- [Google BiT 官方 PyTorch 实现](https://github.com/google-research/big_transfer/blob/master/bit_pytorch/models.py)、[BiT 论文](https://arxiv.org/abs/1912.11370)：GN/WS 的来源。
- [ResNet-RS 论文](https://arxiv.org/abs/2103.07579)、[ResNet strikes back](https://arxiv.org/abs/2110.00476)：现代 ResNet 训练与缩放的参考；其监督训练结论不作为 Generals 胜率结论。
- [固定版本 ViT 源码](https://github.com/huggingface/pytorch-image-models/blob/0df212b369a5385b16dfe513d5143a7311ea1ddc/timm/models/vision_transformer.py)、[DeiT-III 官方实现](https://github.com/facebookresearch/deit/blob/main/models_v2.py)、[DeiT-III 论文](https://arxiv.org/abs/2204.07118)。官方 DeiT-III `init_scale=1e-4` 是本文 LayerScale 的默认依据。
- [固定版本 NaFlexViT 源码](https://github.com/huggingface/pytorch-image-models/blob/0df212b369a5385b16dfe513d5143a7311ea1ddc/timm/models/naflexvit.py)、[NaViT 论文](https://arxiv.org/abs/2307.06304)：可变尺寸位置编码、padding 与 packing 的扩展参考。

依赖与许可：工作区当前 `torch=2.13.0+cu130`、`torchvision=0.28.0+cu130`；已将 `timm==1.0.30` 安装在工作区 `.venv`，没有替换 torch。下载的 wheel SHA256 为 `c8e27adf6801a2fdfb570dbbdd171f2f084756eefc673df8cfaab06bd14864f3`。采用包依赖而不是再 vendor 一份视觉仓库；[timm 许可为 Apache-2.0](https://github.com/huggingface/pytorch-image-models/blob/0df212b369a5385b16dfe513d5143a7311ea1ddc/LICENSE)，并保留依赖包中的原始版权/许可。这里只使用 timm 的代码，不下载任何单独许可的预训练 checkpoint。

## 2. 输入、地图和动作的共同合约

观测模块负责区分当前局部观测、短期 history、长期 knowledge、特权 state 和 public statistics。网络只消费定义好的 tensor schema，不在 trunk 内重算视野、读取 simulator 私有状态、或维护跨局记忆。

H4 / P2 时，actor 的示例空间输入为 `48 history planes + 11 knowledge planes + 2 relative geometry planes = 61 channels`；11 个知识通道只附加一次。通道数由 schema 传入，不硬编码在成熟 backbone 中。critic 的 state / history / knowledge 通道配置单独传入，actor 不读取 critic 输入。

`build_backbone(kind, size, in_channels)` 返回具备 `out_channels` 属性的模块。输入 `[N,C,H,W]`，输出 `[N,D,H,W]`，H、W 在 forward 时决定。矩形地图同样支持。actor 的公共特征融合、critic value head、动作概率和合法性由上层实现。

共同动作协议：`WAIT=0`，移动动作 `1 + source*8 + direction*2 + mode`；source 为 row-major 索引，动作总数 `1+8HW`。保持 v17 的 WAIT/MOVE factorization。逐格 head 输出 `[N,8,H,W]`，必须按 `[H,W,8]` 顺序 flatten，不能按通道顺序打散。合法动作 mask 由 MDP 提供，网络不另设战术过滤或对 baseline 的专门规则。

地图尺寸、行列坐标、相对己方大本营坐标、邻接关系、public 归一化和 map mask 应通过共同 geometry/layout 模块获得。改变地图大小不重新初始化模型权重，不把棋盘 resize 成固定 9×9。参数个数与 H/W 无关；ViT 位置表的学习参数固定在参考 9×9 网格，上游负责 runtime 插值。

首版每个run固定一个实际H/W；以该尺寸构造Task、CTS、storage和历史模型工厂，网络参数尺寸仍不依赖H/W。已经有7×11真实PPO/self-play/采样eval的CPU合同。单run内按bucket切换Task/storage/pool尚未实现，不能仅调用policy.set_geometry改变正在运行的流水线。GroupNorm的统计不依赖别的样本，但会包含空间位置；直接padding会改变空间统计。ViT 仅屏蔽 attention 也不够：每个样本的位置编码、pooling、action mask 仍须符合自己的真实尺寸。因此混合尺寸 padding/packing 需要单独实现完整语义，不能宣称现有动态尺寸能力已经完成该功能。

## 3. ResNet 详细结构

直接构造 `ResNetV2(layers=[3,4,6,3], channels=scaled_channels, stem_type='deep', avg_down=False, output_stride=32, num_classes=0, global_pool='')`。保留官方三个 3×3 卷积的 deep stem、标准四 stage、16 个瓶颈、每 stage 首块的 projection shortcut 与 final normalization。三个档位只统一缩放通道：以 canonical (256,512,1024,2048) 和 stem64 分别按约 0.196、0.64、2.02 缩放，stage 通道 round 到16、stem到8且至少16；选择倍率只匹配参数预算，不通过训练胜率搜索。

适配仅为：将 stem max-pool 替换为 Identity，将所有官方卷积（包括 shortcut）的 stride 改为1，使用 `forward_features` 跳过分类 head。先用 output_stride32 构造是为了保留 canonical dilation=1，不能直接传 output_stride1 使上游把步长换成扩张卷积。同步将 `feature_info` 的 reduction 改为1，避免未来特征接口仍错误报告32倍下采样。保留所有 padding、shortcut 连接与算子顺序。

GroupNorm 统一用4 groups，使三档的 narrow deep stem 与官方舍入后的瓶颈中间宽度均可整除；不为各档搜索不同归一化超参数。完整上游模型负责 Kaiming/GN 初始化，`zero_init_last=False` 与其 WS 构造默认一致。关闭 stochastic depth。

| 档位 | Stage widths | Stem width | Stage depths | 输出D | 实际 backbone 参数量，C=61 |
|---|---|---:|---|---:|---:|
| 1M | 48 / 96 / 208 / 400 | 16 | 3 / 4 / 6 / 3 | 400 | 997,384 |
| 10M | 160 / 320 / 656 / 1312 | 40 | 3 / 4 / 6 / 3 | 1312 | 9,740,084 |
| 100M | 512 / 1040 / 2064 / 4144 | 128 | 3 / 4 / 6 / 3 | 4144 | 96,735,392 |

这是一项明确的 full-model dense-map 适配，不能将它称为未经改动的 ImageNet ResNet-50，也不能照搬其分类精度。16个瓶颈和 deep stem 已提供足够多的局部传播层；扩容主要比较表达与优化能力，不只增加感受野。

## 4. ViT 详细结构

使用上游 `VisionTransformer`：`patch_size=1` 的投影 → learned 2D positional embedding → L 个 Pre-LN attention/MLP blocks → final LayerNorm。`class_token=False`、`num_classes=0`、`global_pool=''`，直接返回地块 token，适配器 reshape 为 `[N,D,H,W]`。

每个 block 使用 QKV bias、MLP ratio=4、GELU、LayerNorm eps=1e-6、两路 LayerScale 初值 1e-4；关闭 dropout、position dropout、patch dropout、attention dropout 与 drop-path。Attention 直接采用 timm 的 fused PyTorch SDPA 路径；不要新增手写 attention 或额外 xFormers 依赖。训练与推理保留同一确定性网络变换，策略采样来自 categorical action distribution。

| 档位 | D | Transformer blocks L | Heads | Head dim | 实际 backbone 参数量，C=61 |
|---|---:|---:|---:|---:|---:|
| 1M | 128 | 5 | 4 | 32 | 1,011,200 |
| 10M | 320 | 8 | 5 | 64 | 9,915,200 |
| 100M | 896 | 10 | 14 | 64 | 96,602,240 |

`dynamic_img_size=True` 复用 timm 的二维位置插值；只插值位置参数，不插值兵力/障碍/动作格子。无需设置地图尺寸上限。能力边界：新尺寸能 forward，并不证明没训练过的尺寸能正确理解相同距离或战术。先验证 9×9、7×11、13×13 的结构合约，再用后续 mixed-size training 与 held-out 尺寸验证行为泛化。若插值导致明显迁移失败，再单独比较固定二维坐标/上游 RoPE/NaFlex，而不是同时堆入多个位置技巧。

## 5. 参数与计算的公平比较

参数预算以 **actor 独立参数** 为准，不把 critic 或历史池复制的权重算入一个 actor。critic 在同一比较阶段使用相同独立结构；若 later 证明 critic 是容量瓶颈，再开单独实验。不能同时替换 actor/critic 并把全部增益归给 actor。

当前“相同 critic”指结构与参数量相同。构造顺序是先actor再critic，仅使用整体训练seed；不同actor消耗不同RNG，因此未经逐张量核验不能声称配对critic初始权重相同。这是单seed比较的归因限制，现有容量门禁不证明训练强度差异。

共同 head 先 `Conv1×1(D→128)` 投影，再使用 `Linear(128+13→128)` 的 public context、`Conv1×1(128→8)` 的逐格 head、`Linear(256→128)→Linear(128→1)` 的 WAIT/MOVE gate。除输入投影外 head 参数为52,233，输入投影为128D+128。这样不会因为 ResNet 输出通道更宽，给它附加不同规模的深 action head。最终 run manifest 仍须从真实模型重新计算参数，不以文档估算代替实际数据。

三档 backbone 参数差约1.39%、1.80%、0.14%；加入共同 head 后 actor 配对差约1.94%、0.48%、0.57%，均小于3%。

| 档位 | ResNet actor：共同 head | ViT actor：共同 head | 9×9 ResNet forward GMAC | 9×9 ViT forward GMAC |
|---|---:|---:|---:|---:|
| 1M | 1,100,945 | 1,079,945 | 0.0843 | 0.0901 |
| 10M | 9,960,381 | 10,008,521 | 0.8003 | 0.8349 |
| 100M | 97,318,185 | 96,769,289 | 7.8712 | 7.9348 |

MAC 表是按 Conv/Linear 及 attention QK/AV 乘加推导的估算，C=61、81 tokens，不包含 GN/LN、WS、激活、softmax、内存搬运，也不是硬件实测。以“一次乘加为一个 MAC”计数；若按一次乘加两 FLOPs 报告，应乘 2，并写清约定。同参数不等于同 latency，尤其 WS、GN、attention kernel 和小网格会影响实际速度。

正式比较同时给出：

1. **计算效率**：固定输入尺寸与模式，记录 actor forward 的 p50/p95 latency、samples/s；再测实际 self-play 收集和 PPO update，以环境 transitions/s、iteration collect/update 秒数和 GPU peak memory 为准。包含对手推理，不只测 learner。
2. **样本效率**：随机初始化、同 MDP、同 PPO 和同 pool 协议，画采样 eval win-rate 对累计 transitions；共同样本区间内比較。各档都有 1–3 小时训练预算，慢模型可能未达到快模型的样本数，必须报告这一差异。
3. **预算效率**：同一批训练曲线另以 wall-clock 小时为横轴，报告 1h / 2h / 3h 内实际最好表现；不为了同样的 steps 把某个慢模型暗中训练更多小时。
4. **评价协议**：同一 sampled eval、相同地图与双座位、独立 held-out map/sampling seeds；报告每个 baseline 和均值。Baseline 只用于测量，不混入 self-play。选出的最佳 checkpoint 还应与 v17 以及同阶段其他策略做有限 cross-play，避免只以 baseline 排名推导世界冠军级能力。
5. **选择规则**：选择吞吐与胜率 Pareto frontier 上适合本机的配置，不只看参数更大；若 100M 在三小时内学习很少，记录低采样量及更新次数，不能据此宣布 100M 的长期能力较差。

归因边界：六个容量条件使用同一约1M ResNet critic，能比较配对actor架构与容量；轻量v19–v22的critic仅约5–14k，因此tiny→MapPPO还同时改变critic容量与计算设置，不能将系统差异全部归因actor变大。若需要这一因果结论，须另加tiny actor＋同一1M critic对照，而不是用六档现有比较代替该对照。可变尺寸的前向合同也只证明能运行，不证明已完成多尺寸训练或尺寸外泛化。

先进行短 warm-up 的真实计算基准，再决定 N/T 与 physical microbatch；不凭参数量直接承诺 GPU 能容纳或能达到某个吞吐。三档都保留配置并实际测量，100M 若需要减环境数量，使用与它配对的 ResNet/ViT 同样 N/T，另用轻量对照量化 rollout 长度变化的影响。

## 6. 单机显存、长 rollout 与 100M

当前逻辑 rollout 样本数为 `16384×32=524288`。可保持样本数用 `8192×64`、`4096×128`、`2048×256`、`1024×512`。降低 N 能减少实时环境、history/opponent 输入与整批推理临时激活；**如果 N×T 不变，展开 rollout 观测总量基本不变**。长 rollout 同时改变 GAE 的连续上下文，因此不能将这项变化视为完全不影响训练算法的显存技巧。

H8留在最后单独消融。按当前9×9/P2单帧985个FP32值、N16384/T32，actor历史rollout由H4的7.6953GiB增至H8的15.3906GiB，增7.6953GiB；learner加两个seat的原始history frames再增0.7214GiB，task返回的独立flatten clone再增0.2405GiB。已明确的缓存增量至少约8.6572GiB，还未计push/cat、索引物化、激活和瞬时副本，不能将其当作完整峰值预测。BF16 forward不改变现有FP32 rollout dtype；降低N而保持NT也不会减少这部分rollout总量。执行H8前需对实际存储方案做显存门禁，不能只修改history配置后直接启动。

100M float32 actor 的权重约 0.37 GiB；Adam 参数、梯度、两个状态合计约 1.49 GiB，仅为 actor。历史池若有八个全 GPU actor，权重另约 2.98 GiB；当前实现实际复制整policy，需另计每份critic。池大小8约束可选快照与磁盘，已淘汰但仍有未结束episode引用的模型继续驻留GPU，resident数量可以超过8；发布新模型也有临时重叠。保持整局固定对手，因此不能通过中途换模型释放这些引用。容量门禁必须包含活动尾部、真实rollout、learner/Adam、CUDA workspace、额外eval模型，不能按“权重不到0.4GiB”推断能用原batch8192。

100M ResNet 输出D=4144 时，physical batch=64、9×9 的一个末阶段 float32 feature tensor 约82 MiB；autograd 会保存多阶段、多分支数据。ViT 的 MLP 中间宽度为 4D，也有显著激活成本。措施按复杂度由低到高：

- 给 learner/opponent forward 分块；eval 不保留梯度。
- 使用较小的 optimizer physical microbatch，按 microbatch 样本数加权累积一次 logical minibatch 的梯度；optimizer step、clip-gradient 和 adaptive LR 仍按逻辑 minibatch/现有 PPO 时序执行。
- Advantage normalization 保持同样范围，不能每个小 chunk 单独归一化；固定 monitor probe 的定义也不随分块改变。
- 在验证 categorical log-prob、value 和 KL 计算稳定后测试 bf16 autocast；概率、GAE、return 与损失归约保持 float32。它不是当前 CPU 合约测试已经验证的能力。
- 只有实测仍不足时再采用上游 activation checkpointing；不要默认增加状态 offload 或复杂并发系统。

100M 约 8 GMAC / actor forward，训练计算确实显著增大。减少 N 并保持 N×T 不变并不减少每轮全部样本所需的网络乘加。它的 1–3 小时报告必须同时给出 interactions、optimizer steps、快照数量与时间预算，不能只画 iteration step。

模型与存储保持固定 retention：历史对手 checkpoint 应只保存 inference 权重，正式训练 resume checkpoint 才保留 optimizer 状态；模型池与 best/recent checkpoints 不无限增长。大模型 eval 的 CPU 成本也需实测，并限制待执行 eval 队列，避免 evaluation backlog 使结果曲线落后数小时。

## 7. 已完成的直接验证与待验证项

当前实现位置：`rl/src/generals_rl/backbones.py`。`build_backbone` 的三个尺寸常量和实际 timm 类在代码中可直接检查。未修改 rsl_rl vendor 的 residual/attention 实现；成熟网络是包导入。

已运行 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=rl/src:rl/tests .venv/bin/python -m unittest test_backbones -v`，5 项通过：

- 直接调用完整 timm `ResNetV2` / `VisionTransformer`，而非同名自制 block；验证 canonical ResNet50 stage深度3/4/6/3、deep stem、全stride1、全dilation1和feature_info reduction1；ViT patch=1、无 CLS token、LayerScale 初值来自官方。
- 同一 1M 模型处理 9×9、7×11、13×13，并保留每格输出，结果有限。
- 相同输入在 train/eval 下结果完全一致；保持 batch shape 不变、修改别的样本数值时，该输入输出逐bit一致，无 BatchNorm。单样本与多样本 kernel 的结果在float32舍入误差内一致，不要求跨kernel逐bit一致。
- 输入和上游 residual / attention 权重的梯度有限且非零。
- 在 meta device 审计六个模型真实参数量，不分配 100M 实际权重；检查配置和输入错误。

可选更新适配位于 `rl/src/generals_rl/student_ppo.py`。`physical_minibatch_size=None`、`amp_bfloat16=False` 时仍直接调用 vendor `PPO.update()`，旧 Light policy 无需增加配置。开启分块时每个 physical chunk 按实际样本数 / logical batch 样本数加权；全局或每 logical minibatch 的 advantage normalization 范围保持原样。完整 logical batch 仅执行一次 adaptive categorical k3 聚合、学习率调整、全参数梯度裁剪、Adam step 和 monitor；k3 仍是 `mean(clamp_min(expm1(log_ratio)-log_ratio, 0))`，学习率倍数1.5、上下限1e-5 / 1e-2与现有 vendor 一致。每 logical batch 保留一次完整形状的 multinomial 调用，避免 physical chunk 改变采样随机数消耗。

BF16 是独立 opt-in：Map policy 的 rollout/inference 与 StudentPPO update 必须使用相同 `amp_bfloat16` 值。算法仅当 policy 显式提供该字段时检查一致性；双向失配在初始化时报错，旧 Light 默认路径不受影响。Map policy 在网络 forward 后将 raw logits / value 转 float32，Categorical、log-prob、entropy、PPO loss 和 KL 在 float32 计算；master weights、gradients、Adam states 保持 float32，bf16 不使用 fp16 GradScaler。只支持 feedforward categorical、teacher_env_ratio=0、单设备，不支持非零 stochastic dropout、BatchNorm、RND、symmetry 或 recurrent update。

已执行以下 CPU 命令，15 项通过（8 microbatch、5 monitor、2 legacy StudentPPO），测试过程隐藏 CUDA，没有启动正式训练：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=rl/src:env/src:rl/vendor/rsl_rl:rl/tests .venv/bin/python -m unittest \
  test_student_microbatch test_ppo_monitor \
  test_student_ppo.StudentPPOTests.test_update_matches_original_ppo_complete_parameters_adam_and_rng \
  test_student_ppo.StudentPPOTests.test_local_student_gets_task_gradients_in_one_optimizer_and_critic_latent_stays_detached -v
```

证据包括 logical17 / physical6 的6/6/5不等长 chunk、physical1与大于 logical batch 的 chunk；loss、完整 model / Adam 与 monitor 在 float32 舍入容差内一致，RNG 完全一致。直接计数确认 clipping / Adam / monitoring 各为 epochs×logical minibatches 次；max_grad_norm=.01时累积后的梯度范数达标；异质 k3 与学习率上下限匹配 vendor；禁用分块和匹配false精度字段时 model / Adam / loss / RNG 逐bit一致。分块运算在不同矩阵 kernel 与归约顺序下不保证 float32 逐bit一致。

这些CPU验证不等同于10M/100M GPU训练。后续GPU的实际证据与边界如下；正式从零训练的胜率仍须逐版写入 `doc/rl_train_log.md`。

## 8. 初次GPU校准与下一步门禁

诊断脚本和原始JSON在 `outputs/moe_cts/iterations/architecture_preparation/`。单4090空闲时，每个worker独立进程、随机模型、合成9×9观测，真实MapPPO/StudentPPO，logical8192/physical64，一次计时；没有完整环境、NT存储或历史池。数学检查另外比较17样本一次logical更新与6/6/5累积，保持原验收阈值，核对RNG、LR及clip/Adam/monitor各一次。

| 1M actor | 精度 | 8192样本一次Adam秒数 | 孤立peak allocated GiB | 梯度cosine（17 vs 6/6/5） |
| --- | --- | ---: | ---: | ---: |
| ResNet | BF16 forward / FP32 master | 2.5127 | 0.9246 | 0.999882 |
| ViT | BF16 forward / FP32 master | 1.7840 | 0.8202 | 0.999860 |
| ResNet | 严格FP32，matmul highest且两类TF32关闭 | 1.9479 | 0.8285 | 0.999999999999459 |
| ViT | 严格FP32，matmul highest且两类TF32关闭 | 1.5419 | 0.8772 | 0.999999999999515 |

初始high精度下两个FP32 worker失败；ResNet的具体断言为gradient cosine0.99638低于既定0.999，RNG/LR/更新次数通过。显式关闭matmul及cuDNN TF32后两种架构都通过原阈值，loss差异最大2.39e−7。证据说明失败与CUDA精度设置相关，尚未隔离matmul、卷积TF32和具体kernel各自的作用；不放宽阈值、不把最初失败抹去。BF16网络层实测为BF16，categorical logits/value与master/gradient为FP32。上述单repeat小分块不能推出一般速度排名，当前BF16也未比严格FP32更快。

physical64的投影显示默认每100更新评测无法在1–3h内产生足够曲线。下一空闲GPU窗口先比较physical512的两种精度，再用共同精度测1024/2048；不立即增加compile、删除上游层或减少NT。正式配置需经过真实环境、成熟池与监控/eval的完整显存门禁。按实测iteration秒数选择两种架构共同的发布/评测间隔，并记录各模型真实更新数量；projection和微基准都不是正式训练吞吐。

为复现严格FP32，训练配置增加可选顶层 `cudnn_allow_tf32=false`，与 `matmul_precision="highest"` 配套；默认null保持现有cuDNN设置，训练退出或异常均恢复调用前设置。CUDA评估复用并记录实际cuDNN精度；若报告与显式配置不一致，父进程拒绝接受该评测。已有冻结v20不受这项候选配置控制改动影响。CPU控制/eval/trace/wall-cap检查37 tests、41 subtests通过，新增报告拒绝合同随后单独执行12 tests、16 subtests通过。此配置入口尚未做正式GPU训练验证，不把上述孤立GPU微基准冒充对新入口的验证。

候选整体CPU检查一次执行80 tests、117 subtests全部通过（28.47秒），日志 `architecture_preparation/candidate_integration_cpu_tests.log`；其中含网络/动态几何、微批次、原PPO监控及准备中的评测/课程合同。另用随机ResNet1M、0 PPO更新完成32局实际GPU BF16采样评估，四条真实loss轨迹共528tick在CPU/native/Core按记录动作精确重放并生成四段MP4；缺少的四类win明确标注。证据 `evaluation_preparation/cuda_smoke_random_1m/cuda_gate.json` 及 `analysis/demo_manifest.json`。该门禁父进程仅微小占用，不能证明训练满池下eval峰值或100M容量。

容量审查补充：`PeriodicEvaluator` 的 CUDA 报告验收现同时检查显式配置的 matmul precision，以及 policy 的 AMP 推理 dtype（默认 float32，开启 AMP 则 bfloat16）；字段缺失或不符时拒绝评测。未配置 matmul precision 时不强制该项，CPU 路径保持原样。新增四项 stub 合同连同原精度元数据/worker 路由检查共七项 targeted CPU tests 通过，实际 session81866、exit0；没有运行新游戏或 GPU 测试。仅修改评测验收及既有测试，不改变 PPO/vendor，也不改变已冻结的 v21/v22。
