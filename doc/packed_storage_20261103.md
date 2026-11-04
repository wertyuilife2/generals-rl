# Rollout二值特征无损压缩实验

日期：2026-11-03。基于[H4历史去重](history_storage_20261103.md)，将rollout中的二值特征按位保存，
取minibatch时恢复原FP32观察。原完整地图课程实验仍在U14停止，没有恢复长训练。

单卡实测基础allocated从53.030 GiB降至18.694 GiB，再节省34.336 GiB。
CPU和确定性CUDA/BF16的真实PPO对照通过，观察、动作、GAE、loss、模型、Adam和RNG完全一致。
八卡对照已完成：相同2048物理批的吞吐差异为−0.56%，4096通过，8192在实际PPO中OOM。

## 存储格式与代码边界

[编解码器](../rl/src/generals_rl/rollout_codec.py)属于RL训练模块，支持当前P2、14个地图通道、13个公开标量的布局：

- 地图通道0–8、12、13是二值，11个标志打包到一个int16；
  通道9–11的log兵力、攻击风险和战斗差值保留FP32；公开标量也保留FP32。
- K11的前10通道打包到int16，最后的age保留FP32。本实验未进一步量化age。
- action mask从逐元素bool改为每8位一个uint8，尾部不足8位补零并在解码时裁掉。
- 其他观察键保持原格式。CUDA使用torch.compile融合gather与解码，CPU使用相同张量表达式；
  无新增Python依赖或CUDA手写算子。

编码值仅用于rollout物理缓存。实时task观察、特征构造、模型、BF16设置、env和Arena协议均沿用原实现，
导出的策略包也不依赖该训练模块。Vendor仅接收通用encode/gather工厂，保留历史前缀、episode边界和标量存储；
特征通道知识全部留在RL侧。PPO及KL probe统一通过observation_batch读取完整观察。

以下是实验验证的配置，两个存储选项独立；实验后已按用户要求作为包内默认，
并启动[新的全尺寸224轮训练](map_curriculum_full_packed_20261103.md)。物理批为本轮通过的4096：

```json
{
  "runner": {
    "storage_history": {"policy": 4},
    "storage_codec": "generals_rl.rollout_codec.GeneralsObservationCodec",
    "algorithm": {"physical_minibatch_size": 4096}
  }
}
```

可用于现有checkpoint的resume；编解码器不参与policy源码身份或权重格式，
仅改变训练缓存表示。正式使用时仍须选择适合地图分布和显存的物理批。

## 正确性

- 38项CPU测试、57个subtest通过，覆盖真实PPO、异步及连续reset、跨rollout、末尾短rollout、
  部分键的KL probe、teacher/student索引、真实benchmark/profile和分布式基准回归。
- 所有2048种11-bit组合均通过往返测试，连续值包括负零，按FP32位模式比较；
  随机mask包含不足一个字节的尾部，K11 age包含0–2000范围的不同值。
- CPU真实PPO对照同时测试仅位打包，以及位打包＋历史去重，loss、参数、Adam和RNG逐项相同。
- H200：相同U14模型权重，64个真实随机尺寸环境，连续两轮T8及PPO，episode上限3，
  强制发生reset和换座。在确定性CUDA/BF16下全部字段、loss和更新状态精确一致，最大差异0。
- 地图课程真实训练及checkpoint恢复也使用压缩存储通过测试。

## 容量与单卡显存

每卡N=16,384、T32、H4、23×23画布。以下为按形状和dtype计算的主要缓存容量，包含H−1历史前缀：

| 缓存 | 历史去重后 | 再加位打包 |
|---|---:|---:|
| Policy历史 | 15.85 GiB | 3.98 GiB |
| Critic观察 | 14.49 GiB | 3.64 GiB |
| K11 | 11.37 GiB | 1.55 GiB |
| Action mask | 2.07 GiB | 0.26 GiB |
| 合计 | 43.77 GiB | 9.43 GiB |

实测含实时状态与观察、完整rollout及模型的基础allocated：53.030→18.694 GiB。
单卡探测关闭模型编译，保留BF16；每层bank为16张。Packed CUDA编解码本身始终编译，并在计时前暖身。

| 地图输入 | 存储 | 物理批 | 实测reserved峰值 | 加完整bank后预计reserved |
|---|---|---:|---:|---:|
| 全23×23 | 历史去重 | 2048 | 91.79 GiB | 93.94 GiB |
| 全23×23 | 位打包＋去重 | 2048 | 57.49 GiB | 59.64 GiB |
| 全23×23 | 位打包＋去重 | 4096 | 94.21 GiB | 96.36 GiB |
| 全23×23 | 位打包＋去重 | 8192 | OOM | — |
| 官方随机尺寸 | 位打包＋去重 | 2048 | 48.52 GiB | 50.67 GiB |
| 官方随机尺寸 | 位打包＋去重 | 4096 | 77.20 GiB | 79.35 GiB |
| 官方随机尺寸 | 位打包＋去重 | 8192 | 133.66 GiB | 135.81 GiB |

完整65,536张／层bank比16张探测增加约2.151 GiB，最后一列为容量估算；
该探测仅执行一次前向/反向，不包含正式PPO的KL probe等所有临时量。
8192对地图形状分布敏感，不能从随机尺寸探测成功推断固定全23×23训练也能运行。

## 存储profile

同一完整N/T，计时前完成所有编解码暖身；以下为3次CUDA event测量的均值：

| 操作 | 仅历史去重 | 再加位打包 |
|---|---:|---:|
| 写入完整T32 rollout | 32.23 ms | 43.37 ms |
| 8192逻辑批采样4 epochs | 582.14 ms | 641.47 ms |
| 合计 | 614.37 ms | 684.84 ms |

位打包增加bit操作及编解码，单独存储工作约增加70 ms／轮。
torch.profiler可见融合的cat重建、bitwise/index/stack解码kernel和写入复制；
采样仍需输出完整FP32网络输入，所以更小的持久缓存不等于同倍数的取样加速。
这里的百分比仅适用于存储操作，整体训练是否受影响须看八卡完整PPO计时。

## 八卡训练吞吐

已比较：历史去重／2048，以及位打包＋去重／2048、4096、8192。
每组使用独立输出目录，从同一U14 checkpoint、Adam和recent8历史池恢复3轮至U17；
首轮为编译暖身，比较后2轮。每卡N=16,384，T32、E4、64逻辑minibatch，
每卡逻辑批8192、全局65,536，每轮4,194,304条fresh learner transitions。
使用官方随机18–23格分布、23×23画布、BF16、有效token attention、模型编译和KL probe。
每层bank为512张，评估关闭；阶段计时含双方推理、env、重置、GAE与PPO，排除初始化、存盘和历史池发布I/O。
各阶段按最慢rank记录，fresh样本只计一次，不乘PPO epochs。

| 存储 | 物理批／卡 | 采样均值 | GAE＋PPO均值 | fresh transitions/s | 相对本轮历史去重2048 |
|---|---:|---:|---:|---:|---:|
| 历史去重 | 2048 | 17.36 s | 105.45 s | 34,152 | 基准 |
| 位打包＋去重 | 2048 | 18.36 s | 105.14 s | 33,962 | −0.56% |
| 位打包＋去重 | 4096 | 17.43 s | 101.96 s | 35,129 | +2.86% |
| 位打包＋去重 | 8192 | — | — | OOM，未完成首轮 | — |

后三组中可用两组的单轮速率分别为33,857–34,068和35,007–35,253/s，
对照为33,785–34,527/s。每项只有两轮正式测量，不将−0.56%解释为已证明的长期回归或完全无开销。
此前实验的历史去重4096为35,450/s，采用三轮正式测量，属于不同批次，不能混作本轮严格匹配对照。

| 存储 | 物理批／卡 | allocated峰值 | reserved峰值 |
|---|---:|---:|---:|
| 历史去重 | 2048 | 86.34 GiB | 119.29 GiB |
| 位打包＋去重 | 2048 | 52.00 GiB | 83.43 GiB |
| 位打包＋去重 | 4096 | 79.05 GiB | 137.63 GiB |

上表为所有rank的最大峰值，包含首轮编译暖身，不能与仅分配静态缓存的18.694 GiB混用。
reserved包含分配器缓存，波动明显；例如packed4096的rank0在U17更新后的当前allocated为22.38 GiB、
当前reserved为91.70 GiB，均低于整组峰值。显存压缩没有保证分配器保留量按相同比例下降。
本次每层512张的bank比正式65,536张约少2.14 GiB；完整bank与异步GPU评估的组合尚未实测。

## 8192失败诊断与验收

单卡随机尺寸的一次前后向能运行，不代表完整PPO能运行。真实八卡8192在首轮内发生OOM，
此时尚无可用于吞吐统计的完整更新。未自动缩批，也没有将失败样本计入结果。

异常清理还暴露出既有分布式退出问题：py-spy捕获rank0的异常为torch.OutOfMemoryError，
其主线程停在destroy_process_group；其他rank在adaptive LR collective中等待。
确认后只向本次8192 torchrun发送SIGTERM，保留PID/start ticks核验、堆栈、异常locals及终止记录，
所有GPU显存随后释放。distributed_training.py与原395文件源码的SHA一致；
本轮未改动分布式错误处理，实现正常路径和OOM清理是分开的验证范围。

验收确认三项成功运行均从U14恢复至U17，新增12,582,912条样本，课程累计71,303,168；
三个checkpoint模型有限、optimizer存在，原U14 checkpoint的SHA保持不变。
原实验395文件、历史去重396文件、位打包398文件的冻结源码均通过SHA核验，
GPU精确PPO对照最大差异0。

## 建议

本次无损压缩的显存收益达到预期，同物理批吞吐接近原实现。
需要更多显存余量时使用压缩＋2048；本轮推荐的较快组合为压缩＋4096。
8192不适合作为当前通用配置：真实更新OOM，且全23×23单卡探测也OOM。
主要更新开销仍在GAE＋PPO，压缩存储并不会按容量缩减比例提高网络计算速度。
实验完成时未修改默认或恢复原长训练；随后已将该存储方案设为默认，并在新目录从头运行224轮。
原U14实验仍保持停止，后续状态见上述新训练报告。

## 产物

H200：`/DATA/disk2/yuhangwang/code/generals-rl/outputs/experiments/packed_storage_20261103/`。
本地：`outputs/analysis/packed_storage_20261103/`。
原历史去重实现继续使用上一轮396文件冻结源码；候选另存398文件及SHA清单。
正确性JSON、显存JSON、profile traces、有效配置、训练日志和checkpoint均保存在独立实验目录，权重不进入Git。
本地 `summary.json` 汇总结果，`evidence/acceptance.json` 为验收记录，
`evidence/diagnostics/` 保存8192失败及人工终止的完整证据。
