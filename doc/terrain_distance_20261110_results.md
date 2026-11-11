# Temporal + HL128：路径与时限消融

更新：2026-11-11T01:59:56+00:00；队列状态：`v30_9x9_temporal_hl128`。

[实验配置与路径定义](terrain_distance_20261110.md)。三组各224轮、单种子、3.758B新玩家样本。当前与其他任务共享8×H200，速度不可当作独占吞吐。

| Run | PPO轮 | 原时限 | 原时限七对手 | 统一2048四基线 | 统一2048七对手 | 玩家样本/s | 峰值allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| v30_9x9_temporal_hl128 | 5/224 | 2048 | — | — | — | 170,653 | 19.35/20.56 |
| v31_9x9_temporal_hl128_4096 | 0/224 | 4096 | — | — | — | — | 0.00/0.00 |
| v32_9x9_temporal_hl128_paths | 0/224 | 2048 | — | — | — | — | 0.00/0.00 |

![统一2048 tick](assets/terrain_distance_20261110/curves_cap2048.png)

![统一1024 tick及旧run参考](assets/terrain_distance_20261110/curves_cap1024.png)

短时限曲线从同一局的终局tick截取：超过时限的胜/负均计平局；不是再次独立采样。模型内部clock仍使用其训练配置。旧v25/v28仅在1024口径图比较；旧训练上限2000，本轮2048/4096，且旧吞吐在独占GPU下测量。

TensorBoard：`outputs/experiments/terrain_distance_20261110_actual01/runs/<run>/events.out.tfevents.*`。
