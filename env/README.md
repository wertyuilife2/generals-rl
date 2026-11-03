# generals-env

游戏环境子项目，源码包名为 `generals`，发行包名为 `generals-env`。支持 CPU/CUDA/MPS 批量模拟；人类游玩使用 [Arena 网页界面](../arena/README.md#人类参赛)。

当前规则固定为 2026-10-28 核查的官网 **31.4.5 普通 1v1 / FFA**，默认按官方尺寸与公平性规则随机生成地图。
CPU、CUDA、MPS 和 Arena 使用相同动作时序、迷雾及结算语义；版本证据与验证边界见[实施记录](../doc/official_rules_alignment.md)。

从仓库根目录安装和启动：

```bash
conda run -n generals-rl python -m pip install -e ./env -e ./baselines --no-deps --no-build-isolation
conda run -n generals-rl python -m generals simulate --players 4 --max-ticks 2000 --seed 42
```

不安装时，在仓库根目录使用 `PYTHONPATH=env/src:baselines/src python -m generals ...`。
纯规则与批量环境只需要 `generals-env`；AI 模拟还需要 `generals-baselines`，对应 `simulate` 可选依赖组。
这条 `PYTHONPATH` 写法用于 Bash 等 POSIX shell；当前内置 AI 与 CPU 原生后端支持 Linux/macOS。

CPU 原生后端支持 Linux 与 macOS。Linux 需要支持 OpenMP 的 C++17 编译器；macOS 需要
Apple clang 与已安装的 `libomp`，非标准位置可用 `GENERALS_LIBOMP_PREFIX` 指定。
首次使用自动编译，后续复用缓存。单局调用 `generals.make_engine(state)`，批量调用 `VecEnv`；
二者执行同一个 `backends/cpu/step.cpp`，Arena 不再维护执行后端。Python `CoreEngine` 继续作为独立规则参考。

CPU/CUDA/MPS 共用 `VecEnv.step()`、`observe(player_ids)` 和可复用的 `observer(player_ids)`。
批量观测由 env 返回原始局部信息；特征、历史、知识、动作映射与模型推理属于 `rl/` 的策略实现。
CPU 的迷雾、公开障碍、统计脱敏和生命周期字段由一次 C++ 调用生成，返回独立快照；
不增加观测线程。NumPy 参考实现保留用于逐字段一致性验证，Arena 和策略无需自行重建观察。

Apple Silicon 可用 `device="mps"`，需要支持 `torch.mps.compile_shader` 的 PyTorch。
Metal 内核直接编译同一份 `backends/rules.h`，共享行动排序、交换将军、增长、离场宽限、AFK 和最终结算；
每 tick 在 GPU 上执行，不回退到 CPU。状态仍使用 int64 兵力、tick 和伤害，单张地图最多 4096 格。
`fog_obstacles`、`has_kill`、`warn_afk`、`neutralize_in` 与 CPU/CUDA 使用相同观察边界；
迷雾中的山和普通城市只公开不可区分的障碍标记，未被看到的已转化将军城市仍隐藏。
`neutralize_in` 在设备侧为 float32，其他整数与布尔状态逐项精确对齐。

```python
import torch
from generals import MapConfig, VecEnv

with VecEnv.from_config(MapConfig(players=4), batch_size=1, device="mps") as env:
    read = env.observer([0])
    env.step(torch.zeros((1, 4), dtype=torch.int64, device="mps"))
    raw = read()  # 观察留在 MPS 设备上
```

MPS 的回归覆盖官方回放、动作优先级与交换、50 tick 离场中立化、AFK、自动/部分重置及批量迷雾观察。
CPU/MPS 对照 benchmark 仍使用预先生成的相同动作，逐 tick 计时 `step` 加完成同步：

```bash
PYTHONPATH=env/src python -m generals.benchmarks.step --backend both --device mps \
  --sizes 25 --players 4 --batch-sizes 128 --steps 200 --warmup 50 --repeats 3 \
  --output outputs/mps-step.json
```

规则、操作、依赖和开发说明统一维护在 [doc/README.md](../doc/README.md)：

- [安装与平台支持](../doc/README.md#quickstart)
- [网页游玩](../arena/README.md#人类参赛)
- [统一 CPU/GPU API 与状态协议](../doc/README.md#api)
- [观测、路径和控制器接入](../doc/README.md#features)
- [基准及性能结果](../doc/README.md#benchmark)
- [环境性能验证记录](../doc/README.md#results)
- [测试与历史数据](../doc/README.md#verification)
