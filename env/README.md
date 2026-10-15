# generals-env

游戏环境子项目，源码包名为 `generals`，发行包名为 `generals-env`。支持 CPU/CUDA/MPS 批量模拟及 Tk 桌面游玩。

从仓库根目录安装和启动：

```bash
conda run -n generals-rl python -m pip install -e ./env -e ./baselines --no-deps --no-build-isolation
conda run -n generals-rl python -m generals play --players 4 --seed 42
```

不安装时，在仓库根目录使用 `PYTHONPATH=env/src:baselines/src python -m generals ...`。
纯规则与批量环境只需要 `generals-env`；AI 游玩和模拟还需要 `generals-baselines`。

CPU 原生后端支持 Linux 与 macOS。Linux 需要支持 OpenMP 的 C++17 编译器；macOS 需要
Apple clang 与已安装的 `libomp`，非标准位置可用 `GENERALS_LIBOMP_PREFIX` 指定。
首次使用自动编译，后续复用缓存。单局调用 `generals.make_engine(state)`，批量调用 `VecEnv`；
二者执行同一个 `backends/cpu/step.cpp`，Arena 不再维护执行后端。Python `CoreEngine` 继续作为独立规则参考。

Apple Silicon 可通过 `make_env(..., device="mps")` 选择原生 Metal 后端，需要支持
`torch.mps.compile_shader` 的 PyTorch（本机验证版本为 2.14.1）。普通 `step(actions)`
与 CUDA 接口相同；性能比较统一复用 `benchmarks/step.py`，每 tick 等待完成，排除动作生成。
安装依赖入口为 `python -m pip install -e './env[gpu]'`；MPS 本身不需要 CUDA 或 libomp。
API、缓存约束及可复现吞吐测量见 [统一文档](../doc/README.md#api)。

## 随机地图生成

`MapGenerationConfig` 提供标准随机地图参数，默认值对应 generals.io v31.4.3 的普通
2–8 人地图。Arena 默认使用这一接口：

```python
from generals import MapGenerationConfig
from generals.core.random_map import sample_map_config
from generals.core.mapgen import generate_map

options = MapGenerationConfig()  # 官方默认
config = sample_map_config(players=2, seed=42, options=options)
state = generate_map(config, seed=42)

# 同一套生成规则也可使用指定尺寸或自定义参数。
options = MapGenerationConfig(city_army_min=30, city_army_max=60, spawn_fairness=.8)
config = sample_map_config(4, 42, options, width=25, height=25)
state = generate_map(config, 42)
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `width`, `height` | `0.5` | 官方尺寸滑块，0–1；固定宽高可覆盖它们 |
| `mountain_density`, `city_density` | `0.5` | 官方密度滑块，0–1，并非格子占比 |
| `spawn_fairness`, `city_fairness` | `0.5` | 出生可扩张空间与城市路径分配公平性 |
| `min_general_spacing` | `9` | 随人数调整前的曼哈顿距离排除阈值；1v1 得到 `>14.4` |
| `city_army_min`, `city_army_max` | `40`, `50` | 守军随机范围，按四舍五入生成；配对城市共享值 |
| `close_city_distance` | `6` | 出生点附近城市的曼哈顿距离上限 |
| `require_close_city` | `True` | 城市数量足够时，要求每位玩家附近有城市 |
| `max_attempts`, `spawn_attempts`, `city_attempts` | `1000`, `1000`, `8` | 有界重试预算；不可满足时抛出 `MapGenerationError` |

自动尺寸采用 `round(B + (4 + P/2) * U²)`，其中
`B = 16 + min(8, P) + max(0, (P-6)/2)`，`U` 为独立均匀随机数；非默认尺寸滑块调整基准长度。
城市目标数量按 `5 + P * (2 + U)` 计算，再应用城市滑块和公平性放置。
山地先按 `面积 * (0.2 + 0.08 * U)` 的强度重复撒点，并预留转成城市的障碍；
实际山地占比不等于该强度。不合格地图整张重试，不拆山修复，必须存在将军间无需攻城的路径。

`sample_map_config` 返回通过校验的尺寸和 `generation_attempt`，可用
`MapConfig(**json.loads(json.dumps(dataclasses.asdict(config))))` 跨进程复现。
生成缓存只保存初始状态，返回独立动态数组。数据结构及现有 CPU/GPU 战斗后端保持共用。

规则参考：[官方 v31.4.3 公开客户端](https://generals.io/generals-main-prod-v31.4.3-485dbf55.js)，
[官方更新记录](https://generals.io/versions)。支持普通 FFA/1v1，不包含组队或特殊 modifier；
本地 RNG 与官方 JavaScript RNG 不逐种子对应。零城市密度明确生成零城市。

为保持固定尺寸训练和历史地图不变，直接使用原有 `MapConfig(...)` 时，
`generation=None` 继续使用原生成器和原种子序列。它的顶层 `mountain_density`、
`city_density` 是面积比例；启用 `generation` 时改用嵌套参数中的官方滑块，顶层密度不参与生成。
未调整 RL 配置或训练分布。

规则、操作、依赖和开发说明统一维护在 [doc/README.md](../doc/README.md)：

- [安装与平台支持](../doc/README.md#quickstart)
- [GUI 操作](../doc/README.md#gui)
- [统一 CPU/GPU API 与状态协议](../doc/README.md#api)
- [观测、路径和控制器接入](../doc/README.md#features)
- [基准及性能结果](../doc/README.md#benchmark)
- [当前实现的性能验证](../doc/README.md#results)
- [测试与历史数据](../doc/README.md#verification)
