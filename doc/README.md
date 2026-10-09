# Generals 环境开发与使用说明

更新日期：2026-10-03。本文说明当前唯一的 CPU/OpenMP 和 GPU/CUDA 单步实现，以及 GUI、规则和测量方法。

项目提供 2–8 人 Generals 对战、单个人类与 AI 的桌面游玩、无界面模拟，以及 CPU/OpenMP 和 GPU/CUDA 批量环境。规则源自原 JS 游戏，交互参考 generals.io；不保证与官网或原 JS 的随机决策逐步相同。强化学习算法独立放在 `rl/`，本文说明环境本身及其接口。

- [安装与运行](#quickstart)
- [游戏规则、地图与 AI](#rules)
- [GUI 操作](#gui)
- [架构与依赖边界](#architecture)
- [批量环境 API](#api)
- [观测、路径与 RL 扩展](#features)
- [Benchmark 与 profiling](#benchmark)
- [当前性能验证](#results)
- [验证与历史存档](#verification)

<a id="quickstart"></a>
## 1. 安装与运行

### 依赖与平台

| 用途 | 依赖与支持情况 |
|---|---|
| 参考引擎、无界面 AI 模拟 | Python ≥3.11、NumPy ≥1.24；不需要 PyTorch/CUDA |
| GUI | 额外需要 Tk 和图形桌面；源码兼容 Windows/Linux，Linux/X11 已实测，Windows 尚未实机验收 |
| CPU 批量模拟 | 当前原生加载器支持 Linux，需要 C++17/OpenMP 编译器，可用 `CXX` 指定 |
| GPU 批量模拟 | Linux、PyTorch ≥2.13、CUDA 设备、nvcc/CUDA 开发库、C++ 编译器和 Python 开发头文件 |

在仓库根目录安装到现有 conda 环境：

```bash
conda run -n generals-rl python -m pip install -e ./env -e ./baselines --no-deps --no-build-isolation
conda run -n generals-rl python -m generals play --players 4 --seed 42
```

`--no-deps --no-build-isolation` 适用于依赖和 setuptools 已齐备的本机环境。新环境可使用 `python -m pip install -e ./env`；GPU 可选依赖入口为 `python -m pip install -e './env[gpu]'`，仍需准备适配的驱动和编译器。`python -m tkinter` 可检查 Tk 窗口。

不安装包时，可在仓库根目录设置源码路径运行（Windows 的路径分隔符用 `;`）：

```bash
PYTHONPATH=env/src:baselines/src conda run -n generals-rl python -m generals play --players 4 --seed 42
```

也可在 Linux 仓库根目录使用 `PYTHONPATH=env/src:baselines/src conda run -n generals-rl python -m generals ...`。基准脚本会自行加入源码搜索路径，可直接从仓库根目录运行。

### 常用命令

安装后在仓库根目录运行：

```bash
conda run -n generals-rl python -m generals watch --players 8 --size 35 --speed 10
conda run -n generals-rl python -m generals simulate --players 4 --size 25 --games 10 --max-ticks 20000 --seed 42
conda run -n generals-rl python -m generals simulate --players 2 --ais aggressive,random --games 1 --max-ticks 2000 --record outputs/game.jsonl
conda run -n generals-rl python -m generals replay outputs/game.jsonl
```

`simulate` 没有实时等待，输出对局数量、胜者、截断和吞吐等 JSON 统计。`replay` 校验初态、已记录动作及逐 tick 摘要，不调用 AI，也不是图形录像播放器。输出/录像使用新路径，避免覆盖已有结果。

地图参数包括 `--players 2..8`、`--size 15|25|35`，或 `--width/--height`；默认 25×25、4 玩家。山脉密度默认 0.2、范围 0–0.4；城市密度默认 0.1、范围 0–0.2。`--visibility local|full` 控制策略视野，`--seed` 为非负整数。`--ais` 按 AI 席位填写逗号分隔的策略名；GUI 人类席位通过 `--human-seat` 指定。完整参数可用各子命令的 `--help` 查看。

<a id="rules"></a>
## 2. 游戏规则、地图与 AI

### 回合和战斗

1. 一个 tick 先增兵，再按玩家编号 0…P−1 处理行动，每位存活玩家最多一步。此前发生的占领、淘汰和继承立即影响后续玩家。
2. 主城及已占城市每 2 tick 增加 1 兵；已占普通地块每 50 tick 增加 1 兵。第 50 tick 城市没有额外领土增兵，中立城市不增长。
3. 玩家可 WAIT，或从己方格向上下左右相邻的非山脉格移动。普通派出 `max(0, n-1)`，半兵派出 `n//2`。执行时来源为 0/1 兵，合法指令仍消耗行动并转移 0 兵，不等待积兵；无效游戏移动也消耗时隙。
4. 移入己方格合兵；其他目标按 1:1 抵消。进攻兵力严格大于守军才占领，打平留下 0 兵但归属不变。0 兵不能占领空地。
5. 攻占敌方主城立即淘汰该玩家，主城变为普通城市；其余领土交给胜者，驻军变为 `max(1, n//2)`。直接攻下的主城按战斗后的剩余兵力保留，不再减半。
6. 最后一名存活者立即获胜，本 tick 尚未行动的被淘汰玩家跳过。达到时间上限是截断，不按兵力强行指定胜者；自然终局优先于同 tick 的截断。

兵力与 tick 使用 `int64`。CPU/参考引擎保留溢出报错；GPU 为吞吐移除保护性溢出检查，调用方须保证输入、计算中间值、公开总兵力和 tick 均在 int64 范围内。超出此范围的 GPU 行为不作保证。API 参数类型/形状错误与游戏内无效移动分开处理。

### 地图生成

每局先随机打乱所有格点，无放回选取 `floor(H×W×mountain_density)` 个山脉和 `floor(H×W×city_density)` 个中立城市；城市初始 40 兵。城市没有最小间距或排斥采样，因此随机地图可能出现相邻城市和小簇。

主城从其余普通格选择，尝试保持出生点距离；空间不足时逐步放宽要求。每位玩家起始只有主城及 1 兵。所有非山脉格（包括空地、城市和主城）保证四邻接连通。

连通修复先检查一个连通分量，必要时执行一次多源 0–1 BFS，再沿共享父链移除山脉，时间和额外空间均为 `O(H×W)`。它保留城市数量和主城位置，实际山脉密度可能略低于设置值；不承诺全局最少移除山脉。

相同版本、配置和种子可复现。批量初始化按 `[seed, lane, 0]` 派生各环境的地图种子，不是复制同一张图；扩大批量时已有 lane 的地图保持稳定。主种子还用于独立策略随机流；预提交动作的 `step()` 本身不生成决策。

### 四种基础策略

| 名称 | 行为 |
|---|---|
| `aggressive` | 偏向合兵、探索和进攻，高估值敌方主城及城市 |
| `expansion` | 优先空地和可攻取城市，扩张到空地且兵力较多时使用半兵 |
| `defensive` | 支援薄弱领土和受威胁结构，谨慎扩张，部分支援使用半兵 |
| `random` | 带占空地、优势进攻等偏置的随机评分，较多兵力时可能分兵 |

策略保留 JS 的行为倾向，使用有界候选搜索和独立随机流；不保证与 JS 每步选同一动作。路径失效、来源丢失或攻击未占领时会清除/重新规划，策略可能僵持，模拟用时间上限截断。benchmark 的随机动作生成器与这里的 `RandomAI` 是不同工作负载。

<a id="gui"></a>
## 3. GUI 操作

GUI 使用 Tk、参考引擎和 `Runner`，支持一个人类对抗其余 AI，或全 AI 观战。启动后先预览地图，设置人数、策略、地图和视野，再点击“开始”。暂停时可规划队列和完整 tick 单步。

| 鼠标／按键 | 操作 |
|---|---|
| 左键己方格 | 选择来源 |
| 左键规划光标相邻格、WASD、方向键 | 追加一步移动/攻击，规划光标前移 |
| 重复点击规划光标、Z、半兵按钮 | 下一条入队指令使用半兵，入队后恢复普通模式 |
| Shift+左键；取消选择后再点己方格 | 强制换来源，清除旧队列 |
| Q、清空队列 | 清空尚未执行的指令 |
| E、撤销末步 | 删除最后一条尚未执行的指令 |
| 空格、取消选择 | 取消选择，已排队指令继续执行 |
| H、选择主城 | 选中自己的主城 |
| 中键/右键拖动；滚轮或 +/− | 平移；缩放 |
| Home、适应窗口 | 居中并适配地图 |
| P、暂停；单步按钮 | 暂停/继续；暂停时推进一个完整 tick |

队列最多 64 步，逐格输入，不自动寻路到远处点击目标。零兵移动到己方格会正常推进队列；未占领目标或来源失守则清除失效的后续路径。所有基本操作都有鼠标入口。

“同图重开”恢复初始地图和种子，“新地图”使用新种子；人类淘汰后可继续全图观战。人类模式窗口失焦自动暂停。滚轮按 Tk 窗口系统适配，缺中文字体时使用英文标签。高吞吐模拟使用无界面模式；GUI 的有限 tick 追赶用于保持交互响应。

<a id="architecture"></a>
## 4. 架构与依赖边界

```text
env/
  src/generals/
    api.py / vector.py       统一创建入口与 VecEnv
    protocol/                配置、动作、Controller、数组字段、BatchStep/VectorBackend
    core/                    独立参考规则、状态、地图、观测、路径、记录
      batch.py               CPU/GPU 共用的冷初始化和状态转换
    backends/cpu/            C++/OpenMP 单 tick 执行、NumPy/ctypes 缓冲区
    backends/gpu/            CUDA 单 tick 内核、Torch 状态与原生启动桥
    controllers/             人类队列及按需加载 baseline 的交互工厂
    runtime/                 Runner 与 PlaySession
    gui/                     Tk 展示与输入
    benchmarks/              workload、step 计时、CPU/CUDA profiling
  benchmarks/                benchmark_step.py、profile_step.py 两个入口
  tests/                     规则、设备、接口、GUI 和工具测试
baselines/                   独立的手写策略包 generals_baselines
arena/                       统一对战、版本登记、结果记录与 Elo 排名
rl/                          算法、模型、训练与模型专用特征
outputs/                     运行结果与本地历史归档
doc/README.md                本文，统一文档入口
```

依赖方向是 `baselines → generals`、`arena → generals`、`rl → generals + baselines + arena`；环境不反向导入训练代码。交互工厂仅在创建 AI 时加载 baselines，规则和批量环境不依赖策略实现。CPU/GPU 共用数据协议、`VecEnv` 和冷初始化工具，热路径保留独立实现以便互相校验。参考规则、四种 AI、GUI 不并入优化内核。

CPU 每次 native 调用推进一个 tick，ctypes 释放 GIL，OpenMP 静态分配环境并行；每局内部仍按玩家顺序行动。增量维护统计及城市/主城增长索引，周期性全图增长、继承、重置和缓存重建保留必要扫描。

GPU 使用 CUDA 单 tick 内核：棋盘常驻设备张量，统计和增长索引跨调用缓存；默认根据 batch 选择 32×4（B≤2048）、8×16（2048<B<32768）、32×8（B≥32768），组内 leader 执行顺序行动。共享库与 launch 参数持久保存，正常 step 经 CPython 原生桥更新动作指针和当前 stream；宽高/玩家数为编译常量，缓存已占领结构的增长索引、统计和计数，总格点数至少 2000 万时还使用 32 位格点元数据缓存。不发生逐 tick CPU 地图处理或状态上传。

`VecEnv.step/synchronize` 在构造时直接绑定后端方法，不增加每步转发层。CUDA 编译器可用 `GENERALS_NVCC` 显式指定，否则优先查当前 Python 环境，再查 PATH。编译结果按源码、工具链及形状/线程配置缓存到用户私有临时目录；首次编译不计入基准。缺少依赖时明确报错。GPU 只接受默认配置或兼容参数 `kernel_backend="cuda"`。

公开根包只有 `make_env, VecEnv, MapConfig, Action, ActionKind, Direction, MoveMode, BatchStep`。其他功能从所属子模块导入；旧 `generals_env/generals_cpu/generals_gpu/generals_protocol` 包名已停用。发行包名仍是 `generals-env`，控制台命令可用 `generals` 或 `generals-env`。

<a id="api"></a>
## 5. 批量环境 API

### 创建与推进

CPU 例子：

```python
import numpy as np
from generals import MapConfig, make_env

with make_env(MapConfig(25, 25, 4), device="cpu", batch_size=10000,
              threads=26, auto_reset=True, episode_ticks=500, seed=42) as env:
    actions = np.zeros((env.batch_size, env.players), dtype=np.int64)
    result = env.step(actions)
    env.check_errors()
```

GPU 使用相同入口，动作预先放到设备上，省略 CPU 专属的 `threads`：

```python
import torch
from generals import MapConfig, make_env

with make_env(MapConfig(25, 25, 4), device="cuda", batch_size=10000,
              kernel_backend="cuda", auto_reset=True, episode_ticks=500, seed=42) as env:
    actions = torch.zeros((env.batch_size, env.players), dtype=torch.int64, device=env.device)
    result = env.step(actions)
    env.synchronize()
    env.check_errors()
```

默认 `mode="vector"`，返回 `VecEnv`。实时入口为 `make_env(MapConfig(players=4), mode="realtime", seed=42).run()`，返回的类型是 `generals.runtime.play.PlaySession`；`watch=True` 启用观战，实时模式使用 CPU 参考引擎。

批量后端支持每图最多 4096 格、2–8 玩家，同批次宽、高和人数相同，各自地形/初态可以不同。`VecEnv.from_states(states, device=...)` 导入已有 `GameState`；`VecEnv(initial, width=..., height=..., device=...)` 接受对应 NumPy/Torch 字段字典；`to_states()` 显式导出独立 CPU 状态副本。

### 状态和动作协议

设 `B` 为环境数、`N=H×W`、`P` 为玩家数，格点编号 `cell=y×W+x`。`env.state` 为下表数组/张量字典：

| 字段 | 形状 | dtype | 含义 |
|---|---|---|---|
| `terrain` | `[B,N]` | `uint8` | 0 平地，1 山脉 |
| `structure` | `[B,N]` | `uint8` | 0 普通地，1 城市，2 主城 |
| `owner` | `[B,N]` | `int16` | −1 空地，−2 中立城市，0…P−1 玩家 |
| `army` | `[B,N]` | `int64` | 非负驻军 |
| `alive` | `[B,P]` | `bool` | 存活标志 |
| `general_pos` | `[B,P]` | `int32` | 初始主城位置，淘汰后仍保留 |
| `tick` | `[B]` | `int64` | 已完成 tick 数 |
| `terminated` | `[B]` | `bool` | 自然终局 |
| `winner` | `[B]` | `int32` | 胜者编号，未终局为 −1 |

动作是对应设备的 `int64[B,P]`：

```text
WAIT = 0
MOVE = 1 + source × 8 + direction × 2 + mode
direction: 上=0，右=1，下=2，左=3
mode: 全部留一=0，半兵=1
```

`Action.encode()/decode()` 提供标量编码。负编码作为无效移动处理，不能解码为原地转移。提交的是预选联合动作，不会在玩家行动前自动重新调用策略；若需要严格逐时隙观测和决策，使用参考 `Runner` 或引擎事务接口。

### 输出、重置和同步

| `BatchStep` 属性 | 形状与类型 | 含义 |
|---|---|---|
| `moves` | `int64[B,P,9]` | `active, valid, source, target, moved, captured, eliminated, target_owned, reason` |
| `stats` | `int64[B,P,3]` | 总兵力、领土数、已占城市数 |
| `done` | `bool[B,2]` | 自然终局、时间截断 |
| `errors` | `int32[B]` | CPU：0 正常、非零为数值错误；GPU：保留协议字段，不再检测溢出 |

动作 `reason`：0 跳过、1 WAIT、2 移动、3 越界、4 非己方来源、5 山脉。判断行动先看 `active`，再看有效性与结果。

输出借用内部缓冲区，下一次 step/reset 会覆盖；保存历史须显式 `.copy()` 或 `.clone()`。`env.state/stats/done` 是公共视图；初态、计数器、RNG、线程和 kernel 配置通过 `env.unwrapped` 访问。

默认 `auto_reset=False`，自然结束或达到有效 `episode_ticks` 上限后冻结对应环境；0 表示不设时限。开启自动重置后，step 内恢复各环境缓存的初态，**`state` 已是新局，而 `stats/done/moves` 仍描述结束 tick**。需要终止棋盘或终止观测时，关闭自动重置，读取后再 `reset(mask)`。`mask` 为对应设备的 `bool[B]`；显式 reset 恢复初态及初始 RNG，清空所选环境诊断和计数，未选环境不变。

CPU step 同步完成，CUDA 在当前 stream 上异步提交。计时使用 `synchronize()` 等待完成；`check_errors()` 检查协议错误字段并可能同步，不由 `synchronize()` 代替。CPU 数值错误冻结受影响环境，失败 tick 不保证完整，检查后显式重置；GPU 不再提供溢出检查或相应冻结保证。实例的 step/reset 串行调用；跨 CUDA stream 使用时调用方负责建立依赖。

直接编辑 CPU 状态后调用 `invalidate_cache()`；GPU 正常 PyTorch 原地修改会通过版本计数失效缓存，原生 CUDA 写入、`.data` 或 inference-mode 编辑须显式失效。不能替换底层存储、shape 或 dtype，也不要与在途 step 并发改写。只读特征无需失效缓存。使用上下文管理器或 `close()` 释放环境引用。

<a id="features"></a>
## 6. 观测、路径与 RL 扩展

LOCAL 视野覆盖己方领土及八邻域，移动仍是四邻接；山脉不遮挡该范围视野。未知地形/结构为 255、归属为 −3、兵力为 −1，这些值只出现在观测中。总兵力、领土和存活状态公开，敌方城市数在 LOCAL 下为 −1。FULL 明确开放全图；观战 GUI 可显示全图，AI 默认仍各用局部视野。

参考接口位于 `generals.core.observation`：`observe`、`snapshot`、`legal_action_mask` 返回独立观测/掩码。GPU 通过 `env.unwrapped.observe(player_id=0, mode="local")` 和 `legal_action_mask(player_id=0)` 在设备端构建独立张量；地图形状 `[B,H,W]`，移动掩码 `[B,N,4,2]`、WAIT 掩码 `[B]`。观测根据当前状态计算，因此自动重置后属于新局。0/1 兵和预期打不赢的移动仍可合法，掩码不保证后续玩家行动时状态不变。

路径工具在 `generals.core.pathfinding`：`neighbors`、`single_source_distances`、`shortest_path`、`DistanceCache`、`prepare_map`、`all_pairs_distances`。距离为四邻接非山脉图上的边数，不包含留兵、攻城或动态威胁；不可达距离为 −1，路径为 `None`。全点对距离显式计算为 `int32[N,N]`，默认输出预算 128 MiB，需要约 `4N²` 字节；单源缓存按地形内容、尺寸、未知格策略和来源区分，并限制容量。

LOCAL 策略只可使用观测/已知地形及其记忆。`prepare_map(state.layout)`、真值全图距离和 `env.state` 是特权数据，不能通过预计算路径绕过迷雾。未来模型的通道编码、归一化、历史堆叠、奖励及多智能体训练适配放在 `rl/`，游戏规则和通用拓扑保留在环境。

参考引擎/控制器的最小示例：

```python
from generals import MapConfig
from generals.core.mapgen import generate_map
from generals.controllers import make_controllers
from generals.runtime.runner import Runner

state = generate_map(MapConfig(25, 25, 4), seed=42)
controllers = make_controllers(["aggressive", "expansion", "defensive", "random"], seed=42)
summary = Runner(state, controllers, seed=42, visibility="local", max_ticks=20000).run()
```

外部控制器实现 `generals.protocol.controller.Controller` 的 `reset/act/on_result` 协议。`Runner` 不隐式重置调用者传入的控制器；新局创建新状态和控制器。更低层的 `CoreEngine` 在 `generals.core.engine`，支持 `step(actions)` 或 `begin_tick → next_player/observe/apply_action → finish_tick`。

<a id="benchmark"></a>
## 7. 逐 step benchmark 与 profiling

### 唯一计时口径

只测公开 `VecEnv` 的一整个 batch tick：

```python
for action in prepared_actions:
    start = perf_counter_ns()
    env.step(action)
    env.synchronize()
    latency_ns = perf_counter_ns() - start
```

计入 Python API、动作验证、原生调用、完整规则、统计/动作结果/done/errors/计数、局内自动重置，以及**每一步**的完成等待。排除决策生成、动作视图/上传、地图生成、初始化/编译/预热、样本间 reset、结果读回与校验、观测、奖励特征、策略推理和训练。

- **整批 step/s** = `1 / 平均 step 秒数`。
- **环境步/s** = `B / 平均 step 秒数`，这里 B 是环境数，不是玩家数。
- 每轮保存逐 tick 延迟及平均值、P50、P95；多轮主要比较平均 step 延迟的中位数，不用单步 P50 换算吞吐。

每轮从相同初态重放相同动作，预热和测量之间 reset，测量包含冷缓存首步和 step 内的自动重置。每轮测量后验证全部状态、统计、动作结果、done/errors、计数及 RNG 指纹，检查恰好推进 `B × steps` 个环境步。动作与结果验证均在计时外。

### 动作负载 v2

当前 workload 为 **`pre_tick_owned_mobile_step_v2`**。在每个 tick 开始前，所有玩家从同一状态生成动作：均匀选择己方兵力 >1 的格子，再均匀选择非山脉邻格方向和全兵/半兵；没有可用来源则 WAIT。状态通过实际 CPU step 推进，生成完整 `[T,B,P]` 动作历史，再关闭参考环境。CPU/GPU 使用同一初态和同一份动作，GPU 在计时前一次性上传。

采样器只负责随机策略，没有游戏规则；策略 RNG 独立于环境 RNG。规则另由独立 Python `CoreEngine` 的逐 tick 回归测试验证；缓存测试同时比较每步重建与持续复用的结果。所有玩家同时决策后，仍按座位顺序执行；前序玩家可能改变后序玩家的来源，因此少量无效/零兵动作是正常结果，报告保留相关计数。

### 两个入口

以下在仓库根目录运行；无需安装时，两个脚本会自行定位 `env/src`。输出用新路径，不能覆盖已有文件：

```bash
# CPU/GPU 同一负载对比；也可 --backend cpu 或 gpu
OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=spread OMP_PLACES=threads conda run -n generals-rl python env/benchmarks/benchmark_step.py --backend both --threads 26 --sizes 25 --players 4 --batch-sizes 1000 10000 100000 --output outputs/step.json

# 同样的逐步基准，加一次单独的 CPU/GPU profile
OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=spread OMP_PLACES=threads conda run -n generals-rl python env/benchmarks/profile_step.py --backend both --threads 26 --batch-sizes 10000 --output outputs/profile.json

# 已安装包也可直接使用同一个 benchmark 实现
conda run -n generals-rl python -m generals benchmark --backend cpu --threads 26 --output outputs/cpu-step.json
```

默认参数：25×25、4 玩家、10,000 环境、1000 测量 tick、每轮 1000 预热 tick、3 轮、horizon=500、seed=42。`--sizes`、`--players`、`--batch-sizes` 支持列表，`--sizes` 也支持 `40x25`。批量先比较 1k/10k/100k，再按结果显式细分。CPU/GPU 依次测量，不同时争用计算资源。

CPU 默认 `--threads 0` 由 OpenMP 可用处理器数决定；本机此前扫描的最佳设置是 26 线程、静态调度、`ACTIVE/spread/threads`。这是本机测量结果，并非所有硬件的最优值。库不会修改进程的 OpenMP 设置。

GPU 使用上述自适应线程配置，`--gpu-group-threads` 和 `--gpu-envs-per-block` 可覆盖。GPU 原生桥只使用当前 Python 头文件和 C++ 编译器，不需要 pybind11 或 libtorch 开发依赖。`--steps`、`--warmup`、`--repeats`、`--episode-ticks`、`--seed` 和 `--device` 控制测量条件。工具只比较当前 CPU/GPU 实现。

报告包括硬件、OpenMP 亲和设置、代码哈希、编译参数、初态/动作/末态指纹、计数、逐步延迟和 CPU/GPU 加速比。profile 报告中的 `summary` 始终来自**无插桩基准**；`profile` 字段是诊断采集结果，不能拿来比较吞吐。

### Profile 的边界与解释

| 后端 | 方法 | 主要输出 |
|---|---|---|
| CPU | 单独编译 profiler，直接包含当前 `backends/cpu/step.cpp`；默认按 tick 轮转采样 1/32 的环境 | native 区间、worker span、缓存准备、增长、行动/淘汰、输出/计数、自动重置、缓存回写 |
| GPU | PyTorch/CUPTI 捕获真实 CUDA step；每个 tick 一个 CPU annotation | 实际 kernel 时长、CUDA API、传输、kernel 外时间、逐 tick Chrome trace |

CPU 标记在普通构建中编译为空，不需要复制规则或字符串替换源码；采样计数按 cache line 隔离，避免 profiler 自己引入 worker 之间的伪共享。`--cpu-sample-stride` 可调整采样密度；逐阶段计时是采样 worker 时间，不能当成整批墙钟时间相加。阶段时钟会放大小阶段的占比；native 与最长 worker 的差值包含 OpenMP 进入/汇合和调度，不能简单称为纯 barrier 开销。另直接记录进入并行区到首个 worker、worker 开始时间跨度和最后 worker 完成后的汇合尾部。profile 提供与无插桩基准的耗时比，明确扰动程度。

GPU 先在测量 marker 外预热 profiler 回调、恢复初态，再采集逐步调用。初始化、reset 和读回事件不会混入被测 tick 的汇总。trace 保存为输出 JSON 同目录的 `*_25x25_p4_b10000.cuda.trace.json` 等文件，可用 Perfetto 打开。GPU annotation 不是第二次 step；kernel 时间取区间并集，避免重复计算。CUDA launch/synchronize 与 GPU 执行重叠，不能相加；kernel 外时间还包含主机工作、空闲间隔和 profiler 开销。

两种 profile 都按真实轨迹将 tick 归类，优先级为：有任一环境 reset、首次冷启动、任一环境全图增长（50 tick）、任一环境结构增长（2 tick）、仅行动。类别互斥，保存各类步数/平均/P50/P95及总耗时占比。CPU 数值分析可结合逐步 native 行数据；GPU 可同时查看每类 kernel 时间。

本机 `perf_event_paranoid=4`，无法使用 perf 硬件计数器。当前 profile 能区分主机提交/完成等待、真实设备工作和规则阶段，**不能证明**内存带宽饱和、cache miss 或具体 warp stall；trace 中寄存器/occupancy 信息也不是实测硬件利用率。

基准代码分为配置、工作负载准备、计时和 profiling。`native.cpp` 只有动作采样及实际 CPU 规则源码的 profiling 包装，没有第二份模拟实现。

<a id="results"></a>
## 8. 当前性能验证

清理前后的对照使用相同冻结地图和 1000 tick 动作轨迹，25×25、4 玩家、seed=42、每 500 tick 自动重置。只测每 tick 的 `env.step(action)` 和完成同步，排除决策、初始化及预热。CPU 使用 26 线程，GPU 使用默认自适应配置；硬件为 i7-14700KF / RTX 4090。

清理前后各重复 5 轮，在 1k、10k、100k 三个规模下，CPU/GPU 的最终状态、统计、动作诊断、done/errors、计数器和 RNG 指纹全部与冻结参考轨迹一致。原始样本保存在本地 `outputs/cleanup/20261003/{cpu,gpu}-comparison.json`。

本轮测量期间另有训练进程持续占用 GPU 和一个 CPU 核，GPU 延迟出现明显的阶段性波动。因此这批计时数据只存档，**不据此认定提速或证明没有性能回退**；空闲设备上的性能复测仍待完成。当前保留此前选定的增量统计、增长索引、原生启动、自适应线程和压缩元数据优化。1k/10k/100k 配置的 CUDA 寄存器数仍为 40/40/48，理论驻留块数仍为 12/9/5；共享内存占用从 2464/9856/4928 降至 2368/9472/4736 字节。

这些结果适用于该工作负载；不同地图、玩家数、动作分布和机器负载会改变吞吐。线程配置和压缩元数据阈值可以显式覆盖，便于在其他设备复测。

<a id="verification"></a>
## 9. 测试与本地归档

```bash
conda run -n generals-rl python -m unittest discover -s env/tests -v
```

GUI 测试需要图形显示；CUDA 测试需要真实设备和编译器。本轮 163 项环境测试（含 Linux GUI 和 CUDA）通过；独立 wheel 的 CPU/GPU 启动、两种 profile 入口以及 CUDA memcheck/racecheck/synccheck/initcheck 均通过。规则覆盖独立 Python/CPU/CUDA 逐 tick 一致性、0/1 兵转移、半兵、按座位继承与淘汰、增长周期、迷雾、动作掩码、重置、冻结、缓存失效、非默认流和打包安装。GPU 检查 int64 有效范围内的精度；CPU 另外检查溢出错误语义。

本次清理前的源码、测试及完整文档保存在本地 `outputs/cleanup/20261003/before/`，哈希清单位于同目录 `before.json`。原 GPUv2 存档为 `outputs/archive/gpuv2/20261003/before.tar.gz`。本地存档和原始测量不打包、不提交；本文保存现行接口与验证结论。恢复历史实现时应解压到独立目录。

跨算法与模型版本的统一评测见 [Arena 使用说明](../arena/README.md)。
