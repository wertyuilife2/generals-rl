# generals-env

CPU 上的最简 Generals 游戏：2–8 方自由混战，一个人类对抗 1–7 个 AI，或全 AI 观战。GUI 和无界面运行共用同一套规则；所有运算在 CPU 上完成。

## 依赖与安装

Python >=3.11、NumPy >=1.24；桌面游玩额外要求 Python 的 Tk 支持及图形显示环境。已提供的 conda 环境包含这些组件，原型不使用 PyTorch/CUDA。

从仓库根目录运行：

```console
conda run -n generals-rl python -m pip install -e ./env --no-deps --no-build-isolation
```

`--no-deps --no-build-isolation` 适用于依赖及 setuptools 已备齐的本机环境。全新 Python 环境可使用 `python -m pip install -e ./env`，并通过该 Python 发行版／操作系统安装 Tk。`python -m tkinter` 可以检查窗口能否创建。

## 图形游玩

```console
python -m generals_env play --players 4 --seed 42
python -m generals_env play --players 3 --human-seat 1 --ais aggressive,defensive
python -m generals_env watch --players 8 --size 35 --seed 7 --speed 10
```

在 `generals-rl` 环境内执行上面的命令；未激活环境时使用 `conda run -n generals-rl` 前缀。

启动后首先显示地图预览。可设置人数、席位策略、人类座位、地图尺寸、密度、种子和视野模式，再点“开始”。人类局部视野预览不会显示未知主城。游戏进行期间地图设置锁定，暂停允许规划队列和完整 tick 单步。

| 鼠标／按键 | 操作 |
|---|---|
| 左键己方格 | 选中军队来源 |
| 左键规划光标相邻格／WASD／方向键 | 追加一步移动或攻击，光标前移 |
| 重复点击规划光标／Z／半兵按钮 | 下一条入队动作为半兵，入队后恢复普通 |
| Shift+左键／取消选择后重新点击 | 强制换到另一己方来源，并清空旧队列 |
| Q／清空队列 | 取消所有尚未执行的动作 |
| E／撤销末步 | 撤销最后一个尚未执行的动作 |
| 空格／取消选择 | 取消选择，已经排队的动作继续执行 |
| H／选择主城 | 选择自己的主城 |
| 中键或右键拖动 | 平移地图 |
| 滚轮／+／− | 缩放地图 |
| Home／适应窗口 | 居中并适配地图 |
| P／暂停 | 暂停或继续 |
| 单步 | 暂停时推进全体玩家的一整个 tick |

所有基本操作都有鼠标入口，键盘只是快捷方式。路径最多排队 64 步；只能逐格输入，不自动走向远处点击目标。来源仅有 0／1 兵时，指令照常执行并转移 0 兵，不等待增兵；转移到己方格时正常推进队列和选中位置。0 兵不能占领空地或敌方格，未攻占时按原规则清除后续路径；来源失守时同样清除失效路径。

“同图重开”恢复相同地图和种子；“新地图”生成并展示新种子。人类淘汰后可选择继续观战。焦点离开游戏窗口时，人类模式自动暂停。缺少中文字体时界面使用英文标签。

Windows 和 Linux 使用同一份源码。鼠标滚轮事件按 Tk 窗口系统适配。Linux 必须有可用的图形显示会话；纯 SSH／无显示环境使用 `simulate`。本次 Linux/X11 已验证，Windows 和跨系统 DPI 仍需实机验收。

## 无界面模拟

```console
python -m generals_env simulate --players 4 --size 25 --games 100 --seed 42 --max-ticks 20000
python -m generals_env simulate --players 2 --ais aggressive,random --visibility full --max-ticks 2000
```

循环没有真实时间等待、不加载 Tk、不创建画面。输出 JSON，包括自然胜者、截断数量、各座位胜场、实际 tick 数和吞吐。固定座位按编号行动，比较策略时应交换座位。

通用参数：

- `--players`：2–8 的任意整数。
- `--size`：15／25／35，默认 25；程序接口及 `--width`／`--height` 也支持其他合法尺寸。
- `--mountain-density`：0–0.4；`--city-density`：0–0.2。
- `--seed`：非负整数。同种子、配置和软件版本可复现。
- `--ais`：按 AI 席位排列的逗号分隔名称：`aggressive,expansion,defensive,random`。人类模式不在该列表里填写人类。
- `--visibility local`：默认。AI 与人类使用各自领土及八邻域观测；观战 GUI 仍可显示全图。
- `--visibility full`：明确向所有控制器开放全图，适合调试。
- `--max-ticks`：模拟模式默认 20000；达到上限是截断，不按兵力判胜。
- `--debug`：模拟模式启用 tick 边界的状态不变量检查。
- `--output path.json`：将统计写入新文件；不覆盖已有文件。

## 动作记录和校验重放

```console
python -m generals_env simulate --players 4 --seed 42 --max-ticks 3000 --record outputs/match.jsonl
python -m generals_env replay outputs/match.jsonl
```

记录包含初始数组、规则版本、配置、每个玩家的实际动作及逐 tick 状态摘要。重放直接执行动作，不重新运行 AI，并校验每步结果。`--record` 要求单局且目标文件不存在；默认不开启，以免影响高速模拟。此功能提供程序级验证，不包含录像播放器。

## 规则摘要

- 起始主城 1 兵，中立城市 40 兵；山脉不可进入。
- 每 tick 先增兵，再按玩家 0…P−1 依次行动。后行动者看见此前的结果。
- 主城／已占领城市每 2 tick +1；已占领普通地块每 50 tick +1。第 50 tick 城市没有额外领土增兵。
- 只能四邻接移动；普通派出 `max(0, n-1)`，半兵派出 `n//2`。1 兵指令转移 0 兵，照常消耗一个行动时隙，不等待增兵。
- 攻守按 1∶1 抵消；只有进攻严格大于守军才占领。打平留下 0 兵但不改变归属。
- 主城被占则立即淘汰，主城变城市；其余领土转交，驻军 `max(1, n//2)`，直接攻下的主城不再减半。
- 最后一方自然获胜；已结束的状态禁止继续推进。

## Python 接口

```python
from generals_env import MapConfig, generate_map
from generals_env.controllers import make_controllers
from generals_env.runner import Runner

config = MapConfig(width=25, height=25, players=4)
state = generate_map(config, seed=42)
controllers = make_controllers(["aggressive", "expansion", "defensive", "random"], seed=42)
runner = Runner(state, controllers, seed=42, visibility="local", max_ticks=20000)
summary = runner.run()
```

`Runner` 不会隐式重置传入控制器。新的对局创建新的状态与控制器；GUI 的同图重开从保存的初始状态复制。自定义控制器实现 `reset(player_id, seed)`、`act(observation)`、`on_result(result)` 协议。

直接提交动作可使用：

```python
from generals_env import Action, CoreEngine, Direction

engine = CoreEngine(generate_map(MapConfig(players=2), seed=42))
source = int(engine.state.general_pos[0])
result = engine.step([Action.move(source, Direction.RIGHT), Action.wait()])
```

`step` 接受完整玩家槽位序列，已淘汰者填 WAIT。本 tick 中途被淘汰者的预提交动作自动跳过。开局只有 1 兵，示例在目标可通行时会执行一次零兵转移，不改变兵力或归属；越界、山脉等无效动作也消耗该玩家的时隙。

需要严格复现内置 AI 的信息时刻时，使用 Runner，或按照 `begin_tick()` → `next_player`／`observe()`／`apply_action()` → `finish_tick()` 调用。预先选定一组动作并不表示各策略获得了与逐时隙决策相同的信息。

`observe(state, player_id, mode)` 和 `snapshot(state, player_id=None)` 返回独立只读快照。`player_id=None` 明确表示全图观战权限。`stats.army`、`stats.territory` 与存活状态公开；LOCAL 下 `stats.cities` 仅自身为真实值，其他玩家为 `-1`，FULL 返回全体真实值。`legal_action_mask(obs)` 返回 `[H*W,4,2]` 掩码和 WAIT 标志，预计打不赢不构成非法动作。

## 路径与特征接口

`generals_env.pathfinding` 提供 `neighbors`、`single_source_distances`、`shortest_path`、`DistanceCache`、`prepare_map`、`all_pairs_distances`。

距离是四邻接非山脉图上的移动边数，不包含留守、攻城代价或动态威胁。全点对距离需显式调用，默认限制输出内存为 128 MiB。缓存按地图尺寸、地形内容、未知格策略和来源区分，并有容量上限。

LOCAL 策略只能把观测／已知地形交给路径工具，不能使用真值地图的最短路暗中获知山脉。`prepare_map(state.layout)` 属于全信息调试／离线预处理入口。

## 测试与性能

```console
python -m unittest discover -s env/tests -v
python env/benchmarks/benchmark_cpu.py --size 25 --players 4 --ticks 2000
python env/benchmarks/benchmark_cpu.py --size 35 --players 8 --ticks 2000 --memory
```

以上从仓库根目录执行。GUI 测试在没有显示服务时跳过，并明确报告。`--memory` 使用标准库 tracemalloc 测量 Python 分配的峰值，会降低性能，不能与关闭追踪的吞吐直接比较。详见仓库 `doc/implementation_report.md`。
