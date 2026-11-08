# Generals baselines

独立的 C++ 手写策略包 `generals_baselines`，Python 负责控制器接口与 NumPy 随机数状态，
依赖游戏环境，不依赖 Torch、训练或 GUI。需要 Linux/macOS 和 C++17 编译器（可用 `CXX` 指定）；
首次创建控制器时编译，之后按源码与编译参数哈希复用本地缓存。

```bash
python -m pip install -e ./env -e ./baselines
python -m generals simulate --players 2 --ais aggressive,random --games 2 --max-ticks 2000
```

`AggressiveAI`、`ExpansionAI`、`DefensiveAI` 和 `RandomAI` 从原环境控制器迁移，
策略行为及种子派生保持不变。`random` 是带扩张/进攻偏好的随机启发式，
与 `generals_rl.opponents.RandomOpponent` 的合法动作均匀采样不同。

```python
from generals_baselines import create_controller, make_controllers

policy = create_controller("aggressive", player_id=0, seed=42)
players = make_controllers(["expansion", "defensive"], seed=42)
```

公共生命周期为 `generals.protocol.controller.Controller`：
`reset(player_id, seed)`、`act(observation) -> Action`、`on_result(move_result)`。
策略只接收该玩家可见的观测。`generals.controllers` 提供无界面模拟的控制器工厂，
在创建 AI 时加载本包；人类输入由 Arena 网页界面处理。

四种策略的记忆、候选评估、最近目标选择、BFS、路径生命周期、半兵规则和随机动作评分都在
`src/generals_baselines/policy.cpp` 中实现；`base.py` 是生命周期与数组绑定，不保留另一套 Python 规划器。
每个控制器独立持有原生状态，仅接收允许的观测。调试用的 `terrain/owner/army/path` 返回独立快照。

每个控制器都实现通用 `reset_batch(player_ids, seeds)`、`act_batch(raw)`、`on_result_batch(results)` 接口，
支持 2–8 人。`raw` 直接来自 env 的公开批量观测；策略不接收完整环境状态，也不再自己生成迷雾。
一次原生调用串行处理所有 lane，每局的规划记忆和随机流独立，返回借用的 `[B] int64` 动作数组。
Arena 按与其他策略相同的能力检测、缓存和回执路径执行，无 baseline 专用批执行分支。

```python
policy = create_controller("aggressive")
policy.reset_batch([0, 1], [42, 43])  # 最终席位种子，不再次派生
actions = policy.act_batch(env.observe([0, 1]))
```

需要回执时，`on_result_batch` 接收本组席位的 `moves[B,9]`、`actions[B]`、`active[B]` 和 `player_ids[B]`。
`batch_device="cpu"` 声明原始数组在 CPU 上，Arena 会整批转换；不增加策略线程。
`requires_move_results=True` 声明对精确移动回执的依赖，线上运行器按此能力拒绝不兼容接入。

C++ 在持有 NumPy BitGenerator 锁时直接调用其原生随机函数，保持原有种子派生、洗牌和抽样顺序，
不预取随机数，不跨席位共享随机流。优化前后的完整动作与逐 tick 状态通过黄金轨迹及 1000 局对照校验。
策略 CPP 源码也参与 Arena 的算法版本哈希；替换实现不会被当成旧版本继续评测。

```bash
PYTHONPATH=env/src:baselines/src python -m unittest discover -s baselines/tests -v
python -m pytest baselines/tests -q
```
