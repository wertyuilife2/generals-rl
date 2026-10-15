# Generals baselines

独立的 C++ 手写策略包 `generals_baselines`，Python 负责控制器接口与 NumPy 随机数状态，
依赖游戏环境，不依赖 Torch、训练或 GUI。需要 Linux/macOS、C++17 编译器（可用 `CXX` 指定）
以及 Python 开发头文件；NumPy 头文件随其 Python 包提供。
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
策略只接收该玩家可见的观测。人类输入和混合人类/AI 工厂仍在 `generals.controllers`；
该工厂在需要 AI 时加载本包。

四种策略的记忆、候选评估、最近目标选择、BFS、路径生命周期、半兵规则和随机动作评分都在
`src/generals_baselines/policy.cpp` 中实现；`base.py` 管理生命周期，`host.cpp` 直接绑定 NumPy
数组并调用原有策略 C ABI，不保留另一套规划器。绑定仍校验类型、形状和可变规划参数，
非连续观测按需复制。构建缓存包含 Python/NumPy 版本和 ABI。
每个控制器独立持有原生状态，仅接收允许的观测。调试用的 `terrain/owner/army/path` 返回独立快照。

C++ 在持有 NumPy BitGenerator 锁时直接调用其原生随机函数，保持原有种子派生、洗牌和抽样顺序，
不预取随机数，不跨席位共享随机流。优化前后的完整动作与逐 tick 状态通过黄金轨迹及 1000 局对照校验。
策略 CPP 源码也参与 Arena 的算法版本哈希；替换实现不会被当成旧版本继续评测。

```bash
PYTHONPATH=env/src:baselines/src python -m unittest discover -s baselines/tests -v
python -m pytest baselines/tests -q
```
