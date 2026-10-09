# Generals baselines

独立的 NumPy 手写策略包 `generals_baselines`，依赖游戏环境，不依赖 Torch、训练或 GUI。

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

```bash
PYTHONPATH=env/src:baselines/src python -m unittest discover -s baselines/tests -v
```
