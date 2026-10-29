# 官方线上接入

`OnlineBackend` 将服务器消息转换为 env 的 `Observation`，接受同一套 `Action`。
它不在本地重新结算游戏，也不改变 CPU/CUDA 仿真后端。
Arena 使用现有策略注册表加载 C++ 策略或通用策略包；RL 的特征和推理仍由包内代码负责。

## 安装和使用

在仓库根目录、`generals-rl` conda 环境中：

```bash
python -m pip install -e 'env[online]' -e arena
# 显式注册专用 Bot 账号；此命令不加入比赛。
generals-arena online-account --account razor --username '[Bot] Razor Example'
# 加入指定自定义房间。参数文件与源码共同确定策略版本。
generals-arena online heristic/gpt-6-razor/v7 \
  --params heristic/gpt-6-razor/v7/params.json \
  --account razor --room razor-test --output outputs/online/razor-test
```

策略参数也可为通用策略包目录、`policy/<版本哈希>` 或 `@联赛别名`；后两种配合
`--league 目录` 使用。策略包能力声明必须覆盖服务器实际的地图尺寸和玩家数。
内置路径规划 baseline 依赖精确移动回执，暂不支持线上使用；`baseline/wait` 可用于联调。

默认 Bot 站。`--site main` 选择主站，并需要独立身份（主站用户名不能以 `[Bot]` 开头）。
保存的账号绑定服务器，不能跨站复用。也可使用环境变量凭据：默认分别读取
`GENERALS_BOT_USER_ID` / `GENERALS_MAIN_USER_ID`，或指定 `--user-id-env NAME`。

账号写入 `config/secret/<账号>.json`（Git 忽略，目录 0700、文件 0600）；
`--account-dir` 可另选路径。注册失败保留同一身份以便重试，输出不含身份密钥。

公开队列需显式选择 `--queue 1v1` 或 `--queue ffa`，此时不传 `--room`。
`--games N --output 新目录` 顺序执行多局；报告在每次尝试前后保存，失败会停止队列，
不会把连接错误当作战败，也不会自动加入另一局。创建输出目录下的 `stop-after-game`
文件可在本局结束后停止。线上结果不写入离线 Elo 池。

## 接口和规则边界

```python
from generals.backends.online import OnlineBackend
from generals.runtime.online import run_online

backend = OnlineBackend(user_id=private_user_id, room="test")
result = run_online(backend, lambda first_observation: my_controller, seed=42)
```

运行器负责连接和控制器的生命周期。策略需实现 `reset(player_id, seed)` 和 `act(obs)`。
服务器没有等价于本地 `MoveResult` 的权威回执，因此不调用 `on_result`；声明
`requires_move_results=True` 的控制器会被拒绝。C++ `previous` 为空。

- 所有增量消息依次解码。决策期间收到新帧时，丢弃旧动作并重新决策。
- 每次操作清空遗留移动队列后发送动作；WAIT 或单兵格移动只清空队列。
  网络 `step` 可能跨多个 tick，不等价于离线每次推进一个 tick。
- 迷雾中的山/城混淆信息通过 `fog_obstacles` 暴露；未知地形仍为 255。
  传递公开 `has_kill`、`warn_afk` 和 `neutralize_in`，敌方城市总数未知时为 -1。
- 支持普通 2–8 人、至多 4096 格；已识别的组队、特殊地形和修饰器会被拒绝。
  接受两层和三层地图编码；无隧道的标准局不会使用第三层 tunnelLimits。
- 出局不等于整局结束：`game_lost` 结束当前席位，FFA 不据此猜测冠军。
- 接收队列溢出、断线或协议错误会停止会话，不丢弃增量数据继续游戏。
- `report.json` 保存结果及官方回放链接；逐局 JSONL 是服务器消息记录，
  不是可用于本地全信息回放的文件。身份和入队消息不写入记录。

主站公开客户端字段按本仓库已核查的 31.4.5-e5c53eb3（2026-10-28）更新。
本次集成使用模拟消息测试，没有注册真实账号或参加线上对局；正式使用前应先用自定义房间联调。

```bash
python -m pytest env/tests/test_online.py arena/tests/test_online_accounts.py arena/tests/test_online_packages.py -q
```
