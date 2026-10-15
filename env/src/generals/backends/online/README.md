# 官方线上 backend

`OnlineBackend` 将官方 Bot 站或主站的玩家视角转换成现有 `Observation`，接收现有
`Action`。C++ 策略继续由 Arena 的 `NativeController` 调用，保持原有 C ABI 和动作编码。

安装可选依赖（项目根目录）：

```bash
.venv/bin/python -m pip install -e 'env[online]'
```

线上连接使用 HTTPS / WebSocket 和 Socket.IO 5.x（Engine.IO 4）。2026-10-14
对 `https://botws.generals.io` 的无账号握手验证：EIO=3 返回 unsupported protocol，
EIO=4 成功。官方开发文档里的 EIO=3 示例已经过时。

## 选择 Bot 站或主站

`online` 和 `online-account` 均支持 `--site bot|main`：

| 站点 | Socket.IO 地址 | 房间与回放网站 |
| --- | --- | --- |
| `bot`（默认） | `https://botws.generals.io` | `https://bot.generals.io` |
| `main` | `https://ws.generals.io` | `https://generals.io` |

主站使用独立账号配置，保存在项目内的 `config/secret/razor-main.json`。例如：

```bash
.venv/bin/generals_arena online-account --site main \
  --account razor-main --username 'GPT6 Razor Bot'

.venv/bin/generals_arena online @gpt-6-razor-v3 --site main \
  --account razor-main --room razor-main-test \
  --record outputs/online/razor-main-test-001.jsonl
```

自定义房间地址为 `https://generals.io/games/razor-main-test`。账号注册命令只注册；
第二条命令才会加入房间。已保存的账号会自动选择其绑定服务器，因此后续可省略
`--site`。现有 `razor` Bot 站账号不能配合 `--site main` 使用，也不会被覆盖或迁移。

需要公开队列时，显式将 `--room ...` 替换为 `--queue 1v1` 或 `--queue ffa`；
连续比赛同样使用 `--games N --output 新目录`。主站的 1v1 入队沿用官方客户端默认值，
允许匹配 Bot。所有站点共用策略、观测和动作接口；结果中的回放 URL 指向对应站点。

使用环境变量身份时，主站默认读取 `GENERALS_MAIN_USER_ID`，Bot 站默认读取
`GENERALS_BOT_USER_ID`，不会跨站回退。`--user-id-env NAME` 可显式选择变量。
`--server` 是自定义 Socket.IO 地址入口，与 `--site` 互斥；直接指定已知主站地址也会
使用主站协议。未知自定义服务器不推测回放网站，`replay_url` 返回 `null`。

主站禁止用户名以 `[Bot]` 开头，因此使用 `GPT6 Razor Bot` 明确标识自动策略；
Bot 站仍要求 `[Bot]` 前缀。两站分别校验，不共用名称限制。

主站参数依据 [官方公开客户端 v31.4.3](https://generals.io/generals-main-prod-v31.4.3-485dbf55.js)：
注册及入队附带公开客户端字段，`gio_error` 等服务器拒绝消息会终止会话。
2026-10-15 已完成主站无身份 Socket.IO 握手、离线接口验证及独立账号注册。
主站注册实测拒绝 `[Bot]` 前缀，使用 `GPT6 Razor Bot` 成功；注册失败时仍保留原身份凭据。
随后使用冻结的 Razor v3 完成 10 场主站 1v1（5 胜 5 负），无连接失败、策略超时或错误。
记录位于项目 `outputs/online/razor-main-10-20261015T0516Z/report.json`，不影响本地 Arena 评分。

## 运行 C++ 策略

推荐先自动建立专用 Bot 账号（只注册账号，不排队、不开始比赛）：

```bash
.venv/bin/generals_arena online-account \
  --account razor --username '[Bot] GPT6 Razor'
```

从项目根目录运行时，凭据自动生成并写入 `config/secret/razor.json`，目录权限 0700、
文件权限 0600。整个 `config/secret/` 已由项目 `.gitignore` 排除，不提交到 Git。
命令只输出公开元数据，不输出 `user_id`。成功后重复执行直接复用；
注册失败或连接中断会保留同一份凭据，不自动换账号。若回执超时，注册状态可能不确定，
应保留该文件进行恢复，不能删除后靠生成新身份重试。

使用保存的账号：

```bash
.venv/bin/generals_arena online @gpt-6-razor-v1 --account razor --room razor-test
```

`--account-dir` 可以指定其他私有凭据目录。保存账号绑定注册服务器，不能用
`--server` 将同一凭据转发到另一站点。`--account` 与 `--user-id-env` 互斥。

也可以继续使用环境变量凭据：

预先将**专用 Bot 账号**的秘密 `user_id` 放入 `GENERALS_BOT_USER_ID` 环境变量。
它不是公开用户名，不要提交到 Git 或通过命令行参数传入。

```bash
.venv/bin/generals_arena online heristic/gpt-6-razor/policy.cpp \
  --params heristic/gpt-6-razor/params.json \
  --room razor-test \
  --record outputs/online/razor-test-001.jsonl
```

打开 `https://bot.generals.io/games/razor-test` 加入同一房间。
仅在需要强制开始自定义房间时添加 `--force-start`。
首次注册新 Bot 名称可以添加 `--username '[Bot] Razor'`；名称必须唯一，后续运行
应省略该参数。客户端会等待 `error_set_username` 返回成功后再加入房间。

还支持现有 `policy.zip`、`@gpt-6-razor-v1`、`policy/<完整版本哈希>`。保存版本只读
已有工件；源码输入使用临时编译工件。上述命令不创建正式评测任务、不更新本地榜单。

联调完成后可显式选择公开队列：

```bash
.venv/bin/generals_arena online @gpt-6-razor-v1 --queue 1v1
```

连续采样 100 局并保存逐局记录（输出目录必须不存在）：

```bash
.venv/bin/generals_arena online @gpt-6-razor-v1 --account razor --queue 1v1 \
  --games 100 --output outputs/online/razor-100
```

`report.json` 在每局开始和结束时更新。网络错误和排队超时单独保留，不计作完成比赛；
间隔 30 秒后尝试新会话，连续三次网络失败后停止。策略或协议错误会保留失败记录并
立即停止后续比赛。比赛按顺序进行，不并发使用账号，不重发中断局内的动作。
该报告仅反映线上样本，不是本地 Arena 全量评测成绩。

需要在迭代策略时安全停队，可在本次输出目录中创建 `stop-after-game` 空文件。
运行器会打完当前对局，在下一局开始前保存 `status=paused` 并退出；已完成局数和
结果保留。随后用新输出目录运行新版本，并按完成数计算剩余局数。

`--queue ffa` 为 FFA。默认自定义队列必须指定 `--room`，不会意外进入公开匹配。
默认每次命令运行一局。超时、断线或协议错误会结束本次连接，重新运行开始新的会话。
`--start-timeout` 默认 120 秒，`--update-timeout` 默认 15 秒。

## Python 接口与抽象边界

```python
import os
from generals import make_env
from generals.runtime.match import run_match

backend = make_env(
    mode="online",
    user_id=os.environ["GENERALS_BOT_USER_ID"],
    room="razor-test",
)
result = run_match(backend, my_controller, seed=42)
```

主站使用 `make_env(mode="online", site="main", user_id=主站凭据, room="razor-main-test")`。
Python 接口的 `site` 与 `server` 同样互斥。

`my_controller` 使用原来的 `reset / act / on_result` 协议；线上 `on_result` 不会被
调用，因为公开协议没有与本地 `MoveResult` 等价的权威回执。需要精确回执的控制器
应声明 `requires_move_results = True`，运行器会拒绝不兼容的 backend。
C++ 的 `previous` 保持空指针，Razor 当前不依赖此字段。

相同运行器也可以使用本地 CPU 后端：

```python
from generals import MapConfig, make_env
from generals.runtime.match import run_match

result = run_match(
    make_env(MapConfig(25, 25, 2), mode="match", seed=42),
    {0: controller_a, 1: controller_b},
)
```

两个后端实现 `generals.protocol.match.MatchBackend`：

- `start() -> MatchFrame`：获得第一份观测。
- `step({seat: Action}, frame_id=...) -> MatchFrame`：提交受控席位动作并等下一份观测。
- `close()`：释放资源；线上会离开比赛或取消排队。

本地控制所有席位，线上只控制服务器分配的一个席位。`MatchFrame` 包含观测映射、
修订号、可获得的结果和结束信息，不暴露隐藏 `GameState`。原 CPU/GPU `VectorBackend`
及原正式联赛运行器保持原样。

## 时序与信息约定

- 每个 map/city diff 均按接收顺序应用，再使用最新观测。策略计算期间若收到新帧，
  旧动作不会发送，`StaleFrameError.frame` 提供新帧；`run_match` 自动重新决策。
- 每次决策先发 `clear_moves`，移动再发 `attack`。WAIT 也清除遗留队列。
  这不代表动作已执行，也不能消除网络延迟期间服务器继续推进的竞态。
- `tick` 默认保留服务器 `turn` 的半回合计数。一次 `step` 可能跨多个 tick。
  `--tick-offset` 仅用于显式相位校准，不能补造中间棋盘或保证本地、官方规则完全相同。
- 只转换服务器给本席位的信息。普通迷雾 `-3` 的地形为 `255`；迷雾障碍 `-4` 的
  地形为 `Terrain.FOG_OBSTACLE = 2`，表示尚未辨明的山或城市，不等同于已知山。
  两者均保持 `visible=False`、兵力 `-1`、归属 `-3`、结构 `255`。C++ 通过现有
  `terrain` 数组读取值 2，无需改变 ABI 布局；Razor v2 据此规划探路，见到实物后
  再使用实际地形和城市信息。敌方城市总数为 `-1`，本方城市数从可见己方城市统计。
- 目前支持至多 4096 格、2–8 人、无组队的标准规则。已识别的额外地形、修饰器和
  超出人数范围的比赛会明确拒绝；它们需要扩展协议及策略能力后再支持。
- 接受旧版两层地图及新版三层地图；按当前测试约定忽略第三层 `tunnelLimits`，
  仅保留其原始值用于后续增量更新对齐，不向策略表达隧道规则。
- `game_lost` 表示本席位结束，`winner_id=None`；击杀者不等于整场 FFA 冠军。
- Socket.IO 心跳和接收线程只做网络 I/O、事件入队；策略在运行器线程串行调用。
  C++ 原生 10ms watchdog 保留，网络等待不计入决策耗时。网络/协议错误不会冒充决策超时。
- 每个 backend 只连接一次，断线不自动重连，不重发不确定是否已经执行的动作。
  接收队列溢出会终止会话，不丢弃 diff 后继续使用损坏的状态。

## 记录与验证

`--record` 用独占新建方式写 JSONL，包含接收到的比赛消息及发送的动作，不记录
身份、注册或加入队列消息。其格式不是本地全状态回放；最终对局以官方 replay ID 为准。
退出时输出 JSON，包括决策计时、过期决策数、结束原因和官方回放 URL。

离线测试：

```bash
.venv/bin/python -m pytest env/tests/test_online.py arena/tests/test_online_native.py arena/tests/test_online_accounts.py -q
```

测试不创建账号或参加公开匹配。上线前仍需在自定义房间验证实际游戏事件、增长相位
和策略表现；传输握手成功不等于已经完成实战兼容性验证。

参考：[官方 API](https://dev.generals.io/api)、
[Python Socket.IO 客户端](https://python-socketio.readthedocs.io/en/stable/client.html)。
