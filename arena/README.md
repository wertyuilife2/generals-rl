# Generals Arena

统一比赛管理：固定循环赛、持续联赛和训练评估共用策略准入、比赛执行、结果、回放和评分基础。
持续联赛登记策略后生成增量对局，分别计算 1v1 GoR 和 FFA Codeforces 评分；浏览器提供分池榜单、对局历史、回放和人类排位。
Arena 接收通用策略包；神经网络的特征、动作映射及推理由包内代码负责。
公开联赛使用逐局 GoR／Codeforces 更新；1v1 训练评估使用 GoR 胜率模型的整批拟合，支持固定校准参考。不同评分池的分数不可直接混用。

当前游戏规则为 2026-10-28 核查的官网 **31.4.5 普通 1v1 / FFA**，由 env 提供。所有模式共用行动优先级、增长、迷雾、离场和 FFA 结算；证据与边界验证见[实施记录](../doc/official_rules_alignment.md)。旧规则联赛、策略包和回放需要对应归档源码，不能沿用原分数追加新规则比赛。

C++ 启发式使用 [统一接入接口](RESEARCH.md)：`generals-arena commit 源码 --name 别名`
自动编译并完成 1000 局正式评测；`generals-arena run --player ... --player ...` 自由测试。
Arena 管理源码版本、10ms 限时、评分和录像。可直接使用
[60 分钟研究提示词](AUTORESEARCH_PROMPT.txt)，其中最多三次正式评测是该研究任务的约束。

## 快速开始

在仓库根目录运行（已有 `.venv` 时先 `source .venv/bin/activate`）：

需要 Linux/macOS、C++17 编译器和 OpenMP。macOS 使用 clang 与已安装的 libomp，
非标准位置可设置 `GENERALS_LIBOMP_PREFIX`；编译器可用 `CXX` 指定。首次运行自动编译并缓存。

```bash
python -m pip install -e ./env -e ./baselines -e ./arena
generals-arena league init outputs/arena/main --config arena/configs/league.json
generals-arena serve outputs/arena/main --port 8086 --no-worker
```

打开 **http://localhost:8086**（默认端口也是 8086）。上面的命令创建新联赛；初始尚无比赛记录，下面介绍如何登记策略。
`outputs/arena/native_1000` 是历史 1000 局联赛；旧比赛需使用对应归档源码查看或继续运行。
榜单可切换独立的 1v1/FFA 评分池；通过“对局记录”按人数筛选、
查看回放；通过“人类参赛”创建本地玩家并开始排位，人数可随机或指定。前端直接由 Python 服务提供，无需 Node、npm 或构建。
不能直接打开 `web/index.html`，它需要同源 API。

`--no-worker` 让正式网页与独立的研究评测共用联赛；榜单、录像和人类对局仍正常工作。
不带该参数的 `serve` 会处理队列，所以可以在另一个终端继续 `league add ... --queue-only`。
不加 `--queue-only` 时，`add` 登记后会直接运行待评测队列。完整的 1000 局可能耗时较长，
可先用 `--games 10` 检查接入；实际结果取决于策略、地图和时限，少量测试局不能代表稳定实力。

新联赛只需初始化一次，目标目录必须不存在。在上面创建的联赛中继续登记和评测：

```bash
generals-arena league add outputs/arena/main --entrant arena/configs/aggressive.json --queue-only
generals-arena league add outputs/arena/main --entrant arena/configs/expansion.json --queue-only
generals-arena league add outputs/arena/main --entrant arena/configs/defensive.json --queue-only

# 仅运行剩余任务；本次最多完成 20 个新对局，然后正常退出
generals-arena league run outputs/arena/main --limit 20
# 继续剩余对局（无需重新登记）
generals-arena league run outputs/arena/main
# 修正失败的策略运行问题后重试
generals-arena league run outputs/arena/main --retry

generals-arena league leaderboard outputs/arena/main
generals-arena league leaderboard outputs/arena/main --json
generals-arena league rebuild outputs/arena/main
```

同一版本重复登记是幂等操作，不会重复增加 1000 局；不同版本须使用新名字。
`--games` 是**该新选手任务的总局数**。2 人局更新 `1v1` 池，3–8 人局更新 `ffa` 池，两个池的分数和统计独立。
需要专测某人数时，初始化配置中只保留一个 `player_counts`，如 `[4]`。
AI 可以重复入场，即使只登记一个 baseline 也能生成八人局。每个席位拥有独立控制器和随机种子。
同一算法出现多次时，对局数只加 1，出场数按实例数增加。

## 策略退役、缓存与线上接入

```bash
generals-arena league retire outputs/arena/main @旧策略 --reason '由新版本替代'
generals-arena league leaderboard outputs/arena/main --all
generals-arena league restore outputs/arena/main @旧策略
```

退役隐藏榜单条目并排除未来配对，保留历史评分、对局、策略工件和回放；
不会重算其余选手的评分。评分参考不能退役，涉及未完成任务或已安排比赛的选手需先完成这些任务。
历史 FFA 评分仍在完整参赛池中归一化，隐藏后的活动榜单不保证平均值恰好为 1000。

原生策略通过 CPython/NumPy 桥接调用现有 ABI 2，策略的 10ms 计时与比赛进程隔离保持不变。
临时编译工件被系统清理后会在下次加载时恢复；无策略责任归属的基础设施失败可用 `--retry` 重试，
失败尝试和原始种子仍保留。网页回放使用容量上限 32 MiB 的 gzip 缓存，同一回放的并发加载共用验证结果。

官方线上接入使用同一策略包选择入口，见[接入说明](../env/src/generals/backends/online/README.md)。
线上传输仅增加可选 `env[online]` 依赖，不改变离线比赛和训练流程。

## 随机地图

默认人数为 `[2,3,4,5,6,7,8]`，可配置 2–8 中的任意人数。宽、高分别按官方二次随机分布采样，使用 env 的出生距离、城市公平性与可达性约束：

| 人数 | 每条边的默认范围 |
| --- | --- |
| 2 | 18–23 |
| 4 | 20–26 |
| 8 | 25–33 |

山脉、城市参数均默认为官方滑块的 0.5，不是棋盘占比；标准中立城市兵力为 40–50。不强行连接全部空地。显式 `min_side/max_side` 可选择自定义尺寸分布；省略时保留官方分布。地图范围可用下文的 `league resize` 修订并记录，但更换引擎与评分规则仍需新建联赛。

## 增量赛程

每个新算法有独立持久化任务，默认预算 1000 局。每一组对局都包含该算法，不重新跑旧算法的全组合：

- 新自动评测以 **50% 的对局为 1v1**，其余 50% 在启用的 3–8 人之间均分。
  人数的换位组抽样权重为 `目标对局占比 / P`，避免把“半数换位组”误当作“半数对局”。
  单次有限预算会有波动。自动 1v1 优先匹配不同策略；显式 `run` 仍支持两个相同策略自我对战。
- 按相近 Elo（60%）、较少交手的对手（25%）、均匀探索（15%）有放回混合采样，允许对手和新算法自身出现多个实例。
- WAIT 与其他策略一样正常参赛和更新原始分数，不再安排额外校准配额。1v1 展示分数减去当前 WAIT 原始 GoR 分数；FFA 不使用固定零分基准。
- 每组共用一张图，循环轮换 P 个座位；每个实例的策略随机种子随实例轮换，不会因同名算法覆盖种子。
  最后一组按剩余预算截断，保证总局数准确，可能无法完成最后一轮完整换位。
- 以约 32 局为一个自适应窗口（保留完整换位组），根据窗口开始时的全局分数与交手记录选人。
  完成窗口后再更新配对。窗口持久化，单进程和多进程使用相同赛程；重启优先执行已有赛程。

前 6 场包含其他算法的有效对局用于快速定级，全同算法对局不推进定级；该标签不是统计置信区间。
C++ 策略决策超过 10ms 时，该席位立即投降死亡，记入淘汰时间并正常计算名次和 Elo，不中断评测或取消后续参赛资格。
按环境弃权规则停止行动，领土保留 50 tick 并继续增长、可被夺取，到期后才转中立；将军转城市。损伤领先者／最后存活者的击杀归属同样由 env 结算，Arena 不额外模拟投降规则。
C++ 硬超时退出比赛进程后，Arena 从动作日志重建存活策略状态并继续比赛；事件与计时保存在回放中。
初始化、析构、崩溃和非法返回值仍作为策略错误处理。

## 独立评分池

当前评分策略为 `gor-relative-codeforces-official-31.4.5-v10`。池由实际座位数决定，调用方不能通过标签改变归属：

| 池 | 对局 | 算法 | 展示基准 |
| --- | --- | --- | --- |
| `1v1` | 2 人 | EGF/GoR，不加人类进步奖励 | 当前原始分数减去 WAIT 原始分数 |
| `ffa` | 3–8 人 | Codeforces 预期名次、几何目标名次和表现分 | 初始及全池均值 1000；每次更新严格零和 |

GoR 的原始分数初始化为 0，更新范围为 `[-900,3299]`；WAIT 也正常更新。概率与更新系数始终使用原始 GoR，
展示时才减去 WAIT，不能把显示分数重新送入 GoR 公式。FFA 初始化为 1000，无固定锚点；
均值由更新的零和修正保持，不借助平移排行榜。两个评分池不共享比赛数、定级状态或版本差值。

1v1 使用环境胜者；外部时限截断且没有胜者时为平局。FFA 直接使用环境官方名次：有击杀者优先，然后按存活／淘汰顺序，再按存活玩家的兵力、领土和席位排序。同 tick 淘汰有顺序，不能仅按淘汰 tick 并列。
同策略重复席位先合并其跨策略名次比较，同策略内部不计比较；整组循环换位按 `rating_group_size` 分摊更新权重。
所有策略均在赛前分数上计算本局更新，比赛完成后严格按持久化赛程顺序结算。
没有固定的“500 分差等于 90%”口径：GoR 概率还取决于原始分数位置，FFA 使用 Codeforces 的 400 分逻辑尺度。

```bash
# 不给 --mode 时显示所有已启用评分池
generals-arena league leaderboard outputs/arena/main
generals-arena league leaderboard outputs/arena/main --mode 1v1 --json
generals-arena league leaderboard outputs/arena/main --mode ffa
# 停止服务和评测后，用当前评分规则重新结算此新格式联赛
generals-arena league rebuild outputs/arena/main
```

网页可以切换两个评分池；排名、版本变化、夺冠率、存活和击杀指标均只读取该池的比赛。
夺冠率为同策略任一席位夺冠的局数除以参赛局数；存活指数为（席位数 − 官方最终名次）÷（席位数 − 1），并列取平均名次，同策略席位先取平均，击杀指数使用环境记录的击杀归属，包含官方离场结算。
同策略重复席位先在局内取平均，展示指标不参与评分。置信区间按当前评分模型及独立地图作局部近似；
样本不足或相对分数尚不能识别时显示 `—`，不承诺精确覆盖率，也不使用它重排榜单。

不保留旧 BT 评分路径或旧数据库的运行时迁移。`league rebuild` 只重算当前 schema 7 中的比赛，
不会重跑模拟，也不是旧规则升级命令。历史联赛请使用归档源码读取，或创建新联赛。

## 人类参赛

当前提供**单个人类对多个算法**，人数、地图分布、局部视野、动作边界和 Elo 规则与算法联赛一致；
人类对局隐藏底部控制栏，保留棋盘、右侧统计与缩放按钮；状态和错误显示在右侧面板。
暂不提供多个人类之间的房间或在线匹配。对手由服务器有放回选择，出生座位随机；人类只占一个席位。默认每局均匀随机抽取人数，也可以指定人数。

- 点击己方格子选中，WASD / 方向键或点击相邻格排队移动；光标立即前进，可连续规划尚未占领的路径。
- 再次点击选中格或 Z 切换下一步的半兵模式；Shift + 方向键使用半兵，执行后恢复全兵。
- Q 清空队列、E 撤销最后一步、空格取消选择，G / H 选择主城。棋盘显示排队箭头，半兵箭头为黄色。
- 服务器保存最多 256 步的输入队列，每 tick 跳过无效项后执行至多一步。请求带递增 ID，重试不会重复入队；网络回包不会把光标拉回。
  输入不再因显示 tick 稍旧而被拒绝。沿途遭到阻挡或失去起点时，相应动作按游戏规则失效。
- 操作依据 [generals.io 的移动队列与快捷键](https://generals.io/versions)，不改动底层战斗规则。
- 服务器以固定时钟推进（默认每 0.5 秒一次）；刷新、请求次数、关闭页面均不能暂停或加速计分。
  断线期间继续执行已收到的队列，耗尽后 WAIT；重新打开页面可返回当前对局。
- 对局结束后，按实际人数计入 1v1 或 FFA 榜单，并保留可回放记录。
- 右侧“弃权”按钮调用同一环境离场规则，记录弃权事件并可重放；C++ 对手的隔离比赛进程也使用同一路径。
- 身份凭证自动保存在浏览器，可复制后在另一个浏览器恢复。凭证丢失不能只靠同名取回身份。
- 服务退出 / 崩溃导致的未完成真人局标记为中断、不计分，历史中可见。

这是本机实验服务，默认且仅监听 `127.0.0.1` / `localhost`，不作为公网多人游戏服务部署。
游戏进行中 API 只返回该人类的允许观测，不发送地图种子、敌方主城位置或完整棋盘；
迷雾中的山与城市仅以同一种 `fog_obstacles` 布尔标记显示为山，不能区分类型，归属、兵力和敌方主城继续隐藏；进入视野后恢复真实地形。
算法、RL、桌面与浏览器共享上述原始观测边界；自然终局开放全图。完整回放只在比赛结束后可读取。自定义算法是管理员从 CLI 装载的可信本地 Python 代码，不是隔离的代码沙箱。C++ SDK ABI 已更新为 2，新增混淆障碍、击杀／AFK／离场字段，源码包需重新编译。

## 接入算法版本

推荐的 C++ 入口：

```bash
generals-arena commit ./heristic/my_agent/policy.cpp --name "my-model-comet-v1"
generals-arena run --player baseline/aggressive --player ./heristic/my_agent/policy.cpp \
  --size auto --games 100 --seed 42 --output ./results/test01
generals-arena status
```

`--name` 直接作为榜单显示别名；多文件策略传入目录。无需创建会话或手动注册。
同一版本重复提交恢复原任务或返回已有成绩，不会重复计分；新版本使用新别名。
自由测试不会进入正式榜单，数据也由 Arena 保存，`--output` 仅导出副本。
完整的尺寸选项、历史版本对战和工件位置见 [接入文档](RESEARCH.md)。

自定义 Python 控制器仍可使用管理接口：

创建一个 entrant JSON，然后执行 `generals-arena league add outputs/arena/main --entrant 文件.json --games 1000`：

```json
{
  "name": "search-v2",
  "kind": "custom",
  "factory": "my_agents.search:create_controller",
  "version": "git-commit-or-release-id",
  "kwargs": {"simulations": 100},
  "artifacts": {"checkpoint": "./models/search-v2.bin"}
}
```

工厂实现 `create_controller(*, player_id, seed, **kwargs) -> Controller`，控制器提供
`reset(player_id,seed)`、`act(observation) -> Action`、`on_result(move_result)`。
每局重置内部状态，依据 `observation.width/height` 处理可变棋盘；需要支持所有已启用人数。
可选 `validate_rules(map_config,visibility)` 在每局开赛前检查实际规则。
`artifacts` 路径相对于 entrant JSON，哈希后以同名关键字传入工厂。

名称、版本、参数、baseline 的 Python/C++ 源码和工件哈希参与选手身份。自定义算法的代码版本由实现者明确声明，
运行过程中须保持登记版本；不要在同一名字下替换实现。
RL 先自动导出完整策略目录，再 `commit 目录`。Arena 只读取通用 manifest，不解析网络或 PPO 配置；
详见 [策略包](RESEARCH.md#策略包)。权重手动共享，不纳入 Git。

## 保存与恢复

`outputs/arena/main/league.sqlite3` 保存不可变规则、版本信息、任务、已排赛程、所有已完成结果、
压缩动作录像和排名。结果、录像、任务进度与分数在同一个 SQLite 事务提交，避免只更新一半。
原始录像包含初始完整状态、每 tick 的结构化动作以及状态摘要；回放由环境重演并逐 tick 校验，
浏览器接收压缩后的差量帧，可以暂停、拖动时间轴、改变速度。棋盘支持滚轮缩放和拖动平移。
高倍速播放逐 tick 绘制，25× 对应默认时钟的每秒 50 tick；显示器或浏览器跟不上时降低实际速度，不跳帧追赶。
渲染只重绘变化格，统计面板复用节点并按兵力降序排列（同兵力按领地、席位排序）。
人类对局通过长轮询接收新的状态或队列确认，服务器缓存每 tick 的观测，避免重复生成视野和重复重绘。

正式 `commit` 默认最多 4 局并行，使用 `--workers N` 调整（1–32）。私有 `run` 默认 1 局，也支持该参数。
并行 worker 在独立进程中执行整局比赛，进程可跨局复用；单 worker 的纯 Python／baseline 对局可直接在当前进程运行。局内 C++ 策略和 env CPU 后端保持单线程串行，协调器只传输整局任务与结果。
严格按持久化赛程顺序提交结果，避免完成速度影响 Elo 顺序。
需要同时查看界面时运行 `generals-arena serve outputs/arena/main --no-worker`，由独立 CLI 处理评测。
使用 WAL 支持榜单读取、批量任务与真人结果并发。系统文件锁确保只有一个批量 worker 和一个网页服务；
锁在进程退出后自动释放。失败或未完成的算法局保留赛程；重试失败局会生成新的 attempt ID，并保留失败记录与回放，地图和策略种子不变，已完成局不会重算。
恢复排名不需要加载策略或 checkpoint，但回放需要匹配的环境规则。备份时停止服务后复制整个目录，
或使用 SQLite backup API；服务运行时不要只拷贝主数据库而漏掉 WAL。

地图范围可通过一次明确的配置修订调整，无需清空账号、分数或历史：

```bash
# 先结束当前真人对局并停止 serve / worker，再修改后续地图范围。
generals-arena league resize outputs/arena/main --min-side 15 --max-side 20 --side-cap 35
generals-arena serve outputs/arena/main --no-worker
```

该操作记录旧 manifest 到 `rules_history`，旧对局继续保留原 `rules_id` 和地图，新对局使用新版本。
地图配置升级允许更新人类迷雾视图，但拒绝引擎、评分、模拟或调度代码发生变化时复用同一榜单。
存在进行中 / 待执行对局，或仍有服务持锁时，修订操作会拒绝执行。

实现按四个职责目录组织，根目录只保留 CLI、模块入口和公共工具：

```text
generals_arena/
├── policies/       # entries：准入与加载；artifacts/code：工件与源码；native：C++ 策略宿主；builtin：WAIT
├── matches/        # runner、game、batch、worker：执行；replay：回放；results：终局；config：MatchConfig
├── competitions/   # tournament/evaluation/private：模式；schedules/matchmaking：赛程；jobs/store：持久任务
│                   # rating/rating_session/uncertainty：评分；leaderboard：统计；submission：策略发布
└── web/            # server：HTTP；live：真人时钟与命令队列；HTML/JS/CSS：浏览器界面
```

`competitions/tournament.py` 是固定循环赛入口，`competitions/evaluation.py` 是训练用固定面板入口，持续联赛使用增量匹配与持久任务。
各模式共用 `matches.MatchRunner` 执行、终局格式和 GoR/Codeforces 评分。单局与批量执行保留各自必要的性能和生命周期边界。
策略工件层不依赖赛事数据库；比赛 worker 不依赖赛事存储；文件锁统一接收目录。评分重建完成后由发布工作流刷新报告。
WAIT 在各模式中使用同一个可信内置策略和身份，支持标量与批量接口，不再编译原生 WAIT。
旧模块、导入路径和类名已删除，没有转发壳或旧别名。Python 调用方使用 `competitions.*`、`policies.*` 等正式路径。
单局使用 `generals.make_engine()`，批执行使用 `generals.VecEnv`，均调用现有 env CPU/CUDA 内核，Arena 不另写游戏规则。
`baselines/` 的四种策略在 `policy.cpp` 中执行记忆、候选评分、BFS、路径跟踪和随机动作评估，
Python 只负责绑定和控制器生命周期，保留 RNG 顺序及动作轨迹。前端复用 `../bzy-nya.github.io/gen` 的全屏布局、地形 SVG 图标、
玩家颜色、浮动统计面板和底部控制设计；`board.js` 适配服务器观测，不复制客户端游戏规则或 AI。

模拟复用引擎每 tick 已计算的全局统计，通过 env 的 `observe(..., stats=...)` 构造观测；视野构造集中在 env，使用两次轴向扩张。
每个玩家的棋盘仍是独立的只读副本，敌方城市数仍隐藏；与环境的原始观测逐字段比较保持一致。
单局使用一个 OpenMP 线程，C++ 策略在同一对局进程内串行调用。
Arena 的单局执行复用 env CPU 内核，批执行可选 CPU/CUDA；特征、模型与推理实现属于策略包，录像启用时保留完整 tick。

新联赛使用 schema 7，回放使用 schema 3 并固定规则 ID；独立保存各池的分数，不维护旧击杀回填缓存。源码摘要绑定赛程、执行与评分实现，旧联赛不能直接使用新代码追加比赛，
需要使用其归档源码，或创建新联赛。不会静默改写旧 manifest 或评分历史。

```bash
python -m pytest arena/tests baselines/tests -q
node --test arena/tests/web/*.test.mjs
python arena/benchmarks/simulation.py
python arena/benchmarks/verify_league.py outputs/arena/main --workers 1
```

主要 CLI 为 `commit / run / status`，模式入口为 `tournament / evaluate / league`，网页为 `serve`。
`run` 显式指定各席位，复用持久化比赛基础，但写入独立私有结果，不更新正式榜单。
固定对手的 CPU/CUDA 批评估使用 `generals_arena.competitions.evaluation`，发布策略使用 `commit`。

| 入口 | 当前执行方式 | GPU 与批推理 |
| --- | --- | --- |
| `commit / run / league` | CPU 单局执行；任务 worker 可并行不同对局，保存 SQLite 结果和录像 | 策略包可配置 CUDA 推理，环境仍是 CPU；`commit/run` 提供 `--device`，`league add` 在 entrant JSON 中设置；不做跨对局神经网络批推理 |
| `tournament / evaluate` | 按相同地图配置与策略实例组合分组，调用共享 `MatchRunner` | 可选 CPU/CUDA VecEnv；支持 `reset_batch/act_batch` 的策略按组批推理 |

独立 C++ 策略包（`kind="cpp"`）保留 CPU 隔离比赛进程和超时边界，不能传给 GPU 批执行。
四个内置 baseline 属于 `kind="baseline"`，可参与 GPU 环境批评估；二人局用原生 batch，FFA 逐席位调用标量控制器。
`--batch-size 0` 表示一次执行一个完整的兼容赛程组，不是显存探测或自动调优。NN 包与普通 Python Controller 不受 C++ SDK 的 10ms 决策计时约束。


## 历史结果

原 `outputs/arena/native_1000` 的比赛与评测结论属于当时的归档代码和评分规则，见 [历史数据说明](TRIAL-NATIVE.md)。
它不作为当前双池评分的基准。新实现不修改旧库、旧回放或旧报告；继续开发请创建新的联赛目录。

## 固定循环赛模式

循环赛枚举所有 P 个不同策略的组合，每组使用相同地图面板，并循环轮换座位。它与持续联赛、训练评估共用
`MatchRunner`、终局结果及可验证回放，不维护独立引擎。重复相同策略的多个实例可用已有 `run --player` 显式安排。

`roster.json` 使用普通策略配置列表，可混合 baseline、自定义控制器、C++ bundle、导出的 NN 策略包：

```json
[
  {"name": "random", "kind": "baseline", "policy": "random"},
  {"name": "aggressive", "kind": "baseline", "policy": "aggressive"}
]
```

`rules.json` 示例：

```json
{"map": {"players": 2}, "max_ticks": 2000}
```

```bash
generals-arena tournament --entrants roster.json --config rules.json \
  --maps-per-group 20 --seed 42 --record-replays --output outputs/arena/rr
# 对支持 CUDA 推理的包与 baseline，可附加 --backend gpu --device cuda:0
# batch-size=0 表示同一兼容策略组全量并行
```

结果写入 `report.json`，含完整赛程结果、阵容相对 Elo/CI、评分证据和执行计时；录像位于 `replays/`。
普通循环赛和持续联赛按赛程顺序应用 GoR/Codeforces 更新。2 人固定赛程加入真实 WAIT 对局；FFA 不加入固定锚点。
校准循环赛可用 `--calibrate-profile PATH` 将全部结果联合拟合并保存参考分数，后续训练评估固定这些分数。
固定模式当前每次完成整份预算，训练任务恢复和公开联赛的增量恢复由各自持久队列负责。

## 训练用批量评估

`generals_arena.competitions.evaluation.make_profile` 固定地图配置、人数、对手声明、种子、换位和预算；
`profile_id()` 解析并绑定对手工件版本、赛程、执行源码、校准数据与评分规则。
`evaluate(bundle, profile, backend="cpu"|"gpu", device=..., batch_size=...)` 运行批量比赛。
支持 `reset_batch/act_batch` 的策略直接接收 `VecEnv.observer()` 的原始局部数组观测；二人局的四种 baseline 使用原 C++ batch，
其他 Controller 使用标量接口。Arena 不构造神经网络特征。`record_replays=True` 输出可逐 tick 校验的 Arena 录像。

1v1 评估使用 `gor-batch-fit-v1`：保留 GoR 胜率函数，将整批结果联合拟合，消除逐局计分的顺序依赖。每个已观测策略对平滑一局虚拟平局，换座比赛各计半份权重。校准 profile 固定所有参考的原始分数；未校准 profile 固定 WAIT 为 0 后拟合当前面板。FFA 周期评估仍使用 Codeforces 均分 1000 的规则。
`evaluate([...], profile, ...)` 或 CLI `--candidates JSON` 执行共同候选复评：候选挑战参考，同时候选相互循环赛，一次联合评分。当前共同复评要求 1v1。无需重复参考之间的校准对局。
报告保留完整胜率矩阵和 `report.win_matrix.csv`（行策略对列策略的胜／平／负），未经比赛的单元格为空。CI 为固定参考条件下的 Fisher 近似，未包含参考校准误差或策略克制影响；选 best 仍需固定 device/backend/batch。
评估不修改公开联赛数据库，其分数与公开联赛分数不可直接混用。

评估默认不保存逐 tick 录像，需要时显式使用 `--record-replays`。`--device` 统一设置候选及所有 package 对手的推理设备。
profile 中 bundle 路径相对配置文件解析，支持 `${GENERALS_POLICY_DIR}`（默认 `~/.cache/generals/policies`）。配置和哈希可提交 Git，完整策略包通过文件共享。profile 创建示例见 [RL 文档](../rl/README.md#模型导出与-arena-评估)。

```bash
generals-arena evaluate --bundle outputs/policy_bundle --profile profile.json \
  --backend gpu --device cuda:0 --batch-size 128 --output outputs/arena/evaluation

# roster.json 为普通策略声明列表；1v1 自动补 WAIT。
generals-arena tournament --entrants roster.json --config rules.json \
  --maps-per-group 200 --calibrate-profile calibrated.json --evaluation-maps 100 \
  --backend gpu --device cuda:0 --batch-size 0 --output outputs/arena/calibration

# finalists.json 为导出包路径或策略声明列表。
generals-arena evaluate --candidates finalists.json --profile calibrated.json \
  --backend gpu --device cuda:0 --batch-size 0 --output outputs/arena/finalists
```
