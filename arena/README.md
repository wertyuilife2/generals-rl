# Generals Arena

随机地图上的持续联赛：登记 baseline / 自定义算法后生成增量对局，记录录像，计算以
random baseline 为零点的 Elo。浏览器提供榜单、对局历史、回放和人类排位。
**目前不评测 RL，也不做 RL 尺寸适配；本功能不修改 `rl/`。**

## 快速开始

在仓库根目录运行（已有 `.venv` 时先 `source .venv/bin/activate`）：

```bash
python -m pip install -e ./env -e ./baselines -e ./arena

# 创建联赛，自动登记 random 零分基准
generals-arena league init outputs/arena/main --config arena/configs/league.json

# 每个新版本默认 1000 局；先入队，启动网页时由后台执行
generals-arena league add outputs/arena/main --entrant arena/configs/aggressive.json --queue-only
generals-arena league add outputs/arena/main --entrant arena/configs/expansion.json --queue-only
generals-arena league add outputs/arena/main --entrant arena/configs/defensive.json --queue-only

generals-arena serve outputs/arena/main --port 8000
```

打开 **http://127.0.0.1:8000**。通过“排名池”切换双人 / 四人 / 八人榜；通过“对局记录”
查看回放；通过“人类参赛”创建本地玩家并开始排位。前端直接由 Python 服务提供，无需 Node、npm 或构建。
不能直接打开 `web/index.html`，它需要同源 API。

`serve` 会处理队列，所以可以在另一个终端继续 `league add ... --queue-only`。
不加 `--queue-only` 时，`add` 登记后会直接运行待评测队列。完整的 1000 局可能耗时较长，
可先用 `--games 10` 检查接入；实际结果取决于策略、地图和时限，少量测试局不能代表稳定实力。

```bash
# 仅运行剩余任务；本次最多完成 20 个新对局，然后正常退出
generals-arena league run outputs/arena/main --limit 20
# 继续剩余对局（无需重新登记）
generals-arena league run outputs/arena/main
# 修正失败的策略运行问题后重试
generals-arena league run outputs/arena/main --retry

generals-arena league leaderboard outputs/arena/main --players 4
generals-arena league leaderboard outputs/arena/main --players 2 --json
generals-arena league rebuild outputs/arena/main
```

同一版本重复登记是幂等操作，不会重复增加 1000 局；不同版本须使用新名字。
`--games` 是**该新选手任务的总局数**，分配到当时人数足够的排名池，并非每个池各 1000 局。
需要专测某人数时，初始化配置中只保留一个 `player_counts`，如 `[4]`。
八人算法局需要至少八个不同的已登记算法；八人人类局需要七个算法对手。

## 随机地图

默认人数为 `[2,4,8]`，也可配置 2–8 中的任意人数。每个地图种子独立抽样宽、高、山地密度和城市密度。
宽高范围按 `ceil([min_side,max_side] * sqrt(P/2))` 随人数扩大，让人均面积大致稳定：

| 人数 | 每条边的默认范围 |
| --- | --- |
| 2 | 18–23 |
| 4 | 26–33 |
| 8 | 36–46 |

山地密度在 15%–22%、城市密度在 2%–3.5% 随机取值。地图使用现有环境生成器，保证通行区域连通，
并保留其主城间距启发式；所有城市初始兵力仍为环境的 40。
这是本项目的可配置分布，不宣称复刻原版地图算法。参考了
[generals.io 官方更新记录](https://generals.io/versions) 中按玩家数量缩放地图及关注出生公平性的设计；
没有引入原版新增的特殊地形或规则。规则固定后不能在同一联赛中修改，修改配置需要另建目录。

## 增量赛程

每个新算法有独立持久化任务，默认预算 1000 局。每一组对局都包含该算法，不重新跑旧算法的全组合：

- 从人数足够的排名池中抽取本组人数。
- 按相近 Elo（60%）、较少交手的对手（25%）、均匀探索（15%）混合采样，不重复选同一个对手。
- 周期性强制加入 random 基准，保持与零点的直接比较。
- 每组共用一张图，循环轮换 P 个座位；每位选手的策略随机种子随选手轮换。
  最后一组按剩余预算截断，保证总局数准确，可能无法完成最后一轮完整换位。
- 完成一组后根据新分数和交手记录继续选人。已经生成的组持久化，不因重启改变。

榜单标注前 30 局为“定级中”，这只是样本量提示，不是统计置信区间。
算法异常会停止该任务并显示错误，不把异常伪造成输局；其他任务仍可继续。

## Elo 定义

每个人数独立计分，所有新选手初始为 0。使用已有 `generals_baselines` 的 `random` 策略
作为固定 0 分的参照；它是仓库里的随机启发式 baseline，不是 RL 的均匀随机动作策略。

```text
E(i,j) = 1 / (1 + 9 ** ((Rj - Ri) / 500))
ΔRi = K / (P - 1) * Σj (Sij - E(i,j)),   K = 32
```

因此 `Ri - Rj = 500` 时，期望两两得分为 90%；相差 1000 时为 `81/82 ≈ 98.78%`。
双人无平局时可以解释为胜率；有平局时是 `胜率 + 0.5 × 平局率`。
多人中比较的是**存活名次**，不意味着高出 500 分就有 90% 的整局夺冠率。

存活到最后的玩家高于已淘汰玩家；淘汰更晚者胜，同一 tick 淘汰者平。
达到时限时，所有尚存活玩家并列，其分数仍高于已淘汰玩家。
每个比较使用赛前 Elo，同时更新，并除以对手数，避免人数越多单局变分越大。
random 始终保持 0，其余玩家照常更新，因此锚定联赛不要求全体分数总和不变，分数也可以为负。

整局胜 / 负 / 平另行统计：自然结束的最终胜者记胜，其余记负；超时整局记平。
多人“整局胜率”与评分用的“两两得分”是两个不同指标。Elo 依赖对手池、地图分布和比赛顺序，
不能直接横比不同联赛或不同人数的分数，也不承诺实际胜率恰好等于公式值。

## 人类参赛

当前提供**单个人类对多个算法**，人数、地图分布、局部视野、动作边界和 Elo 规则与算法联赛一致；
暂不提供多个人类之间的房间或在线匹配。对手由服务器选择，出生座位随机。

- 点击自己的格子选中，方向键 / WASD 或点击相邻格移动；Shift 或“半兵移动”切换半兵。
- H 选择主城，空格等待。每个 tick 最多执行一个动作，同一 tick 的后续输入替换待执行动作。
- 服务器以固定时钟推进（默认每 0.5 秒一次）；刷新、请求次数、关闭页面均不能暂停或加速计分。
  断线期间使用 WAIT，重新打开页面可返回当前对局。
- 对局结束后，人类进入相应人数的同一榜单，并保留可回放记录。
- 身份凭证自动保存在浏览器，可复制后在另一个浏览器恢复。凭证丢失不能只靠同名取回身份。
- 服务退出 / 崩溃导致的未完成真人局标记为中断、不计分，历史中可见。

这是本机实验服务，默认且仅监听 `127.0.0.1` / `localhost`，不作为公网多人游戏服务部署。
游戏进行中 API 只返回该人类的允许观测，不发送地图种子、敌方主城位置或完整棋盘；
完整回放只在比赛结束后可读取。自定义算法是管理员从 CLI 装载的可信本地 Python 代码，不是隔离的代码沙箱。

## 接入算法版本

创建一个 entrant JSON，然后执行 `league add --entrant 文件.json --games 1000`：

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

名称、版本、参数、baseline 源码和工件哈希参与选手身份。自定义算法的代码版本由实现者明确声明，
运行过程中须保持登记版本；不要在同一名字下替换实现。RL 注册目前明确拒绝，后续单独考虑兼容。

## 保存与恢复

`outputs/arena/main/league.sqlite3` 保存不可变规则、版本信息、任务、已排赛程、所有已完成结果、
压缩动作录像和排名。结果、录像、任务进度与分数在同一个 SQLite 事务提交，避免只更新一半。
原始录像包含初始完整状态、每 tick 的结构化动作以及状态摘要；回放由环境重演并逐 tick 校验，
浏览器接收压缩后的差量帧，可以暂停、拖动时间轴、改变速度。

使用 WAL 支持榜单读取、批量任务与真人结果并发。系统文件锁确保只有一个批量 worker 和一个网页服务；
锁在进程退出后自动释放。失败或未完成的算法局保留赛程，重启从该局头重跑，已完成局不会重算。
恢复排名不需要加载策略或 checkpoint，但回放需要匹配的环境规则。备份时停止服务后复制整个目录，
或使用 SQLite backup API；服务运行时不要只拷贝主数据库而漏掉 WAL。

模块划分：`league/config.py` 地图分布，`scheduler.py` 增量赛程，`rating.py` 锚定评分，
`game.py` 对局与录像，`store.py` 数据事务，`service.py` worker 与真人时钟；
`server.py` 提供 API，`web/` 存放前端。所有新逻辑位于 `arena/`。

```bash
python -m pytest arena/tests baselines/tests -q
```

旧的固定地图批量赛 `run / leaderboard / rebuild` 保留以便读取已有实验，使用方法见
[固定地图锦标赛说明](TOURNAMENTS.md)。它使用自己的旧评分口径、JSONL 输出及独立排名池，
不与这里的新联赛合并，不用于本次随机地图评测。
