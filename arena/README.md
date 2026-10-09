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

打开 **http://127.0.0.1:8000**。所有人数共用同一个榜单；通过“对局记录”按人数筛选、
查看回放；通过“人类参赛”创建本地玩家并开始排位，人数可随机或指定。前端直接由 Python 服务提供，无需 Node、npm 或构建。
不能直接打开 `web/index.html`，它需要同源 API。

`serve` 会处理队列，所以可以在另一个终端继续 `league add ... --queue-only`。
不加 `--queue-only` 时，`add` 登记后会直接运行待评测队列。完整的 1000 局可能耗时较长，
可先用 `--games 10` 检查接入；实际结果取决于策略、地图和时限，少量测试局不能代表稳定实力。

```bash
# 仅运行剩余任务；本次最多完成 20 个新对局，然后正常退出
generals-arena league run outputs/arena/main --limit 20
# 继续剩余对局（无需重新登记）
generals-arena league run outputs/arena/main --workers 8
# 修正失败的策略运行问题后重试
generals-arena league run outputs/arena/main --retry

generals-arena league leaderboard outputs/arena/main
generals-arena league leaderboard outputs/arena/main --json
generals-arena league rebuild outputs/arena/main
```

同一版本重复登记是幂等操作，不会重复增加 1000 局；不同版本须使用新名字。
`--games` 是**该新选手任务的总局数**，所有 2–8 人对局共同更新同一个 rating。
需要专测某人数时，初始化配置中只保留一个 `player_counts`，如 `[4]`。
AI 可以重复入场，即使只登记一个 baseline 也能生成八人局。每个席位拥有独立控制器和随机种子。
同一算法出现多次时，对局数只加 1，出场数按实例数增加。

## 随机地图

默认人数为 `[2,3,4,5,6,7,8]`，也可配置 2–8 中的任意人数。每个地图种子独立抽样宽、高、山地密度和城市密度。
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

- 随机抽取本组人数 P。为抵消一组包含 P 次换位，人数抽样权重为 `1/P`，使长期逐局人数分布近似均匀。
- 按相近 Elo（60%）、较少交手的对手（25%）、均匀探索（15%）有放回混合采样，允许对手和新算法自身出现多个实例。
- 周期性强制加入 random 基准，保持与零点的直接比较。
- 每组共用一张图，循环轮换 P 个座位；每个实例的策略随机种子随实例轮换，不会因同名算法覆盖种子。
  最后一组按剩余预算截断，保证总局数准确，可能无法完成最后一轮完整换位。
- 以约 32 局为一个自适应窗口（保留完整换位组），根据窗口开始时的全局分数与交手记录选人。
  完成窗口后再更新配对。窗口持久化，单进程和多进程使用相同赛程；重启优先执行已有赛程。

前 6 场包含其他算法的有效对局用于快速定级，全同算法对局不推进定级；该标签不是统计置信区间。
算法异常会停止该任务并显示错误，不把异常伪造成输局；其他任务仍可继续。

## Rank rating

2–8 人使用同一种名次算法，每个算法版本 / 每个人类有一个全局 rating，初始为 0。
`generals_baselines` 的 random 固定为 0；它是随机启发式 baseline，不是 RL 的均匀随机动作策略。

参考 [Codeforces 的预期名次方法](https://codeforces.com/blog/entry/20762)：实际名次优于赛前预期则加分，反之减分。
本项目用带阻尼的 Newton 表现分估计代替 CF 的几何平均名次，保留指定概率刻度下的零期望漂移。
CF 的[新账号机制](https://codeforces.com/blog/entry/77890)还包含显示分与内部评分的差值；这里使用同一个可见分数，
通过前 6 场较大的学习率快速定级，之后逐步降低波动，不发固定的额外加分。

```text
E(i,j) = 1 / (1 + 9 ** ((Rj - Ri) / 500))
rankA = 1 + average(i 属于 A) Σ(j 不属于 A) [j 名次更好 ? 1 : 同名次 ? 0.5 : 0]
seedA = 1 + Σ(j 不属于 A) E(j,A)
informationA = ln(9)/500 * max(Σ E(j,A)*(1-E(j,A)), 0.04 * 对手席位数)
performanceA = RA + (seedA - rankA) / informationA
RA' = RA + alpha(n) * (performanceA - RA)
```

`n` 为此前有效对局数，前 6 场 `alpha = 0.50, 0.40, 0.32, 0.25, 0.20, 0.16`；之后
`alpha = max(0.025, 0.16 / sqrt(1 + (n-5)/12))`。配置中的 `k_factor / 32` 缩放学习率，上限为 1。
曲率下限防止极端分差下过度调整；它对输赢对称，不分别截断正负加分。
测试中，新选手连续 6 场击败 random 即可超过 500 分，而旧 K=32 更新尚不足 100 分。

存活者高于已淘汰者，淘汰越晚名次越高；同 tick 淘汰或超时仍存活的玩家并列，取平均名次。
第二名和最后一名会有不同的评分变化，即使两者都没有夺冠。兵力排序只用于棋盘统计面板，不用于判定超时名次。
同算法实例独立参赛，但评分时排除内部比较并平均其跨算法名次；每个算法只更新一次，
全同算法对局不变分、不推进定级。所有更新使用赛前评分，random 始终固定 0，其他选手允许负分。

500 分差代表 90% 的**两两名次期望得分**（胜率 + 半个平局率），不是多人整局夺冠率。
这也是模型的刻度，不承诺有限样本实测胜率恰好等于公式值。对手池、地图、人数和赛程顺序都会影响估计。

结果保存每个席位的 `ranks`，以及每种算法的 `rating_changes`：赛前 / 赛后分数、实际 / 预期名次、
表现分、学习率和有效局数。对局记录展示名次及变分。榜单的席位胜负平与夺冠率仅作参考；
自然结束的最终席位记胜，其余记负；超时记平。这些汇总值不参与 rating 计算。

## 人类参赛

当前提供**单个人类对多个算法**，人数、地图分布、局部视野、动作边界和 Elo 规则与算法联赛一致；
暂不提供多个人类之间的房间或在线匹配。对手由服务器有放回选择，出生座位随机；人类只占一个席位。默认每局均匀随机抽取人数，也可以指定人数。

- 点击己方格子选中，WASD / 方向键或点击相邻格排队移动；光标立即前进，可连续规划尚未占领的路径。
- 再次点击选中格、Z 或半兵按钮切换下一步的半兵模式；Shift + 方向键使用半兵，执行后恢复全兵。
- Q 清空队列、E 撤销最后一步、空格取消选择，G / H 选择主城。棋盘显示排队箭头，半兵箭头为黄色。
- 服务器保存最多 256 步的输入队列，每 tick 执行一步。请求带递增 ID，重试不会重复入队；网络回包不会把光标拉回。
  输入不再因显示 tick 稍旧而被拒绝。沿途遭到阻挡或失去起点时，相应动作按游戏规则失效。
- 操作依据 [generals.io 的移动队列与快捷键](https://generals.io/versions)，不改动底层战斗规则。
- 服务器以固定时钟推进（默认每 0.5 秒一次）；刷新、请求次数、关闭页面均不能暂停或加速计分。
  断线期间继续执行已收到的队列，耗尽后 WAIT；重新打开页面可返回当前对局。
- 对局结束后，人类进入统一榜单，并保留可回放记录。
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
浏览器接收压缩后的差量帧，可以暂停、拖动时间轴、改变速度。棋盘支持滚轮缩放和拖动平移。
高倍速播放逐 tick 绘制，25× 对应默认时钟的每秒 50 tick；显示器或浏览器跟不上时降低实际速度，不跳帧追赶。
渲染只重绘变化格，统计面板复用节点并按兵力降序排列（同兵力按领地、席位排序）。
人类对局通过长轮询接收新的状态或队列确认，服务器缓存每 tick 的观测，避免重复生成视野和重复重绘。

批量评测通过 `--workers 8` 并行模拟，并严格按持久化赛程顺序提交结果，避免完成速度影响 Elo 顺序。
需要同时查看界面时运行 `generals-arena serve outputs/arena/main --no-worker`，由独立 CLI 处理评测。
使用 WAL 支持榜单读取、批量任务与真人结果并发。系统文件锁确保只有一个批量 worker 和一个网页服务；
锁在进程退出后自动释放。失败或未完成的算法局保留赛程，重启从该局头重跑，已完成局不会重算。
恢复排名不需要加载策略或 checkpoint，但回放需要匹配的环境规则。备份时停止服务后复制整个目录，
或使用 SQLite backup API；服务运行时不要只拷贝主数据库而漏掉 WAL。

模块划分：`league/config.py` 地图分布，`scheduler.py` 增量赛程，`rating.py` 锚定评分，
`game.py` 对局与录像，`store.py` 数据事务，`service.py` worker 与真人时钟；
`server.py` 提供 API，`web/` 存放前端。联赛逻辑位于 `arena/`；`baselines/` 的候选排序和随机动作评估使用数组计算，保留 RNG 顺序及动作轨迹。前端复用 `../bzy-nya.github.io/gen` 的全屏布局、地形 SVG 图标、
玩家颜色、浮动统计面板和底部控制设计；`board.js` 适配服务器观测，不复制客户端游戏规则或 AI。

新版存储为 schema 3。旧的按人数分榜或普通 Elo 数据库保留为历史实验，不能直接续跑；请创建新的名次评分联赛目录，
避免把旧赛程分布与新实验混合。

```bash
python -m pytest arena/tests baselines/tests -q
node --test arena/tests/web/*.test.mjs
python arena/benchmarks/simulation.py
python arena/benchmarks/verify_league.py outputs/arena/rank_1000 --workers 8
```

旧的固定地图批量赛 `run / leaderboard / rebuild` 保留以便读取已有实验，使用方法见
[固定地图锦标赛说明](TOURNAMENTS.md)。它使用自己的旧评分口径、JSONL 输出及独立排名池，
不与这里的新联赛合并，不用于本次随机地图评测。


## 本次 1000 局试运行

运行目录：`outputs/arena/rank_1000`。报告见 [TRIAL.md](TRIAL.md)。使用以下命令查看：

```bash
generals-arena serve outputs/arena/rank_1000 --no-worker
generals-arena league leaderboard outputs/arena/rank_1000
```

早先普通 Elo 的 433 局保留在 `outputs/arena/unified_1000`，没有覆盖或混入新榜单。
对固定 8 人、45×41、3000 tick 的完整对局，优化前约 48.0 秒，优化后约 6.5 秒；
所有动作与逐 tick 状态摘要相同。可用 `python arena/benchmarks/simulation.py --reference 870e764`
加载 Git 中的原 baseline 作同场对照；该命令仅导出代码到临时目录，不切换或修改工作区。
