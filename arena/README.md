# Generals Arena

随机地图上的持续联赛：登记 baseline / 自定义算法后生成增量对局，记录录像，计算以
random baseline 为零点的 Elo。浏览器提供榜单、对局历史、回放和人类排位。
**目前不评测 RL，也不做 RL 尺寸适配；本功能不修改 `rl/`。**

C++ 启发式使用 [统一接入接口](RESEARCH.md)：`generals_arena commit 源码 --name 别名`
自动编译并完成 1000 局正式评测；`generals_arena run --player ... --player ...` 自由测试。
Arena 管理源码版本、10ms 限时、评分和录像。可直接使用
[60 分钟研究提示词](AUTORESEARCH_PROMPT.txt)，其中最多三次正式评测是该研究任务的约束。

## 快速开始

在仓库根目录运行（已有 `.venv` 时先 `source .venv/bin/activate`）：

需要 Linux/macOS、C++17 编译器和 OpenMP。macOS 使用 clang 与已安装的 libomp，
非标准位置可设置 `GENERALS_LIBOMP_PREFIX`；编译器可用 `CXX` 指定。首次运行自动编译并缓存。

```bash
python -m pip install -e ./env -e ./baselines -e ./arena
generals-arena serve outputs/arena/native_1000 --port 8086 --no-worker
```

打开 **http://localhost:8086**（默认端口也是 8086）。当前正式联赛为
`outputs/arena/native_1000`，保留 4 个 baseline 和完整的 1000 局结果、评分及录像。
旧联赛、旧运行代码副本、迁移备份和接口试验数据已清理。
所有人数共用同一个榜单；通过“对局记录”按人数筛选、
查看回放；通过“人类参赛”创建本地玩家并开始排位，人数可随机或指定。前端直接由 Python 服务提供，无需 Node、npm 或构建。
不能直接打开 `web/index.html`，它需要同源 API。

`--no-worker` 让正式网页与独立的研究评测共用联赛；榜单、录像和人类对局仍正常工作。
不带该参数的 `serve` 会处理队列，所以可以在另一个终端继续 `league add ... --queue-only`。
不加 `--queue-only` 时，`add` 登记后会直接运行待评测队列。完整的 1000 局可能耗时较长，
可先用 `--games 10` 检查接入；实际结果取决于策略、地图和时限，少量测试局不能代表稳定实力。

若需要创建独立的新联赛（目标目录必须不存在），运行：

```bash
generals-arena league init outputs/arena/main --config arena/configs/league.json
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
`--games` 是**该新选手任务的总局数**，所有 2–8 人对局共同更新同一个 rating。
需要专测某人数时，初始化配置中只保留一个 `player_counts`，如 `[4]`。
AI 可以重复入场，即使只登记一个 baseline 也能生成八人局。每个席位拥有独立控制器和随机种子。
同一算法出现多次时，对局数只加 1，出场数按实例数增加。

## 随机地图

默认人数为 `[2,3,4,5,6,7,8]`，也可配置 2–8 中的任意人数。每个地图种子独立抽样宽、高、山地密度和城市密度。
宽高范围按 `min(side_cap, ceil([min_side,max_side] * sqrt(P/2)))` 随人数扩大。默认基准边长 15–20，上限 35，所有人数的地图宽高都在 15–35 之间：

| 人数 | 每条边的默认范围 |
| --- | --- |
| 2 | 15–20 |
| 4 | 22–29 |
| 8 | 30–35 |

山地密度在 15%–22%、城市密度在 2%–3.5% 随机取值。地图使用现有环境生成器，保证通行区域连通，
并保留其主城间距启发式；所有城市初始兵力仍为环境的 40。
这是本项目的可配置分布，不宣称复刻原版地图算法。参考了
[generals.io 官方更新记录](https://generals.io/versions) 中按玩家数量缩放地图及关注出生公平性的设计；
没有引入原版新增的特殊地形或规则。地图范围可通过下文的 `league resize` 明确修订并留存版本；引擎与评分规则变更仍需要另建联赛。

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
人类对局隐藏底部控制栏，保留棋盘、右侧统计与缩放按钮；状态和错误显示在右侧面板。
暂不提供多个人类之间的房间或在线匹配。对手由服务器有放回选择，出生座位随机；人类只占一个席位。默认每局均匀随机抽取人数，也可以指定人数。

- 点击己方格子选中，WASD / 方向键或点击相邻格排队移动；光标立即前进，可连续规划尚未占领的路径。
- 再次点击选中格或 Z 切换下一步的半兵模式；Shift + 方向键使用半兵，执行后恢复全兵。
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
迷雾中的山与城市仅以同一种 `fog_obstacles` 布尔标记显示为山，不能区分类型，归属、兵力和敌方主城继续隐藏；进入视野后恢复真实地形。
该显示规则仅用于人类视图，算法与 RL 的观测协议保持原样。完整回放只在比赛结束后可读取。自定义算法是管理员从 CLI 装载的可信本地 Python 代码，不是隔离的代码沙箱。

## 接入算法版本

推荐的 C++ 入口：

```bash
generals_arena commit ./heristic/my_agent/policy.cpp --name "my-model-comet-v1"
generals_arena run --player baseline/aggressive --player ./heristic/my_agent/policy.cpp \
  --size auto --games 100 --seed 42 --output ./results/test01
generals_arena status
```

`--name` 直接作为榜单显示别名；多文件策略传入目录。无需创建会话或手动注册。
同一版本重复提交恢复原任务或返回已有成绩，不会重复计分；新版本使用新别名。
自由测试不会进入正式榜单，数据也由 Arena 保存，`--output` 仅导出副本。
完整的尺寸选项、历史版本对战和工件位置见 [接入文档](RESEARCH.md)。

自定义 Python 控制器仍可使用管理接口：

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

名称、版本、参数、baseline 的 Python/C++ 源码和工件哈希参与选手身份。自定义算法的代码版本由实现者明确声明，
运行过程中须保持登记版本；不要在同一名字下替换实现。RL 注册目前明确拒绝，后续单独考虑兼容。

## 保存与恢复

`outputs/arena/main/league.sqlite3` 保存不可变规则、版本信息、任务、已排赛程、所有已完成结果、
压缩动作录像和排名。结果、录像、任务进度与分数在同一个 SQLite 事务提交，避免只更新一半。
原始录像包含初始完整状态、每 tick 的结构化动作以及状态摘要；回放由环境重演并逐 tick 校验，
浏览器接收压缩后的差量帧，可以暂停、拖动时间轴、改变速度。棋盘支持滚轮缩放和拖动平移。
高倍速播放逐 tick 绘制，25× 对应默认时钟的每秒 50 tick；显示器或浏览器跟不上时降低实际速度，不跳帧追赶。
渲染只重绘变化格，统计面板复用节点并按兵力降序排列（同兵力按领地、席位排序）。
人类对局通过长轮询接收新的状态或队列确认，服务器缓存每 tick 的观测，避免重复生成视野和重复重绘。

批量评测默认串行执行，C++ 研究策略要求 `--workers 1`。仅 baseline 联赛可选择多个 worker，
并严格按持久化赛程顺序提交结果，避免完成速度影响 Elo 顺序。
需要同时查看界面时运行 `generals-arena serve outputs/arena/main --no-worker`，由独立 CLI 处理评测。
使用 WAL 支持榜单读取、批量任务与真人结果并发。系统文件锁确保只有一个批量 worker 和一个网页服务；
锁在进程退出后自动释放。失败或未完成的算法局保留赛程，重启从该局头重跑，已完成局不会重算。
恢复排名不需要加载策略或 checkpoint，但回放需要匹配的环境规则。备份时停止服务后复制整个目录，
或使用 SQLite backup API；服务运行时不要只拷贝主数据库而漏掉 WAL。

地图范围可通过一次明确的配置修订调整，无需清空账号、分数或历史：

```bash
# 先结束当前真人对局并停止 serve / worker，再修改后续地图范围。
generals-arena league resize outputs/arena/native_1000 --min-side 15 --max-side 20 --side-cap 35
generals-arena serve outputs/arena/native_1000 --no-worker
```

该操作记录旧 manifest 到 `rules_history`，旧对局继续保留原 `rules_id` 和地图，新对局使用新版本。
地图配置升级允许更新人类迷雾视图，但拒绝引擎、评分、模拟或调度代码发生变化时复用同一榜单。
存在进行中 / 待执行对局，或仍有服务持锁时，修订操作会拒绝执行。

模块划分：`league/config.py` 地图分布，`scheduler.py` 增量赛程，`rating.py` 锚定评分，
`game.py` 对局与录像，`observation.py` 算法观测，`views.py` 人类迷雾显示，`store.py` 数据事务，`service.py` worker 与真人时钟；
`server.py` 提供 API，`web/` 存放前端。联赛逻辑位于 `arena/`；模拟统一调用 `generals.make_engine()`，
由 `env/backends/cpu` 的单局适配器执行已有的 `step.cpp`，Arena 不维护独立规则后端。
`baselines/` 的四种策略在 `policy.cpp` 中执行记忆、候选评分、BFS、路径跟踪和随机动作评估，
Python 只负责绑定和控制器生命周期，保留 RNG 顺序及动作轨迹。前端复用 `../bzy-nya.github.io/gen` 的全屏布局、地形 SVG 图标、
玩家颜色、浮动统计面板和底部控制设计；`board.js` 适配服务器观测，不复制客户端游戏规则或 AI。

模拟复用引擎每 tick 已计算的全局统计，算法观测用两次轴向扩张计算视野，减少重复扫描和数组复制。
每个玩家的棋盘仍是独立的只读副本，敌方城市数仍隐藏；与环境的原始观测逐字段比较保持一致。
单局使用一个 OpenMP 线程，C++ 策略在同一对局进程内串行调用。
Arena 复用原 CPU 内核，未修改 RL，也不省略录像 tick。

存储使用 schema 3；当前数据可直接使用当前源码继续运行。

```bash
python -m pytest arena/tests baselines/tests -q
node --test arena/tests/web/*.test.mjs
python arena/benchmarks/simulation.py
python arena/benchmarks/verify_league.py outputs/arena/native_1000 --workers 1
```

主要 CLI 为 `commit / run / status`，管理命令为 `league / serve`。
固定地图批量赛的旧 CLI、示例配置和历史报告已移除；当前 `run` 是显式指定席位的自由测试。
`tournament.py` 及其辅助模块仍作为 Python 兼容接口保留：现有 RL 评测和测试依赖它们；
它们不进入当前联赛或网页。RL 代码保持原样。


## 当前 1000 局数据

CPU 原生后端与 C++ baseline 的运行目录为 `outputs/arena/native_1000`，数据说明见
[TRIAL-NATIVE.md](TRIAL-NATIVE.md)。使用以下命令查看和校验：

```bash
generals-arena serve outputs/arena/native_1000 --port 8086 --no-worker
generals-arena league leaderboard outputs/arena/native_1000
python arena/benchmarks/verify_league.py outputs/arena/native_1000 --workers 1
```

固定地图、种子和动作轨迹的性能检查（重复五次，输出中位数）：

```bash
python arena/benchmarks/simulation.py --repeat 5
```

默认 fixture 为范围内的 35×33 地图，校验每步动作和状态摘要。
需要版本间性能比较时，`--reference` / `--reference-arena` 可从 Git 导出指定版本到临时目录，
无需在工作区保留旧运行代码副本。
