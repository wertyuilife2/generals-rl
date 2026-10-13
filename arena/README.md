# Generals Arena

随机地图上的持续联赛：登记 baseline / 自定义算法后生成增量对局，记录录像，计算以
WAIT 固定为零点的 Elo。浏览器提供榜单、对局历史、回放和人类排位。
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

- 新自动评测以 **50% 的对局为 1v1**，其余 50% 在启用的 3–8 人之间均分。
  人数的换位组抽样权重为 `目标对局占比 / P`，避免把“半数换位组”误当作“半数对局”。
  单次有限预算会有波动。自动 1v1 优先匹配不同策略；显式 `run` 仍支持两个相同策略自我对战。
- 按相近 Elo（60%）、较少交手的对手（25%）、均匀探索（15%）有放回混合采样，允许对手和新算法自身出现多个实例。
- WAIT 作为零分参照，不进入普通对手池。自动任务前期安排 2 组 **1v1 校准**，此后每 50 局
  增加一组，已有交手抵扣额度；1000 局任务最多 40 局（4%）包含 WAIT。窗口内尚未执行的校准
  也计入额度，避免并行规划重复补量。每局最多一个 WAIT，人类匹配和多人自动对局不加入 WAIT。
  显式 `run --player baseline/wait` 不受这一限制。
- 每组共用一张图，循环轮换 P 个座位；每个实例的策略随机种子随实例轮换，不会因同名算法覆盖种子。
  最后一组按剩余预算截断，保证总局数准确，可能无法完成最后一轮完整换位。
- 以约 32 局为一个自适应窗口（保留完整换位组），根据窗口开始时的全局分数与交手记录选人。
  完成窗口后再更新配对。窗口持久化，单进程和多进程使用相同赛程；重启优先执行已有赛程。

前 6 场包含其他算法的有效对局用于快速定级，全同算法对局不推进定级；该标签不是统计置信区间。
决策超过 10ms 时，该席位立即投降死亡，记入淘汰时间并正常计算名次和 Elo，不中断评测或取消后续参赛资格。
领地变为灰色中立地，主城变为城市，原有驻军保留，停止行动和产兵；投降不计为其他玩家的击杀。
C++ 硬超时退出比赛进程后，Arena 从动作日志重建存活策略状态并继续比赛；事件与计时保存在回放中。
初始化、析构、崩溃和非法返回值仍作为策略错误处理。

## Rank rating

2–8 人使用同一种名次算法，每个算法版本 / 每个人类有一个全局 rating，初始为 0。
WAIT 的实际 rating 固定为 0；random 与其他算法正常计分。每局只更新参赛选手，不再通过减去浮动基准分平移全榜。

当前使用**本局名次更新 + 有方向和幅度约束的历史校准**。令 `a = ln(9)/500`，所有本局概率由赛前分数计算：

```text
p(i,j) = 1 / (1 + exp(-a * (Ri - Rj)))
y(i,j) = 两种策略所有席位组合的平均名次得分（领先 1、并列 0.5、落后 0）
w = (1v1 ? 1 : 0.5) / (不同对手策略数 × 同图换位局数)
Nij += w; Tij += w * y(i,j)
Di = Σ(本局对手 j) w * (y(i,j) - p(i,j))
Ii = 0.25 + Σ(所有交过手的 j) Nij * max(0.04, p(i,j)*(1-p(i,j)))
Gi = 1 / (a * Ii)
Bi = Ri + Gi * Di
Li = 2 * Gi * Σ(本局对手 j) w
```

`Bi` 是本局在线更新给出的分数。随后只对本局参赛者拟合累计 Bradley–Terry 名次比较，
未参赛者保持赛前分数、WAIT 保持 0，并限制最终分数：

- 本局应加分时，在 `[Bi, Ri + Li]` 内校准。
- 本局应扣分时，在 `[Ri - Li, Bi]` 内校准。
- 本局实际与预期相等时，保持原分。

因此 **1v1 胜者加分、败者扣分**；独占第一名不扣分、独占最后一名不加分。
多人按全部名次比较的平均表现计分，重复策略的多个席位合并评价，不能只凭一个席位夺冠判整个策略获胜。
同名次按 0.5 比较，因此强者可能因平局失分。每局只更新参赛者，未参赛者不变。

历史校准用于修正长期估计偏差，尤其是 WAIT 少量参赛时的零点偏差。
仅用累计对局数不断缩小在线步长，会把早期错误零点冻结，即使参赛者后来与 WAIT 连续打平也难以修正。
新版在选手再次出场且本局方向允许时逐步校准，不平移全榜，也不强行把负分截到零。
`Li` 将总变分限制在两个完全意外结果对应的在线步长内，避免把积累的偏差一次性计入一局。

`rating_pairs` 持久化累计比较权重 `Nij` 和名次得分 `Tij`。
历史拟合在每个已观察策略对上加一次半胜、半负的平滑，避免全胜记录产生无穷分数。
在线分母中的 `0.25` 是初始阻尼，曲率下限 `0.04` 防止极端胜率下步长过大；
它们与半胜半负平滑均不计入有效样本和置信区间信息。
在线基础步在模型成立时具有零条件期望；加入有约束的历史校准后，不再声称最终变分具有这一性质。
这是一套自定义的按时间结算的评分规则，不是无约束全局极大似然估计。`k_factor` 只供旧规则复现使用。

重建时保持原结算顺序；同图换位共用一份样本权重，但分次结算仍可能产生顺序差异。
WAIT 校准与普通预测准确度分别验证：基准比较的实测期望得分、预测得分，以及消除两者差异所需的零点修正量。
该修正量仅作为诊断，不实际平移分数。单一 Elo 无法完全拟合策略相克和超时平局，负分仍可能是合理结果。

存活者高于已淘汰者，淘汰越晚名次越高；同 tick 淘汰或到达时限仍存活的玩家并列。
第二名和最后一名贡献不同的名次比较，即使两者都没有夺冠。兵力排序仅用于棋盘统计面板。
同策略实例之间不计比较，与另一策略的多个席位先取平均，再对不同策略等权计入。
1v1 的相对权重是多人的两倍；多人每种策略每张独立地图的总权重为 0.5，1v1 为 1。
同图的整组循环换位共用一份权重，避免把相关结果当成多份独立证据。全同算法对局不变分、不推进定级。

新计划保存 `rating_policy = rank-calibrated-outcome-v7` 和 `rating_group_size`。
默认 `league rebuild` 使用各记录原有的规则，恢复相同的历史变分和累计证据。
显式升级已有联赛时，先停止服务和评测，再运行：

```bash
generals_arena league rebuild outputs/arena/native_1000 --current-policy
```

升级在一个数据库事务中重新结算全部已完成结果，并更新待执行计划的评分版本。
地图、动作、回放、结果、任务进度保持不变，不重跑模拟，不保留旧评分备份。
正式 `publication.json` 及已有重测报告的分数快照同步替换，并标记 `rating_recomputed`。
未升级的私有实验仍可按原有规则复现；旧版本兼容代码不影响新版计分。

500 分差代表 90% 的**两两名次期望得分**（胜率 + 半个平局率），不是多人整局夺冠率。
这是模型的概率刻度，不承诺有限样本实测胜率恰好相等。地图、人数、对手风格和赛程仍会影响估计。

结果保存各席位 `ranks` 和各策略 `rating_changes`：赛前 / 赛后分数、实际 / 预期名次、有效局数、累计比较权重、在线变分、历史校准变分及本局幅度上限。
对局记录展示名次及变分。榜单显示以下指标：

- **Elo**：所有人数共用的当前评分，旁边的 `±` 为 95% 估计区间半宽，悬停显示上下界。
- **▲**：当前 Elo 减去同名系列上一已注册版本的当前 Elo，按 `名称-vN` 的数字版本排序
  （例如 `model-kestrel-v10` 对比 `model-kestrel-v2`）。无上一版本、任一版本尚无已完成对局，
  或人类用户均显示 `—`。
- **胜率**：夺冠局数 ÷ 参赛局数。同策略任一席位夺冠即计一场胜利，超时局计入分母。
- **存活指数**：每席位存活 tick 数 ÷ 对局总 tick 数，存活到结束为 100%；首次决策即投降的零 tick 对局中，投降者为 0%，胜者为 100%。
- **击杀指数**：每席位亲自攻占敌方将军的次数 ÷（总席位数 − 1），包含同策略的其他席位；
  不把被击杀者此前的击杀数转移给胜者。

存活、击杀指数均先对同一策略在每局的席位取平均，再对它参与的各局取平均，以百分比显示。
只统计已完成对局（含自我对战），没有对局时显示 `—`。这些展示指标不参与 rating 计算。
新对局结果记录每席位的 `kills`；旧对局首次启动服务时通过 env 后端重放验证并补齐击杀统计，
保存在联赛目录的 `statistics.sqlite3` 缓存中。缓存可删除重建，不修改原始对局、录像或评分。

置信区间为局部名次模型的近似估计：使用 Bradley–Terry 的概率尺度
`p = 1 / (1 + exp(-ln(9) × Elo差 / 500))`，由对手关系图的 Fisher 信息矩阵求相对 WAIT 的方差。
同策略重复席位合并，同图换位按 `rating_group_size` 分摊信息；权重与新版计分一致，1v1 为 1，多人为 0.5。
初始阻尼、曲率下限及半胜半负平滑不加入 Fisher 信息，不能用它们人为缩窄区间。
区间为 `Elo ± 1.96 × 标准误`，会传播 WAIT 校准和间接对手比较的不确定性。
WAIT 本身固定为 `[0, 0]`；选手或基准不足 6 组独立地图、没有通向基准的比较路径，
或信息矩阵无法稳定求逆时显示 `—`。这是在当前在线评分处计算的局部模型近似，
不是精确 95% 覆盖保证，也不用于重排榜单。对手风格、多人相关性及稀疏校准都会限制这一近似。
分数稳定不等于零点已精确识别，WAIT 校准不足时区间仍应保持宽。
模型标准误的背景可参考 [BradleyTerry2 作者文档](https://hturner.r-universe.dev/BradleyTerry2/doc/BradleyTerry.html)。

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

正式 `commit` 默认最多 4 局并行，使用 `--workers N` 调整（1–32）。私有 `run` 默认 1 局，也支持该参数。
每局有独立进程，局内 C++ 策略和 env CPU 后端保持单线程串行；协调线程只传输整局任务与结果。
严格按持久化赛程顺序提交结果，避免完成速度影响 Elo 顺序。
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

模块划分：`league/config.py` 地图分布，`scheduler.py` 增量赛程，`rank_model.py` 名次比较、在线更新与有约束的历史校准，
`rating_policy.py` 计分版本和人数权重（`rating.py` 保留历史规则），
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

## 全算法重新评测

停止服务和旧评测进程后，可以直接清除旧联赛对局、rating 和正式成绩报告，再为每个已注册算法安排 1000 局。
该命令**不保留旧数据备份**；保留策略工件、私有实验和人类账号。WAIT 仍只少量参加校准，不单独安排 1000 局。

```bash
python arena/benchmarks/reevaluate_league.py outputs/arena/native_1000 --reset --games 1000 --workers 4
# 中断后继续：省略 --reset，保留已经完成的新对局。
python arena/benchmarks/reevaluate_league.py outputs/arena/native_1000 --workers 4
```

计分的只读对比基准（使用完全相同的已记录结果，比较按时间顺序的赛前预测误差和 200 局后的变分）：

```bash
python arena/benchmarks/rating.py outputs/arena/native_1000 --output outputs/arena/native_1000/rating_validation.json
```
