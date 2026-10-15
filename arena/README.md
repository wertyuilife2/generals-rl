# Generals Arena

随机地图上的持续联赛：登记 baseline / 自定义算法后生成增量对局，记录录像与评分。
1v1 使用以 WAIT 归一化为零点的 GoR，FFA 使用均值 1000 的 Codeforces rating。浏览器提供榜单、对局历史、回放和人类排位。
**目前不评测 RL，也不做 RL 尺寸适配；本功能不修改 `rl/`。**

C++ 启发式使用 [统一接入接口](RESEARCH.md)：`generals_arena commit 源码 --name 别名`
自动编译并完成 500 局 1v1 + 500 局 FFA 正式评测；`generals_arena run --player ... --player ...` 自由测试。
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
`league add --games` 是该任务的总局数。正式 `commit` 固定为 500 局 1v1 + 500 局 FFA；两种模式分别更新 rating。
需要专测某人数时，初始化配置中只保留一个 `player_counts`，如 `[4]`。
AI 可以重复入场，即使只登记一个 baseline 也能生成八人局。每个席位拥有独立控制器和随机种子。
同一算法出现多次时，对局数只加 1，出场数按实例数增加。

## 随机地图

默认人数为 `[2,3,4,5,6,7,8]`。Arena 调用共享 `env` 的通用生成器，
`map_generation` 参数的默认值对应 generals.io v31.4.3 的标准随机地图，无组队或特殊 modifier。
配置可只写 `"map_generation": {}`；尺寸、山城密度、出生公平性和城市公平性采用官方 0–1 滑块，默认 `0.5`。
密度滑块不等于地图格子的占比。

| 人数 | 每条边的默认候选范围 |
| --- | --- |
| 2 | 18–23 |
| 3 | 19–25 |
| 4 | 20–26 |
| 5 | 21–28 |
| 6 | 22–29 |
| 7 | 24–31 |
| 8 | 25–33 |

宽高独立使用平方随机数抽样，偏向较小地图；不合格时整张重试，包括重新抽样尺寸。
默认城市守军 40–50；1v1 按路径距离配对城市并共享守军值，出生点比较周围可扩张空间；
多人使用按出生点扩展的城市分配。出生点采用曼哈顿间距（1v1 至少 15 格），
将军间必须无需攻城即可互通。城市足够时，每个出生点距离 6 内至少一座城市。
不通过拆山强制修复地图，也不保证所有空地连通。

通用参数及调用方式见 [env 地图生成说明](../env/README.md#随机地图生成)。
规则依据 [官方公开客户端 v31.4.3](https://generals.io/generals-main-prod-v31.4.3-485dbf55.js)
及 [官方更新记录](https://generals.io/versions)。本地使用独立、可复现的 RNG，
不与 JavaScript `Math.random()` 逐种子对应；不包含组队、特殊地形或 modifier。

每局计划保存完整生成参数、成功尝试编号和地图种子；回放还保存初始地图。
旧配置中没有 `map_generation` 的已存在联赛继续解释为旧生成器，不能静默换规则。
完成当前评测并停止服务/worker 后，显式采用新默认：

```bash
generals_arena league configure-maps outputs/arena/native_1000
# 自定义通用参数：JSON 直接包含 MapGenerationConfig 的字段
# generals_arena league configure-maps outputs/arena/native_1000 --config map-options.json
```

这条命令接受当前安装的地图生成代码版本，更新后续地图规则，保留已有成绩和回放；
不会重跑对局或重算历史 rating。若同时改了战斗或评分代码，应新建联赛。
`map_generation: null` 和 `league resize` 仍提供旧版面积密度/尺寸分布，用于历史复现和专门实验。

## 增量赛程

每个新算法有独立持久化任务，默认预算 1000 局。每一组对局都包含该算法，不重新跑旧算法的全组合：

- 正式 `commit` 精确安排 **500 局 1v1 + 500 局 FFA**。配额与进度独立持久化；
  中断续跑、并行、最后一组不足完整换位均不会超过任何模式的配额。
- FFA 在启用的 3–8 人之间按 `1/P` 抽换位组，避免大局人数获得过多比赛。
- 1v1 按相近 rating（80%）、较少交手（15%）、均匀探索（5%）混合配对；FFA 为 60% / 25% / 15%。
  相近程度为 `exp(-abs(分差)/300)`，只读取当前模式的 rating。
  优先无放回选择策略；可用策略足够时每局全部不同，不足时均匀补重复席位，包含目标选手的已有席位。
  显式 `run --player ...` 始终保留用户指定的重复席位和自我对战。
- WAIT 和其他策略使用同一个对手池、权重与重复席位规则；不设出场上限，不强制校准，也可以出现在人类对局中。
- 每组共用一张图，循环轮换 P 个座位；每个实例的策略随机种子随实例轮换，不会因同名算法覆盖种子。
  最后一组按剩余预算截断，保证总局数准确，可能无法完成最后一轮完整换位。
- 以约 32 局为一个自适应窗口（保留完整换位组），根据窗口开始时对应模式的分数与交手记录选人。
  完成窗口后再更新配对。窗口持久化，单进程和多进程使用相同赛程；重启优先执行已有赛程。

前 6 场包含其他算法的有效对局标记为定级中，全同算法对局不推进定级；该标签不是统计置信区间。
决策超过 10ms 时，该席位立即投降死亡，记入淘汰时间并正常计算名次和 Elo，不中断评测或取消后续参赛资格。
领地变为灰色中立地，主城变为城市，原有驻军保留，停止行动和产兵；投降不计为其他玩家的击杀。
C++ 硬超时退出比赛进程后，Arena 从动作日志重建存活策略状态并继续比赛；事件与计时保存在回放中。
初始化、析构、崩溃和非法返回值仍作为策略错误处理。

## 1v1 GoR / FFA Codeforces rating

每个算法版本 / 人类分别拥有两套独立 rating、定级次数、置信区间和展示指标。每局只更新参与策略的原始评分。

**1v1** 使用 EGF/GoR 的 Bradley–Terry 胜率模型及动态更新幅度。所有策略（包括 WAIT）原始评分初始为 0，正常参与结算：

```text
beta(r) = -7 * ln(3300-r)
p(i,j) = 1 / (1 + exp(beta(rj)-beta(ri)))
con(r) = ((3300-r)/200)^1.6
r' = r + con(r) * (实际得分 - p) / 同图换位局数
榜单评分(i) = 原始评分(i) - 原始评分(WAIT)
```

胜 / 平 / 负分别为 1 / 0.5 / 0。Arena 不加入面向棋手成长的 bonus，避免固定策略无故涨分；
原始评分范围为 -900 至 3299，胜者原始评分不扣分、败者不加分，WAIT 适用相同规则。
WAIT 仅作为展示零点，不在计算中锁分。WAIT 参赛导致其原始评分变化时，榜单整体平移；
未参赛者的原始评分和相互分差不变，对局记录只显示本局原始评分变化，不把零点移动计入本局得失分。
胜率预测、匹配和置信区间导数始终使用原始评分，不能把归一化后的分数代回非线性的 GoR 公式。
更新幅度不会随累计局数衰减到零。GoR 的胜率同时取决于双方分数，不再使用固定“分差 500 = 90%”。
公式参见 [EGF 实现源码](https://docs.rs/skillratings/latest/src/skillratings/egf.rs.html)，
[EGF 评级说明](https://www.europeangodatabase.eu/EGD/EGF_rating_system.php)。

**FFA** 使用 [Codeforces 公布的算法](https://codeforces.com/blog/entry/20762)计算预期名次、几何平均目标名次和表现分。
所有策略（含 WAIT）初始为 1000；不同策略等权，不固定 WAIT 分数：

```text
p(i,j) = 1 / (1 + 10^((rj-ri)/400))
seed(i) = 1 + Σ p(j,i)
rank(i) = 1 + Σ 对手策略领先自己的平均得分（领先 1、并列 0.5、落后 0）
target(i) = sqrt(seed(i) * rank(i))
performance(i) = 二分求解使预期名次等于 target(i) 的分数
raw_delta(i) = (performance(i)-ri)/2
delta(i) = (raw_delta(i)-本局不同策略的平均 raw_delta) / 同图换位局数
```

Arena 将 CF 的通缩修正替换为本局零和修正，因此全体已注册策略（含退役者）的均值保持 1000，
不需要移动未参赛者的分数。隐藏退役者后的可见榜单均值可能不等于 1000。400 分差对应约 90.9% 的两两期望得分，
不是多人整局夺冠率。FFA 的变分由全场名次和对手强度共同决定。

存活者高于已淘汰者，淘汰越晚名次越高；同 tick 淘汰或到达时限仍存活的玩家并列。
同策略实例之间不比较，对另一策略的多个席位先取平均，再对不同策略等权。
整组循环换位共用一次更新权重；全同策略对局不变分、不推进定级。
`k_factor` 和旧评分代码仅用于历史规则复现。

新计划保存 `rating_policy = gor-relative-codeforces-v9` 和 `rating_group_size`。
默认 `league rebuild` 使用每条记录保存的规则。升级现有联赛时停止服务和评测，再运行：

```bash
generals_arena league rebuild outputs/arena/native_1000 --current-policy
```

升级在一个事务中按原结算顺序重算所有历史分数，更新待执行计划与正式成绩报告。
地图、动作、结果、回放和任务进度不变；不重新模拟，不保留旧评分备份。
每局记录赛前 / 赛后分数、实际 / 预期名次；GoR 记录更新幅度，FFA 记录表现分与零和修正。
对局记录展示名次及变分。两种榜单都显示 Elo 和版本提升：

- **Elo**：当前所选模式的评分，旁边的 `±` 为 95% 估计区间半宽，悬停显示上下界。
- **▲**：当前 Elo 减去同名系列上一已注册版本的当前 Elo，按 `名称-vN` 的数字版本排序
  （例如 `model-kestrel-v10` 对比 `model-kestrel-v2`）。无上一版本、任一版本尚无已完成对局，
  或人类用户均显示 `—`。
1v1 使用四个互补指标，自我对战不计入这些指标：

- **胜率**：获胜局数 ÷ 与其他策略的对局数，和棋也计入分母。
- **和棋率**：未分胜负的局数 ÷ 与其他策略的对局数。
- **获胜步数**：仅对获胜对局的 tick 数取平均，包含对手投降；受地图和对手强度影响，不用于排名。
- **决策耗时**：每局先计算单步平均决策耗时，再对有记录的对局等权平均；不包含 Arena 和 IPC 开销。
  显示 μs / ms，缺失计时数据（例如人类、部分历史 baseline）显示 `—`，不当作 0；悬停显示样本局数。

FFA 保留以下指标：

- **胜率**：夺冠局数 ÷ 参赛局数。同策略任一席位夺冠即计一场胜利，超时局计入分母。
- **存活指数**：按最终存活名次归一化，计算为 `（总席位数 − 名次）÷（总席位数 − 1）`。
  第一名为 100%，最后一名为 0%；同 tick 淘汰或结束时仍存活的多个席位使用平均名次。
  单挑胜者为 100%、败者为 0%、平局双方为 50%，不会因败者坚持到最后一 tick 而记为 100%。
- **击杀指数**：每席位亲自攻占敌方将军的次数 ÷（总席位数 − 1），包含同策略的其他席位；
  不把被击杀者此前的击杀数转移给胜者。

存活、击杀指数均先对同一策略在每局的席位取平均，再对它参与的各局取平均，以百分比显示。
只统计已完成对局（含自我对战），没有对局时显示 `—`。这些展示指标不参与 rating 计算。
新对局结果记录每席位的 `kills`；旧对局首次启动服务时通过 env 后端重放验证并补齐击杀统计，
保存在联赛目录的 `statistics.sqlite3` 缓存中。缓存可删除重建，不修改原始对局、录像或评分。

置信区间为当前模式胜率模型的局部 Fisher 信息近似，显示 `rating ± 1.96 × 标准误`。
1v1 在原始 GoR 坐标中计算非线性导数，以 WAIT 的当前原始评分为参考，再平移到归一化榜单；FFA 使用 CF 的 400 分尺度，在均值固定为 1000 的子空间求方差，
不需要 WAIT 出场。历史退役者仍参与信息矩阵计算，避免隐藏选手破坏比较路径。
重复席位合并、同图换位分摊证据；每策略每张图的信息权重 1v1 为 1、FFA 为 0.5，保守处理多人相关性。
选手不足 6 张独立地图，或比较图无法识别其相对位置时显示 `—`；1v1 还要求 WAIT 至少 6 张独立地图。
这是模型近似，不是精确覆盖保证，不用于重排榜单，也不以人为先验缩窄区间。

### 1v1 称号

称号只按当前 1v1 展示分数划分，不参与评分或匹配。至少 6 场有效对局后授予称号；
未定级显示 `Unrated`，自我对战不推进定级。称号随当前分数升降，名称颜色仍按策略系列和版本显示，称号单独用徽标着色。
命名与门槛参考 [Codeforces 称号体系](https://codeforces.com/blog/entry/20638)和
[2018 门槛更新](https://codeforces.com/blog/entry/59228)；Arena 的 GoR 与 Codeforces 分数不代表相同实力。

| 当前 1v1 分数 | 称号 |
|---|---|
| < 1200 | Newbie |
| 1200–1399 | Pupil |
| 1400–1599 | Specialist |
| 1600–1899 | Expert |
| 1900–2099 | Candidate Master |
| 2100–2299 | Master |
| 2300–2399 | International Master |
| 2400–2599 | Grandmaster |
| 2600–2999 | International Grandmaster |
| ≥ 3000 | Legendary Grandmaster |

### 胜率矩阵

排行榜下方的矩阵随 1v1 / FFA 切换，只显示未退役算法，不显示人类玩家。
两轴均按新系列、新版本在前排列（系列按首次登记先后，版本按数字倒序）；
baseline 排在算法之后，WAIT 在最末，不随 rating 高低换位。
同系列的名称颜色随版本加深；baseline 使用白色，WAIT 使用灰色。
行代表本方、列代表对手；红色至绿色表示 0% 至 100% 的实测胜率，灰色横线表示未交手或同策略。
1v1 按胜负计算，FFA 按最终名次两两比较；平局不算胜，悬停显示交手局数与平局比例。
重复策略先对该局的席位组合取平均，再对各局等权平均；矩阵不使用 rating 预测值。

侧边评测队列显示完成任务数与累计对局数，未完成任务分别展示 1v1、FFA 进度；
异常任务置顶，已完成记录折叠展示，自动刷新时保留展开状态。

## 人类参赛

当前提供**单个人类对多个算法**，人数、地图分布、局部视野、动作边界和 Elo 规则与算法联赛一致；
人类对局隐藏底部控制栏，保留棋盘、右侧统计与缩放按钮；状态和错误显示在右侧面板。
暂不提供多个人类之间的房间或在线匹配。对手由服务器优先选择不同策略，不足时均匀补重复席位，出生座位随机；人类只占一个席位。默认每局均匀随机抽取人数，也可以指定人数。

- 点击己方格子选中，WASD / 方向键或点击相邻格排队移动；光标立即前进，可连续规划尚未占领的路径。
- 再次点击选中格或 Z 切换下一步的半兵模式；Shift + 方向键使用半兵，执行后恢复全兵。
- Q 清空队列、E 撤销最后一步、空格取消选择，G / H 选择主城。棋盘显示排队箭头，半兵箭头为黄色。
- 服务器保存最多 256 步的输入队列，每 tick 执行一步。请求带递增 ID，重试不会重复入队；网络回包不会把光标拉回。
  输入不再因显示 tick 稍旧而被拒绝。沿途遭到阻挡或失去起点时，相应动作按游戏规则失效。
- 操作依据 [generals.io 的移动队列与快捷键](https://generals.io/versions)，不改动底层战斗规则。
- 服务器以固定时钟推进（默认每 0.5 秒一次）；刷新、请求次数、关闭页面均不能暂停或加速计分。
  断线期间继续执行已收到的队列，耗尽后 WAIT；重新打开页面可返回当前对局。
- 对局结束后，人类进入对应模式的榜单，并保留可回放记录。
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
人类对局页面支持创建多个账号、切换当前账号，以及通过身份凭证恢复账号；账号列表和凭证按联赛隔离保存在当前浏览器中。
切换账号不会结束旧账号正在进行的对局，切回后可以继续。选择固定人数后，每个对手席位都可指定一个在役算法，
或保留随机匹配；允许同一算法重复出场。选择随机人数时，对手也随机匹配。
`POST /api/play` 接收 `players` 和可选的 `opponents` 数组，例如
`{"players":4,"opponents":["算法ID","算法ID",null]}`；数组长度必须为人数减一，`null` 表示随机席位。
人类、退役或违规停用选手不能作为算法对手。已有进行中对局时，此接口返回该账号当前的对局。

排行榜人数分组使用 SQLite 索引，统计报告随分数、名册或重算版本变化而更新；未变化的排行榜不会重复重建 DOM。
回放每 128 ticks 缓存一个状态快照，最多保留 32 个，反向跳转优先恢复最近的快照，仍保留每一个 tick。

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

模块划分：`league/config.py` 地图分布，`scheduler.py` 增量赛程，`mode_rating.py` GoR / Codeforces 评分，`rank_model.py` 历史评分兼容与名次证据，
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
generals-arena league leaderboard outputs/arena/native_1000 --mode 1v1
generals-arena league leaderboard outputs/arena/native_1000 --mode ffa
python arena/benchmarks/verify_league.py outputs/arena/native_1000 --workers 1
```

固定地图、种子和动作轨迹的性能检查（重复五次，输出中位数）：

```bash
python arena/benchmarks/simulation.py --repeat 5
```

默认 fixture 为 35×33、8 人，校验每步动作和状态摘要，并输出最终 RNG 指纹。
`--reference` / `--reference-arena` / `--reference-env` 分别从 Git 导出 baseline、Arena、环境版本；
`--reference-tree PATH` 可读取隔离源码快照中的三个 `src/` 目录。编译、地图初始化和最终录像压缩
均在计时区间外，`advance()` 内的观测、策略、环境、回调及逐 tick 录像记录全部计时。

2026-10-15 在 Apple M4 Pro / Python 3.13.1 / NumPy 2.3.5 上优化策略计算以外的开销：
原生融合可见性和观测字段生成，直接绑定 NumPy 数组调用原有 baseline / Arena C ABI，
减少统计快照、动作转换、环境结果转换和动作日志的重复工作。baseline 的 `policy.cpp`、
CPU 规则内核和 SDK 计时/watchdog 均未修改；观测仍独立且只读，统计仍有不可写的底层存储。
原生绑定需要 Python 开发头文件和 NumPy 自带头文件，首次编译在游戏初始化时完成。

以下为相同种子、每轮交替先后顺序的五轮中位数；每个场景独立运行，表中时间是整局推进耗时：

| 场景 | tick 数 | 优化前 | 优化后 | 提速 |
|---|---:|---:|---:|---:|
| 35×33 / 8 baseline，原有 fixture | 1,944 | 565.9 ms | 263.9 ms | 2.14× |
| 17×15 / 2 baseline | 160 | 17.69 ms | 6.78 ms | 2.61× |
| 25×25 / 4 baseline | 689 | 126.0 ms | 48.55 ms | 2.60× |
| 35×35 / 8 baseline | 2,000 | 520.0 ms | 243.7 ms | 2.13× |
| 25×25 / 2 baseline + 2 原生 WAIT，含动作 journal | 747 | 165.2 ms | 54.36 ms | 3.04× |

最后一行使用执行层 `RecordedGame`、原有原生计时/watchdog 及每 tick journal；其余使用
`simulation.py` 相同的规则层推进循环。均不含数据库、跨进程启动或最终压缩，不能当作整个联赛
的端到端提速。每轮比较完整动作/状态轨迹、最终 RNG、策略记忆/路径、比赛结果；混合局另比较
journal 的全部动作编码。原始样本、工作区快照和对照脚本位于
`outputs/arena-overhead-optimization/{paired.json,paired.log,before,compare.py}`。

验证包括原 ctypes 与新绑定的逐步差分、RNG 和路径记忆、非连续观测、64 位大整数、
运行中修改策略参数、不可写统计、1 行/1 列地图、超时与崩溃恢复。环境 183 项通过、28 项 CUDA
跳过；baseline 26 项通过；Arena 原有及新增测试累计 174 项通过；RL 54 项通过、1 项 CUDA 跳过。
三包独立 wheel 在隔离目录完成冷编译、观测对照和原生策略/baseline 混合对局。

## 全算法重新评测

停止服务和旧评测进程后，可以直接清除旧联赛对局、rating 和正式成绩报告，再为每个已注册算法安排 1000 局。
该命令**不保留旧数据备份**；保留策略工件、私有实验和人类账号。WAIT 与其他未退役算法一样安排评测并参与普通匹配。

```bash
python arena/benchmarks/reevaluate_league.py outputs/arena/native_1000 --reset --games 1000 --workers 4
# 中断后继续：省略 --reset，保留已经完成的新对局。
python arena/benchmarks/reevaluate_league.py outputs/arena/native_1000 --workers 4
```

计分的只读对比基准（使用完全相同的已记录结果，比较按时间顺序的赛前预测误差和 200 局后的变分）：

```bash
python arena/benchmarks/rating.py outputs/arena/native_1000 --output outputs/arena/native_1000/rating_validation.json
```

拆分历史评分时，先停止 Web/worker，再执行 `league rebuild 目录 --current-policy`。
该操作按原结算顺序分别重算两种模式，更新对局 Elo 变化及提交报告，不重新模拟、不保留旧分数备份，地图和回放不变。
FFA 使用均值 1000 的约束，不依赖 WAIT 出场；旧结果可以直接重算。
Web 榜单可以切换 1v1 / FFA；API 使用 `/api/leaderboard?mode=1v1` 或 `?mode=ffa`，默认 1v1。

## 保留一个 commit，重建此前的大乱斗

以下命令只生成新比赛；被保留版本不进入新对手池。其余未退役算法（包括 WAIT）按均衡目标配额交替评测，
使用现有相近 rating 配对、随机地图、循环换位和原生多局并行后端，固定总计 5,000 局 1v1 + 5,000 局 FFA。
重复运行可从持久化窗口续跑。

```bash
python arena/benchmarks/brawl_league.py outputs/arena/native_1000 \
  --preserve-commit gpt-6-razor-v5 --duels 5000 --ffa 5000 --workers 4
# 新比赛全部完成后停止 8086 服务，再安装：
python arena/benchmarks/brawl_league.py outputs/arena/native_1000 \
  --preserve-commit gpt-6-razor-v5 --duels 5000 --ffa 5000 --install
```

安装以一个事务删除其余旧正式对局，按“新大乱斗 → 保留 commit 的原顺序”重排结算顺序并重算全部 rating。
保留 commit 的地图、动作、结果、实际时间与录像不变；对局档案按结算顺序展示。
源码工件、账号和私有测试保留，旧成绩报告更新或删除；不保留旧对局备份，安装后删除新赛程的临时副本。
若生成期间正式联赛新增了对局，安装会停止，避免误删新数据。
