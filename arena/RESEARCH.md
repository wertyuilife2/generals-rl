# C++ 策略接入与比赛

Arena 提供两个主要命令：`commit` 正式参赛，`run` 自由测试。
源码快照、编译缓存、版本、赛程、计时、评分和录像均由 Arena 管理，无需创建研究会话或手写 entrant JSON。

在项目根目录运行：

```bash
source .venv/bin/activate
python -m pip install -e ./arena --no-deps --no-build-isolation
```

`generals_arena`、`generals-arena` 和 `python -m generals_arena` 等价。
默认正式联赛为 `outputs/arena/native_1000`，需要其他联赛时添加 `--league 目录`。
默认目录须已存在；新的独立联赛可由 `league init` 创建。

## 提交到正式榜单

```bash
generals_arena commit ./heristic/my_agent/policy.cpp --name "my-model-comet-v1"

# 多文件策略：目录内的所有 .cpp 联合编译
generals_arena commit ./heristic/my_agent --name "my-model-comet-v2"
```

`--name` 是正式榜单的显示别名，允许空格、中文及模型名中的点号，长度为 1–128 字符。
别名不改变源码身份。省略时使用源码文件/目录名加版本哈希前缀；重复提交同一版本且省略别名会沿用原名。
别名与版本一旦关联就固定：同名不同版本、同版本换别名均明确拒绝，避免重复上榜或覆盖已有记录。
建议 autoresearch 使用 `<模型名>-<模型自行选择的单词>-v<版本号>`，例如 `my-model-comet-v1`。

Arena 自动快照和编译源码，执行 8 局接口预检，通过后登记选手并执行 **500 局 1v1 + 500 局 FFA 正式比赛**（默认最多 4 局并行）。
正式比赛沿用联赛的 2–8 人随机地图、自适应配对和两套独立的名次 rating（1v1 / FFA）；不能通过 `commit` 改小局数或修改正式规则。
FFA 优先使用不同策略，策略不足时才均匀增加重复席位。两套配额分别保存。
中断后重新执行同一命令继续原任务，已完成的局不会重复计分。完成后再次提交返回该次完成时保存的成绩。
如果源码或固定参数变化，会生成新版本；显式指定过别名时，给新版本使用新别名。

可选 `--params 参数.json`，内容必须为 JSON 对象，计入版本身份并传给 `arena_create`。
Arena 本身不限制提交次数；60 分钟、最多三次全量评测是具体研究任务的约束，由研究者遵守。
单个版本有已记录的非决策超时违规时，不能通过换名或重试清除记录，修复实现后才能提交新版本。

命令在 stderr 输出进度，在 stdout 输出 JSON。成功要求 `ok=true`、`completed_games=1000`、
`job.modes` 中两种模式均完成 500 局，且 `errors=0`。决策超时按投降计分。报告包含 `name`、`version`、
`ratings`、`ranks`、`leaderboards`（均按 `1v1` / `ffa` 分组）、`publication_games_by_mode`、完整榜单、计时、错误、`report` 路径和联赛路径；预检失败也保存报告及部分录像。

```bash
generals_arena status
generals_arena league leaderboard outputs/arena/native_1000 --mode 1v1
generals_arena league leaderboard outputs/arena/native_1000 --mode ffa
```

现有网页 http://localhost:8086 无需重启，直接显示新增选手、对局与评分。
网页使用 `--no-worker` 与提交命令配合，正式任务由 `commit` 执行：

```bash
generals_arena serve outputs/arena/native_1000 --port 8086 --no-worker
```

## 自由测试

```bash
generals_arena run \
  --player baseline/aggressive \
  --player ./heristic/my_agent/policy.cpp \
  --player ./heristic/my_agent/policy.cpp \
  --size 15:35 --games 100 --seed 42 --output ./results/test01
```

每个 `--player` 对应一个席位，必须给出 2–8 个，允许相同策略重复，每席位使用独立实例和随机种子。
每局都包含全部指定玩家。第一局的席位按参数顺序排列；同一地图循环轮换席位，之后生成新地图。
最后一组按局数截断。自由测试使用独立数据库，不修改正式榜单或增添正式参赛者。

支持的玩家选择方式：

| 选择方式 | 含义 |
| --- | --- |
| `baseline/random`、`baseline/aggressive`、`baseline/expansion`、`baseline/defensive` | 内置黑盒 baseline；也接受 `baseline/aggressive.cpp` 写法，它是逻辑名称 |
| `/path/to/agent.cpp` | 自动快照并编译该文件及本地 `.h/.hpp` 依赖 |
| `/path/to/source_dir` | 自动快照并联合编译目录中的所有 `.cpp` |
| `policy/<完整版本哈希>` | Arena 已保存的不可变策略，适合与历史自身版本比较 |
| `@正式榜单别名` | 已提交的 C++ 策略版本，可包含空格，必要时给整个参数加引号 |

不需要阅读 baseline 源码。一次 `run` 已经包含快照和编译步骤，无需提前注册。
`--params` 将同一个固定 JSON 参数对象传给本次所有以源码路径指定的玩家；
`policy/` 与 `@别名` 始终使用其已快照的参数。

| 参数 | 默认与含义 |
| --- | --- |
| `--size auto` | 默认值，按人数使用联赛配置；新默认为官方分布，1v1 为 18–23，8 人为 25–33 |
| `--size 20x25` | 固定宽 20、高 25 |
| `--size 15:35` | 宽、高分别在 15–35 内均匀随机，与人数缩放无关 |
| `--games 100` | 总局数，默认 1 |
| `--seed 42` | 固定地图及策略随机种子；默认 42 |
| `--max-ticks 3000` | 自由测试的回合上限；默认继承正式联赛 |
| `--output 目录` | 可选导出报告和含完整录像的 SQLite 数据库；不改变 Arena 自己的存储位置 |

显式指定尺寸时，每边支持 3–64，但仍须满足联赛的出生间距等约束；过小尺寸会给出明确错误，不会自动放宽约束。正式评测继承联赛记录的地图配置。
相同人数、地图设置、seed 和局数下，替换算法不改变地图及席位随机数。
同一版本、同一请求重复运行会继续或读取原对局；需要新的独立样本时改变 `--seed`。
导出目标必须为空，或属于同一 run；不会覆盖别的实验和正式数据库。

stdout 返回 JSON 报告，包含每种策略按人数的平均名次、夺冠率、跨策略两两名次得分、
每步决策耗时及违规记录。重复席位归入同一种策略，内部比较不计入两两得分；
纯同版本自我对战的该项为 null，不代表实力高低。

导出目录包含 `report.json`、`league.sqlite3` 和来源标记；可直接查看回放：

```bash
generals_arena serve ./results/test01 --port 8087 --no-worker
```

## Arena 管理的数据

所有权属于所选联赛，不依赖开发工作区：

```text
outputs/arena/native_1000/
  league.sqlite3                 正式榜单、任务、对局与录像
  policies/<version>/            不可变源码、参数、SDK、编译产物
  runs/<run_id>/                  自由测试和预检的请求、报告、对局数据库
  commits/<version>/
    commit.json                  别名、正式任务 ID、版本
    publication.json             本次全量评测完成时的成绩
```

源码、参数、SDK 和编译环境共同决定版本身份；路径与别名不用于判断是否重复提交。
同一文件的不同版本独立保存，移动或修改工作区不影响历史比赛和已发布对手。
失败记录与部分录像保留，失败局不更新评分。普通中断可恢复；已违规版本不能恢复成正常版本。
`--output` 是导出副本，删除它不会删除 Arena 管理的原始数据。

## C++ 协议

完整定义见 [arena_policy.h](src/generals_arena/sdk/arena_policy.h)，接口示例见
[policy.cpp](examples/cpp_wait/policy.cpp)。可传入任意命名的 `.cpp` 文件，Arena 会递归收集其本地 `.h/.hpp` 引用；
如有其他参与链接的 `.cpp`，提交整个目录。目录中的 `.cpp/.h/.hpp` 被一起快照，所有 `.cpp` 参与编译。`arena_policy.h` 在编译时自动提供；本地有副本时必须与 SDK 一致，不能修改。
本接口不接受额外链接参数或外部策略实现；运行时所需参数通过登记的 JSON 对象传入。

```cpp
#include "arena_policy.h"

extern "C" void *arena_create(int player, std::uint64_t seed, const char *params_json);
extern "C" std::int64_t arena_act(void *instance, const ArenaObservation *observation);
extern "C" void arena_destroy(void *instance);
```

同一策略的所有 `.cpp` 联合编译为共享库；同一场比赛的所有策略库在一个进程中加载，
与 env CPU 后端一起顺序执行，每个席位拥有独立实例和随机状态。策略调用是直接 C ABI 调用，
没有逐决策 IPC、策略进程启停或策略线程。创建时只有席位、策略种子和固定参数，没有地图种子或完整棋盘。
创建过程最多 2 秒，在比赛开始前完成。进程关闭由 Arena 管理，即使析构函数挂起也能终止。
可变策略状态必须保存在各自实例中，不能通过全局变量在席位间共享局内信息；全局数据只存放只读常量。
`params_json` 仅在 `arena_create` 调用期间有效，需要保留时应复制。
决策时传入行优先只读数组，指针仅在本次调用期间有效：

| 字段 | 含义 |
| --- | --- |
| width / height / players / player / tick | 地图尺寸、人数、本席位和当前 tick |
| terrain / structure / owner / army / visible | 当前合法局部观测，共 width × height 格 |
| alive | 所有席位是否存活 |
| total_army / total_land / total_cities | 公开统计；敌方城市数为 −1 |
| general_pos | 自己主城的位置 |
| previous | 本席位上次动作结果；首次为 nullptr，反馈在下次计时决策内交付 |

未知地形/结构为 255、未知归属为 −3、未知兵力为 −1；空地归属 −1、中立城市 −2。
地形 0/1 为平地/山；结构 0/1/2 为普通地/城市/主城。
动作返回值：WAIT=0；移动为 `1 + source*8 + direction*2 + half`。
方向上/右/下/左对应 0/1/2/3；half 为 0 或 1。头文件提供 `arena_move()`。

本地裁判规则见 `doc/README.md` 的战斗、地图、观测章节以及 `env/src/generals/core/engine.py`。
Arena 是先共同决策，再由 env 进行增兵及按席位顺序执行动作。
多人名次由淘汰先后决定；超时存活者并列，兵力不用于打破该并列。
不用阅读 `baselines/` 中的任何实现即可使用本接口。

## 10ms 决策约束

每个通过此接口登记的 C++ 策略均由同一个 Arena 适配器执行，适用于私有评测、正式比赛和真人对战。
策略必须单线程，不能在两次调用之间后台计算。`commit` / `run` 固定使用一个 worker，不提供并行参数。
同场所有策略与 env CPU 引擎在一个比赛进程内直接调用，env 的单局 CPU 适配器也使用一个线程。
10ms 使用 C++ 单调墙钟衡量决策及其原生保护调用开销，不计 Python 构造观测、IPC、
地图生成、游戏引擎更新或赛前创建。观测数组通过指针传入，无逐决策序列化。

**单次超过 10,000,000ns 即失败，没有平均值豁免或宽限。** 原生适配器使用操作系统信号计时器，
不创建 watchdog 线程。返回时也复核耗时，拒绝晚到动作。死循环、超时或原生崩溃会终止当前比赛进程；
不会替换成 WAIT 或自动记为输局。策略创建与销毁各有 2 秒保护上限。

Arena 的外层协调器仅管理比赛任务与持久化，一个串行比赛进程运行整场对局并跨局复用。
算法之间没有进程通信。机器人比赛只在整局结束时返回结果；真人比赛按游戏 tick 同步状态，
这些通信都在决策计时之外。共享动作日志使协调器能在进程异常退出后恢复已执行动作和部分录像，
违规不会拖垮网页服务。该进程隔离用于容纳整场比赛的故障，不是逐席位执行架构。

失败局保留部分录像和违规席位/tick/耗时，不更新 rating，任务标记失败；
C++ 失败任务不能用普通 `league run --retry` 洗掉。违规 C++ 选手保留榜单与历史，
但不再进入新生成的正式或真人赛程；已持久化的赛程不会被悄悄改写。

普通操作系统不是硬实时系统，不能承诺信号恰好在第 10ms 被调度。
C++ 决策本身的调度延迟仍计入其墙钟预算；已知超时不会被接受。
内置 baseline 使用现有实现；新 C++ 策略及其历史版本使用上述统一边界。
这是可信本地代码接口，不是安全沙箱。策略不能修改信号处理/计时器、创建线程或子进程、
通过文件或网络进行额外决策，也不能读取其他席位的状态或隐藏信息。


CLI 成功退出 0，已执行但验收失败退出 1，参数、编译或登记错误退出 2。
