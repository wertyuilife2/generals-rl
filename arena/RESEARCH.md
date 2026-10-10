# C++ 策略研究接口

Arena 负责源码快照、编译、黑盒对战、计时、录像、版本登记和正式发布。
研究者只提交 C++ 策略，不需要 Python 适配器、比赛循环或评分工具。
完整任务提示词见 [AUTORESEARCH_PROMPT.txt](AUTORESEARCH_PROMPT.txt)。

## 创建研究会话

在项目根目录运行，使用现有 `.venv`。Linux/macOS 需要 C++17 编译器；`CXX` 可以指定编译器。
正式联赛须已存在且与当前游戏规则匹配。下面的目录名可换为本次研究唯一名称：

```bash
.venv/bin/python -m generals_arena research init outputs/autoresearch/run01 \
  --league outputs/arena/native_1000 --workspace heuristics/run01
```

该命令创建研究会话和一个仅返回 WAIT 的 C++ 接口示例，同时导出 `arena_policy.h`。
已有源码和研究会话不会被覆盖。空实现用于接线验证，不是竞争策略。
会话记录正式联赛和规则身份，最多允许 3 次正式发布，每次固定 1000 局。
60 分钟研究任务的墙钟预算由研究者管理，此服务不启动 60 分钟计时任务。

## C++ 协议

完整定义见 [arena_policy.h](src/generals_arena/sdk/arena_policy.h)，接口示例见
[policy.cpp](examples/cpp_wait/policy.cpp)。策略源目录必须包含 `policy.cpp`；
所有 `.cpp/.h/.hpp` 被一起快照，所有 `.cpp` 参与编译。`arena_policy.h` 是 Arena 所有的协议，不能修改。
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

## 注册不可变候选

```bash
.venv/bin/python -m generals_arena research register outputs/autoresearch/run01 \
  --source heuristics/run01 --name v001
```

可加 `--params parameters.json`。参数必须是 JSON 对象，原文的规范化形式交给 `arena_create`。
登记由 Arena 快照源码、SDK、参数、编译器信息和编译后的共享库，并自动生成入口及版本哈希。
修改工作区不影响已经注册的候选；相同名字和版本重复注册幂等，更换实现必须换名字。
候选目录保存可读源码、构建元数据、`policy.zip`、`entrant.json` 和登记身份。
加载时校验工件；正式发布还会将包复制到正式联赛的 `policies/`，不依赖开发工作区。
二进制工件绑定登记时的平台；包内含源码，可在其他平台重新登记，但不是同一二进制身份。

## 私有黑盒评测

```bash
# 默认对手池为四种 baseline，默认人数继承正式联赛
.venv/bin/python -m generals_arena research evaluate outputs/autoresearch/run01 \
  --candidate v001 --games 128 --seed 42 --label experiment01

# 可指定 baseline、self 或本会话已登记的任意历史候选；允许重复
.venv/bin/python -m generals_arena research evaluate outputs/autoresearch/run01 \
  --candidate v002 --opponents baseline:aggressive baseline:random v001 self \
  --games 128 --seed 43 --players 2 3 4 5 6 7 8 --label experiment02

# 纯自我对战
.venv/bin/python -m generals_arena research evaluate outputs/autoresearch/run01 \
  --candidate v002 --opponents self --games 32 --seed 44 --label selfplay01
```

每局包含候选，对手池有放回抽样。Arena 生成随机地图和成组座位轮换；相同配置、
种子、对手列表顺序与局数下，替换候选不会改变地图、策略种子或座位轮换。
末组允许截断以满足准确局数。这里使用固定赛程；正式发布继续使用原联赛的自适应配对。
已有实验标签不能代表其他请求；同一命令可恢复被中断但未失败的实验，完成后不会重复比赛。

输出 JSON 报告并保存到 `evaluations/<label>/report.json`。报告包含完成局数、
按人数统计的名次、夺冠率、两两名次得分、决策调用数、最大耗时和违规详情。
两两得分为名次更好 1、并列 0.5、更差 0；排除同候选实例之间的比较，
先平均到每局，再对各人数均值等权平均。未出现的人数不进入宏平均；覆盖范围列在报告中。
纯同版本自我对战的两两得分为 null，不能据此声称实力增长。
报告中的 game_win_rate 是候选任意实例夺冠的局比例；重复实例会影响该指标。

每次实验拥有独立 `evaluations/<label>/league/league.sqlite3`，保存赛程、结果、完整录像及失败部分录像。
不向正式联赛登记候选，也不改变正式分数。可以直接复用网页查看私有实验：

```bash
.venv/bin/python -m generals_arena serve \
  outputs/autoresearch/run01/evaluations/experiment01/league --port 8003 --no-worker
```

## 10ms 决策约束

每个通过此接口登记的 C++ 策略均由同一个 Arena 适配器执行，适用于私有评测、正式比赛和真人对战。
策略必须单线程，不能在两次调用之间后台计算。`workers` 固定为 1；传入其他值会被拒绝。
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

## 正式发布和查询

```bash
.venv/bin/python -m generals_arena research publish outputs/autoresearch/run01 \
  --candidate v002
.venv/bin/python -m generals_arena research status outputs/autoresearch/run01
```

`publish` 自动进行 8 局正式规则下的接口预检，然后冻结并登记到目标联赛，执行该 checkpoint 的 1000 局任务。
最多 3 个发布名额；相同候选再次调用恢复同一任务，不增加名额、不重复计分。
相同源码/参数版本不能用不同名字重复占位。预检通过后开始正式发布即占名额；
之后中断或失败仍保留该名额，预检失败不占名额。
候选已有记录的自身违规会阻止发布，包括作为其他实验的对手时发生的违规，修复后必须注册新版本。
发布只处理自己的任务，不代跑其他待评测选手；不能同时运行第二个正式联赛 worker。

完整发布要求报告 `ok=true`、`completed_games=1000`，并且超时和错误均为 0。
异常退出保留进度；违规失败保留真实记录，不宣称完成。
`strategies/<candidate>/publication.json` 包含完成情况、计时与榜单；原始正式记录由联赛数据库保存。
网页 `http://127.0.0.1:8002` 使用现有正式服务，必要时运行：

```bash
.venv/bin/python -m generals_arena serve outputs/arena/native_1000 --port 8002 --no-worker
```

CLI 成功退出 0，完成但验收失败退出 1，参数/编译/登记错误退出 2。
Python 也可调用 `generals_arena.research` 的 `initialize/register/evaluate/publish/status`，
返回与 CLI 相同的结构化数据；研究者不需要编写比赛或计时循环。
