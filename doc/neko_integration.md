# neko-dev 选择性集成记录

日期：2026-10-29。基线 `main = 8f746238ec5531bcccda7847df9f5e2773bca040`；
来源 `origin/neko-dev = 54145062716ab4e59e30d2a805c2e1218ad29da9`。
在独立 worktree 的 `integrate/neko-dev` 分支操作，保留原工作区未提交的 RL 开发。

## 集成内容

| 内容 | 落点 | 处理方式 |
|---|---|---|
| Kestrel v7 | `heristic/gpt-6-kestrel-v7/` | 加权路线策略，保留源码和参数 |
| Razor v7、实验性 v8 | `heristic/gpt-6-razor/` | 保留两个规划实现；不引入中间版本与 trial 档案 |
| C++ 策略调用加速 | `arena/.../policies/host.cpp` | CPython/NumPy 桥接；适配 ABI 2 全部观察字段，维持计时与错误归属 |
| 临时缓存恢复 | `arena/.../policies/native.py` | 缓存完全或部分被清理后可重建；参数和库校验；并发修复串行化 |
| 策略退役/恢复 | `arena/.../competitions/` | 排除未来配对、默认隐藏榜单，保留历史评分和对局；落库时检查过期赛程 |
| 基础设施失败重试 | 同上 | 无策略责任归属的 C++ worker 失败可以重试；原尝试、种子、席位保留 |
| 存活指数修正 | `competitions/leaderboard.py`、网页 | 按当前环境的官方最终名次归一化，支持击杀优先和并列，不改 Elo 算法 |
| 压缩回放缓存 | `arena/.../web/replays.py` | 32 MiB 压缩字节上限；同一回放并发验证合并，内容变化失效 |
| 官方线上接入 | `env/.../backends/online/`、`runtime/online.py`、`arena/.../matches/online.py` | 服务端观察/动作转换、专用账号、顺序多局及记录；复用通用策略包入口 |

上游以旧 ABI 和规则实现为基础，因此采用选择性移植并记录来源，不把整个源分支标记为已合并。
CPU/CUDA 离线内核、统一比赛执行器和 RL 代码保持基线实现。主分支当前规则仍为
`generals.io-31.4.5-classic-v1`，没有引入另一套离线规则。

沿用主线已有的源码指纹校验：本次运行时变更会使旧联赛的执行签名及旧校准 profile 失效。
后续评测应初始化新联赛并重新校准；旧数据保留，可用对应归档源码查看/继续运行。
本次没有引入旧分支的自动 runtime migration，也没有修改任何现有联赛数据。

策略移植修正了即时可移动兵数、Razor 的独立迷雾障碍字段，以及交换将军后的已知将军位置和路径几何。
Razor 部分增长预测已补上 50 tick 全图增长与建筑增长叠加。前瞻搜索仍保留策略原有的启发式近似，
包括部分按席位近似的回复顺序，不应解释为官方行动优先级的精确模拟。v8 仅作为实验策略，未改为默认。
所有旧分支 Elo/胜率都不能当作当前规则下的评分。

线上接入按当前公开客户端修正主站字段，传递 ABI 2 的迷雾、击杀和离场提示。
Arena 加载普通 C++ 策略或 opaque policy package，RL 特征和模型推理仍由包内代码负责；
没有新增特定模型版本适配器。线上模式不伪造移动回执，不写入离线 Elo 池。
此次只有离线消息测试，没有真实注册或线上对局验证。用法见[线上接入说明](../env/src/generals/backends/online/README.md)。

## 未引入的内容

| 内容 | 原因 |
|---|---|
| 旧 league/tournament/research/runtime compatibility 目录 | 当前 Arena 已统一这些能力，不恢复重复架构和旧导入入口 |
| 旧地图、迷雾、结算实现 | 主线已具备更新的 31.4.5 CPU/CUDA/GUI/RL 对齐版本 |
| 单独的旧 CPU match 运行器、旧 observation 扩展 | 当前执行器和批量观察接口已覆盖，避免再维护另一条路径 |
| 旧评分与人类对局管理整体实现 | 保留主线规则，只移植缺失的退役、统计与缓存改进 |
| Kestrel v5/v6、Razor v1–v6、试验候选与报告存档 | 保留代表性的新策略；仅 Razor 实验目录就有 1502 个文件、约 29.2 MiB |
| MPS 后端 | 上游建立在旧状态/规则协议上；本机没有 Apple GPU，无法验证功能和性能，不直接接入 |
| 模型权重、账户、线上记录、RL 实验修改 | 不属于本次集成 |

## 验证

测试使用 conda `generals-rl`，通过当前 worktree 的 `PYTHONPATH` 导入；没有把系统 editable 安装切到 worktree。
可选线上依赖安装到了该 conda 环境。测试结果以最终日志为准：

| 项目 | 结果 |
|---|---|
| Arena 全量测试 | 238 passed，2 subtests |
| env 全量测试（含 CPU/CUDA 对照和官方回放） | 249 passed，1521 subtests；7 项 GUI 因无桌面显示跳过 |
| RL 官方策略、导出、Arena 注册及联赛接入 | 29 passed |
| 网页 JavaScript | 16 passed |
| env / Arena wheel | 构建通过，包含新增线上包和原生 host.cpp |

原有 Kestrel v4 + aggressive 基线，种子 42/43，各运行 1v1 与 8 人 FFA，1000 tick 上限：
合入前后 4 局的初始/最终状态哈希、胜者、结束 tick、结束原因、名次、击杀、兵力/领土统计及无效动作数完全一致。

新策略每个版本各跑 2 局 1v1 和 2 局 8 人 FFA（其他席位为 aggressive，交替首/末席位），种子 42/43，1000 tick 上限：

| 策略 | 1v1 胜/负/截断 | FFA 胜/负/截断 | 最大单次原生决策耗时 |
|---|---:|---:|---:|
| Kestrel v7 | 2 / 0 / 0 | 1 / 1 / 0 | 0.219 ms |
| Razor v7 | 2 / 0 / 0 | 1 / 0 / 1 | 0.900 ms |
| Razor v8（实验） | 2 / 0 / 0 | 2 / 0 / 0 | 1.968 ms |

12 局均无策略错误或决策超时。这是接口及运行验证，样本量不足以评价强弱或给出可信 Elo。

原生调用微基准：Intel i7-14700KF，Python 3.12，固定 25×25/8 人观察，C++ WAIT 策略，
包含 previous move；预热 1000 次后，7 轮各 20000 次 `NativeController.act`，比较中位数：

| 主线 ctypes 桥接 | 新 CPython 桥接 | 加速比 |
|---:|---:|---:|
| 27.475 µs/调用 | 10.615 µs/调用 | 2.59× |

这是控制器调用开销的微基准，不代表复杂策略整场评估或 RL 训练也能快 2.59 倍。
测量不包含首次编译、加载或地图生成，双方使用相同观察和策略。
运行日志、微基准、原始对局结果保留在 worktree 的 `outputs/validation/`，不提交实验产物。
