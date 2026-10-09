# 共享 CPU 后端与 C++ baseline：1000 局验证

日期：2026-10-10。当前实现复用 `env/src/generals/backends/cpu/step.cpp`，
Arena 不维护另一套规则后端。四种 baseline 的决策迁入 `baselines/src/generals_baselines/policy.cpp`。
RL 源码、训练流程及 CUDA 内核未修改。

## 实现边界

- `generals.make_engine(state)` 在 env 内以单环境 `CPUEnv` 适配 `Action/TickResult`，
  状态数组借用后端存储，不逐 tick 复制整张地图；原有 C++ CPU 单步内核没有修改。
- Arena 的算法局、真人局、录像重演与固定地图比赛均使用这个入口。
  Python `CoreEngine` 留在 env 中作为规则参考与事务接口。
- baseline 的记忆、候选评分、最近目标、BFS、路径跟踪、半兵选择与随机动作评分在 C++ 中执行。
  Python 保留控制器协议、数组校验、生命周期和每席位独立的 NumPy BitGenerator。
  C++ 直接调用 BitGenerator 的原生随机函数，保留洗牌及抽样顺序。
- 支持 Linux/macOS 的源码编译缓存；macOS 使用 clang/libomp，并处理与可选 Torch 共存的
  OpenMP 动态库加载。使用 CPU 和 baseline 不会导入 Torch。
- 算法身份和环境身份包含 C++ 源码哈希；旧实验用保存的原代码读取、运行。

## 完整对局性能

本机 macOS 15.7.7 arm64、14 逻辑 CPU、Python 3.13.1、NumPy 2.3.5、Apple clang 17。
固定八人 35×33 地图、种子、1944 tick，分别重复五次取中位数；测量时没有并行运行其他基准或测试。
每次校验全部动作及逐 tick 状态摘要，完整录像 SHA256 均为
`9d7f3f4a21c338a28a86b234361b6f5cc47f62348e30b613b44703fdcd7b5ede`。

| 实现 | 完整推进耗时 | 吞吐 |
| --- | ---: | ---: |
| `e8b9ec8`：Python 参考环境与 NumPy baseline | 2.447 秒 | 794 tick/s |
| 共享原生 CPU 环境与 C++ baseline | 0.714 秒 | 2724 tick/s |

**加速 3.43 倍，耗时减少约 71%。** 计时包含 AI、观测构造、结果回调和逐 tick 记录，
不含初始化、首次编译、录像压缩或数据库写入。原始五次样本保存在
`outputs/arena/performance/cpp-full-before.json` 与 `cpp-full-after.json`。
这些数字对应本机与固定样本，不代表任意算法和地图的固定加速比。

```bash
python arena/benchmarks/simulation.py --fixture arena/benchmarks/eight_player_35.json --reference e8b9ec8 --reference-arena e8b9ec8 --repeat 5
python arena/benchmarks/simulation.py --fixture arena/benchmarks/eight_player_35.json --repeat 5
```

## CPU/OpenMP 的复用评估

迁移 baseline 前，保持原 Python baseline，仅将环境推进接到原有 CPU 内核。
从上一轮联赛分别选择 2、4、8 人中等局长样本，五次中位数为：

| 人数 / 地图 | Python 环境 | 原生环境适配 | 加速 |
| --- | ---: | ---: | ---: |
| 2 / 20×19 | 0.162 秒 | 0.152 秒 | 1.07 倍 |
| 4 / 25×22 | 0.550 秒 | 0.518 秒 | 1.06 倍 |
| 8 / 30×33 | 2.157 秒 | 2.099 秒 | 1.03 倍 |

只换环境的端到端收益有限，baseline 决策仍占主要时间。
单独重放已准备好的 2770 tick 动作、不包含 AI 和录像时，单环境推进从 0.137 秒降至 0.008 秒；
这个约 17 倍的数字是推进微基准，不能作为完整比赛加速比。

同一形状的克隆环境批次中，B=8 时 OpenMP 1/2/4/8 线程分别用时
0.009/0.034/0.051/0.101 秒；B=512 时 2 线程比 1 线程快约 1.33 倍。
所以 Arena 当前每局设 `threads=1`，用 8 个 worker 处理不同尺寸、人数和局长的比赛。
它执行的是 env 的 OpenMP 编译内核，单局没有额外启动多线程。

迁移前的完整记录为 `outputs/arena/performance/cpu-reuse-final.json`，当时使用临时 macOS 加载器。
该加载器已删除；保留的 `arena/benchmarks/cpu_reuse.py` 现在通过正式 env 接口测量，
使用当前 C++ baseline，重新运行不会再得到上表的旧 Python baseline 耗时。

## 1000 局落盘结果

新目录：`outputs/arena/native_1000`。读取 `outputs/arena/optimized_1000` 中的原始
1000 个算法赛程，固定地图、人数、席位、策略种子和计分顺序，用新源码身份创建独立联赛。
没有覆盖旧录像、人类账号或旧分数。

最终版本使用 8 workers，执行并持久化 1000 局耗时 **58.86 秒**，累计 **1,584,005 tick**。
该计时包含 worker 启动、模拟、录像压缩、数据库提交和评分，排除赛程复制与事后校验。
上一轮自适应调度运行用时 194.63 秒；由于本轮预先固定赛程，不将两者商值作为严格的同流程加速比。
完整对局的可控性能对照见上面的五次基准。

逐局解压录像，校验所有动作和状态摘要，并比较最终统计、淘汰 tick、无效动作与胜者。
**1000 局全部一致，按原顺序重新计算的所有 rating、出场数、胜负平也全部一致。**
地图宽高均在 15–35 之间，允许重复算法与 2–8 人混合对局；任务错误为 0，SQLite 完整性检查通过。

| 算法 | Rating |
| --- | ---: |
| aggressive-v1 | 495.5 |
| expansion-v1 | 220.2 |
| random | 0.0 |
| defensive-v1 | −58.7 |

产物：`verification.json`、`leaderboard.json`、`schedule_source.json`、`run.log`、
`league.sqlite3` 均在新目录；数据库包含所有录像。

```bash
# 目标必须是尚未存在的联赛；仅用于无真人计分、按任务顺序完成的 baseline 赛程
python arena/benchmarks/reproduce_league.py outputs/arena/optimized_1000 outputs/arena/native_1000 --workers 8

# 当前新联赛的榜单、对局记录、回放与真人游戏
generals-arena serve outputs/arena/native_1000 --port 8002 --no-worker

# 只验证旧录像中的策略动作，不向旧联赛写入新结果
python arena/benchmarks/verify_policies.py outputs/arena/optimized_1000 --workers 8 --output /tmp/policy-verification.json
```

8000 与 8001 的旧服务分别使用 `outputs/arena/runtime-a73a024`、`runtime-e8b9ec8` 中保存的代码。
8002 使用当前共享原生后端与 C++ 策略。

## 回归与限制

- `pytest env/tests --ignore=env/tests/test_gui.py -q`：122 通过、28 跳过、738 子测试通过。
- `pytest arena/tests baselines/tests -q`：75 通过、15 子测试通过，包含 HTTP、真人队列与多进程任务测试。
- 原生适配覆盖非法动作、超大整数、同 tick 淘汰、继承、统计快照与溢出报告。
  C++ baseline 覆盖观测隔离、非连续数组、独立 RNG、重置与关闭生命周期。
- env 与 baselines wheel 构建成功，核对包内包含 `step.cpp` 和 `policy.cpp`。
- 本机没有 CUDA 设备，相关测试跳过；完整 env 测试中的 Tk 桌面初始化导致进程中止，
  因而重跑时排除该文件，没有将桌面 GUI 标为通过。此轮没有测量浏览器 FPS。

CPU 批量接口依旧要求同批次宽、高、人数相同、每图最多 4096 格；Arena 当前单局适配满足此限制。
Windows 原生编译暂不支持。跨平台浮点数学库或 NumPy 版本变化仍需重新做轨迹校验，
本报告的逐步一致性结论对应上述测试环境。
