# Generals 游戏环境

2–8 人 Generals 对战，支持 Tk 桌面游玩、四种基础 AI、无界面模拟及 CPU/CUDA 批量环境。游戏环境位于 `env/`，四种手写策略独立放在 `baselines/`，强化学习代码位于 `rl/`，跨策略/版本的对战与 Elo 排名位于 `arena/`。

从仓库根目录，在已有 `generals-rl` conda 环境中安装和启动：

```bash
conda run -n generals-rl python -m pip install -e ./env -e ./baselines --no-deps --no-build-isolation
conda run -n generals-rl python -m generals play --players 4 --seed 42
```

基础运行依赖 NumPy；GUI 需要 Tk 和图形桌面。CPU 批量后端使用 Linux C++/OpenMP，GPU 使用可选的 PyTorch/CUDA。

没有图形桌面时可运行 AI 模拟：

```bash
conda run -n generals-rl python -m generals simulate --players 4 --games 10 --max-ticks 20000 --seed 42
```

统一文档：[环境开发与使用说明](doc/README.md)，包含：

- [安装、不安装运行与 GUI 操作](doc/README.md#quickstart)
- [游戏规则与地图生成](doc/README.md#rules)
- [代码架构](doc/README.md#architecture)和 [CPU/GPU API](doc/README.md#api)
- [逐 tick benchmark 与 profiling](doc/README.md#benchmark)
- [当前实现的性能验证](doc/README.md#results)
- [测试与历史存档](doc/README.md#verification)

后续性能比较以每 tick 的 `env.step(action)` 加完成同步为准，排除决策时间。新测量结果写入 `outputs/`，稳定结论统一更新到上述文档。

强化学习 baseline 已在 `rl/` 实现：直接复制指定版本的 rsl_rl，以最小补丁支持离散 MoE CTS，解耦 MDP 的观测、动作、奖励与对手，并提供后续 MCTS 的整 tick 分支模拟接口。安装、训练、补丁说明和测试见 [RL 使用说明](rl/README.md)。

手写策略的接口、安装和测试见 [Baseline 使用说明](baselines/README.md)。

独立对战和 Elo 排名见 [Arena 使用说明](arena/README.md)：

```bash
python -m pip install -e ./env -e ./baselines -e ./arena
generals-arena serve outputs/arena/native_1000 --port 8086 --no-worker
```

打开 http://localhost:8086 查看已完成的 1000 局榜单、录像，或作为人类玩家参赛。
当前仅保留 `outputs/arena/native_1000` 正式联赛；新建联赛和 C++ 策略接入见 Arena 文档。
2–8 人共用一个 rating，允许重复 AI 实例。新算法默认增量评测 1000 局，C++ 研究评测串行执行。
random baseline 固定为 0 Elo，分差 500 对应 90% 的两两期望得分。
当前不评测固定尺寸的 RL 模型；RL 训练部分保持独立。

C++ 策略可以直接提交或自由测试，无需手动注册：

```bash
generals_arena commit ./heristic/my_agent/policy.cpp --name "my-model-comet-v1"
generals_arena run --player baseline/aggressive --player ./heristic/my_agent/policy.cpp \
  --size 15:35 --games 100 --output ./results/test01
```

`commit` 自动完成 1000 局正式评测，`run` 保存独立测试记录；详见 [接入文档](arena/RESEARCH.md)。
