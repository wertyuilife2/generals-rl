# Generals 游戏环境

2–8 人 Generals 对战，支持 Tk 桌面游玩、四种基础 AI、无界面模拟及 CPU/CUDA 批量环境。游戏环境位于 `env/`，四种手写策略独立放在 `baselines/`，强化学习代码位于 `rl/`，跨策略/版本的对战与 Elo 排名位于 `arena/`。

当前规则固定为 2026-10-28 核查的官网 **31.4.5 普通 1v1 / FFA**；默认使用官方随机地图。RL 暂限 1v1，训练和评估均默认在 2000 tick 截断。规则变化及验证范围见[实施记录](doc/official_rules_alignment.md)。历史模型、Elo 和性能报告对应旧规则，不能直接沿用到当前版本。

从仓库根目录，在已有 `generals-rl` conda 环境中安装和启动：

```bash
conda run -n generals-rl python -m pip install -e ./env -e ./baselines --no-deps --no-build-isolation
conda run -n generals-rl python -m generals play --players 4 --seed 42
```

基础运行依赖 NumPy；GUI 需要 Tk 和图形桌面，内置 AI 需要 Linux/macOS 与 C++17 编译器。CPU 原生后端使用 C++/OpenMP，GPU 使用可选的 PyTorch/CUDA；具体平台与依赖见下方环境文档。

没有图形桌面时可运行 AI 模拟：

```bash
conda run -n generals-rl python -m generals simulate --players 4 --games 10 --max-ticks 20000 --seed 42
```

统一文档：[环境开发与使用说明](doc/README.md)，包含：

- [安装、不安装运行与 GUI 操作](doc/README.md#quickstart)
- [游戏规则与地图生成](doc/README.md#rules)
- [代码架构](doc/README.md#architecture)和 [CPU/GPU API](doc/README.md#api)
- [逐 tick benchmark 与 profiling](doc/README.md#benchmark)
- [环境性能验证记录](doc/README.md#results)
- [测试与历史存档](doc/README.md#verification)

后续性能比较以每 tick 的 `env.step(action)` 加完成同步为准，排除决策时间。新测量结果写入 `outputs/`，稳定结论统一更新到上述文档。

强化学习实现位于 `rl/`：默认采用 F03 消融选出的 ViT 1M dense、H4/K11 与 recent8 历史对手池，配合共享自适应学习率和 gamma 课程。包内 `generals_rl/default.json` 提供配置，`python -m generals_rl.train` 默认训练 224 轮，无需实验调度工具；单卡和八卡复现、CPU 开发覆盖及历史 LightCNN 对照见 [RL 使用说明](rl/README.md)。正式结果见 [H200 scaling](doc/h200_scaling.md) 和 [F03 消融](doc/f03_ablation.md)，完整历史见 [Auto research 记录](doc/auto_research.md)。

手写策略的接口、安装和测试见 [Baseline 使用说明](baselines/README.md)。

独立对战和 Elo 排名见 [Arena 使用说明](arena/README.md)：

```bash
python -m pip install -e ./env -e ./baselines -e ./arena
generals-arena league init outputs/arena/main --config arena/configs/league.json
generals-arena serve outputs/arena/main --port 8086 --no-worker
```

打开 http://localhost:8086 查看榜单、录像或参加人类对局。新建联赛尚无比赛记录，需要先按 Arena 文档登记对手并运行评测。
`outputs/arena/native_1000` 是使用旧源码和评分规则的本地历史结果，不是上述命令生成的数据。
评分按人数分成独立的 1v1 GoR 与 FFA Codeforces 两池，允许重复 AI 实例。新算法默认增量评测 1000 局，C++ 研究评测支持多进程单局并行。
1v1 展示相对 WAIT 的 GoR；FFA 保持全池均值 1000。
固定尺寸 RL 模型使用兼容的联赛配置；导出的策略包负责特征、记忆和推理，Arena 只管理通用包和比赛。
Arena 的循环赛、持续联赛和训练评估共用比赛执行、结果、回放与评分基础。
训练原子保存 checkpoint、导出策略包，调用 Arena 的 CPU/GPU 批量评估。当前默认使用官方随机地图上的 baseline＋WAIT 面板；最终候选相互对战后共同评分，并输出胜率矩阵与 best。旧 9×9 校准池不再作为默认，本轮不运行新校准实验；
部署与共享流程见 [RL 文档](rl/README.md#模型导出与-arena-评估)。公开榜单和训练用相对 Elo 分开报告。

C++ 策略可以直接提交或自由测试，无需手动注册：

```bash
generals-arena commit ./heristic/my_agent/policy.cpp --name "my-model-comet-v1"
generals-arena run --player baseline/aggressive --player ./heristic/my_agent/policy.cpp \
  --size 15:35 --games 100 --output ./results/test01
```

`commit` 自动完成 1000 局正式评测，`run` 保存独立测试记录；详见 [接入文档](arena/RESEARCH.md)。
