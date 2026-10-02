# Generals 游戏环境

2–8 人 Generals 对战，支持 Tk 桌面游玩、四种基础 AI、无界面模拟及 CPU/CUDA 批量环境。游戏环境位于 `env/`，后续强化学习代码独立放在 `rl/`。

从仓库根目录，在已有 `generals-rl` conda 环境中安装和启动：

```bash
conda run -n generals-rl python -m pip install -e ./env --no-deps --no-build-isolation
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
