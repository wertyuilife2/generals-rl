# Generals CPU 原型

可独立游玩的 Python/NumPy 游戏环境，包含 Tk 桌面 GUI、四种基础 AI、局部视野和无界面高速模拟。游戏环境在 `env/`，后续强化学习代码在 `rl/`；环境不依赖 RL、PyTorch 或 CUDA。

## 安装与启动

在已有 `generals-rl` conda 环境中，从仓库根目录运行：

```console
conda run -n generals-rl python -m pip install -e ./env --no-deps --no-build-isolation
conda run -n generals-rl python -m generals_env play --players 4 --seed 42
```

需要安装 NumPy、Tk 和可用的桌面显示服务。Windows 和 Linux 使用同一个 Python 模块入口；本次已在 Linux/X11 上实际验证，Windows 尚未实机验证。没有图形显示的终端可直接运行：

```console
conda run -n generals-rl python -m generals_env simulate --players 4 --size 25 --games 10 --max-ticks 20000 --seed 42
```

- [环境使用说明与接口](env/README.md)
- [完整设计方案](doc/cpu_prototype_design.md)
- [实现和验证记录](doc/implementation_report.md)
- [RL 目录边界](rl/README.md)

## 开发检查

```console
conda run -n generals-rl python -m unittest discover -s env/tests -v
conda run -n generals-rl python env/benchmarks/benchmark_cpu.py --size 25 --players 4 --ticks 2000
```

GUI 测试需要可用桌面；没有显示服务时对应测试明确跳过。设置正确的显示环境后重新执行即可覆盖真实窗口交互。
