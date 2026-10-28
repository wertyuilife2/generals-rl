# generals-env

游戏环境子项目，源码包名为 `generals`，发行包名为 `generals-env`。支持 CPU/CUDA 批量模拟及 Tk 桌面游玩。

当前规则固定为 2026-10-28 核查的官网 **31.4.5 普通 1v1 / FFA**，默认按官方尺寸与公平性规则随机生成地图。
CPU、CUDA、桌面 GUI 和 Arena 使用相同动作时序、迷雾及结算语义；版本证据与验证边界见[实施记录](../doc/official_rules_alignment.md)。

从仓库根目录安装和启动：

```bash
conda run -n generals-rl python -m pip install -e ./env -e ./baselines --no-deps --no-build-isolation
conda run -n generals-rl python -m generals play --players 4 --seed 42
```

不安装时，在仓库根目录使用 `PYTHONPATH=env/src:baselines/src python -m generals ...`。
纯规则与批量环境只需要 `generals-env`；AI 游玩和模拟还需要 `generals-baselines`。
这条 `PYTHONPATH` 写法用于 Bash 等 POSIX shell；当前内置 AI 与 CPU 原生后端支持 Linux/macOS，不能据此认为 Windows 已支持完整 GUI 对战。

CPU 原生后端支持 Linux 与 macOS。Linux 需要支持 OpenMP 的 C++17 编译器；macOS 需要
Apple clang 与已安装的 `libomp`，非标准位置可用 `GENERALS_LIBOMP_PREFIX` 指定。
首次使用自动编译，后续复用缓存。单局调用 `generals.make_engine(state)`，批量调用 `VecEnv`；
二者执行同一个 `backends/cpu/step.cpp`，Arena 不再维护执行后端。Python `CoreEngine` 继续作为独立规则参考。

CPU/GPU 共用 `VecEnv.step()`、`observe(player_ids)` 和可复用的 `observer(player_ids)`。
批量观测由 env 返回原始局部信息；特征、历史、知识、动作映射与模型推理属于 `rl/` 的策略实现。

规则、操作、依赖和开发说明统一维护在 [doc/README.md](../doc/README.md)：

- [安装与平台支持](../doc/README.md#quickstart)
- [GUI 操作](../doc/README.md#gui)
- [统一 CPU/GPU API 与状态协议](../doc/README.md#api)
- [观测、路径和控制器接入](../doc/README.md#features)
- [基准及性能结果](../doc/README.md#benchmark)
- [环境性能验证记录](../doc/README.md#results)
- [测试与历史数据](../doc/README.md#verification)
