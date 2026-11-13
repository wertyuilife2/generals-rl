# 启发式策略的等价性与决策耗时

`decisions.py` 同时编译指定 Git revision 和工作区源码，在同一份公开观测上逐步运行两个独立实例，任何动作不同都会立即失败。两个实例使用相同席位、种子、参数和上一动作回执；只有参考实例的动作推进 `env` CPU backend。

```bash
PYTHONPATH=env/src:baselines/src:arena/src .venv/bin/python \
  heristic/benchmarks/decisions.py --reference BASE_REVISION \
  --ticks 1200 --seeds 3 --players 2 3 4 5 6 7 8 --dense \
  --output outputs/benchmarks/heuristic-decisions.json
```

可以重复传入 `--policy kestrel-v5 --policy razor-v7` 缩小范围。默认参数取策略目录的 `params.json`。临时参考源码和动态库随测试结束删除，不修改工作区、联赛数据或已注册快照。

时间由 C++ 在 `arena_act` 调用周围测量，编译、Python、观测构造和模拟不计入决策时间。每步交替两个构建的执行顺序，报告均值、P95、P99 和速度比。对局使用 15×17、25×27、33×35 地图；`--dense` 额外构造符合 env 状态约束的大军团局面。密集局面和 Razor 历史观测用于动作回归，不混入常规对局的耗时统计。`--profile` 会为参考构建注入函数计时，其结果仅用于定位热点，不能作为正式性能比较。

脚本直接调用策略 C ABI，以隔离策略计算成本并在原版较慢时继续比较动作；它不代替 Arena 的 10 ms watchdog 接入验证，也不保证所有地图上的最坏决策时间。策略仍通过正常的 Arena SDK 编译和限时运行。

## 支持的策略

默认比较以下版本，可通过 `--policy` 指定版本。

| 版本 | 源码 |
| --- | --- |
| Kestrel v1–v5 | `heristic/gpt-6-kestrel-v{1..5}/policy.cpp` |
| Kestrel 默认入口 | `heristic/gpt-6-kestrel/policy.cpp`，与 v4 同步 |
| Razor v1 | `heristic/gpt-6-razor/policy.cpp` |
| Razor v2 | `heristic/gpt-6-razor/v2/policy.cpp` |
| Razor v3 发布版 | `heristic/gpt-6-razor/v3-release/policy.cpp` |
| Razor v6、v7 | 对应目录的 `policy.cpp` 和 `planning.h` |

优化保持原有规则理解、参数、搜索深度、候选集合和比较顺序：

- 缓存同一观测内的兵数、首步风险与公共评分分量，保留浮点加减顺序。
- 只在格子通行状态改变时更新搜索邻接表，方向顺序仍为上、右、下、左。
- 用两遍整数扫描计算九宫格可见性；旧版 Razor 改用等价的九位视野掩码。
- 没有可产生威胁的敌军时跳过早期 Kestrel 的防守动态规划。
- Razor 束搜索排序移动路径指针；兵力调度仅跳过超过子树边数、必为无解的预算状态。

Arena 已注册选手使用不可变快照。要运行优化后的实现，应由 Arena 从上述源码重新编译；本次源码优化不会静默改写历史选手的二进制或评分。
