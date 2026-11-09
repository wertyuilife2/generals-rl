# Aegis v1 / v2

Aegis 是独立实现的单线程 C++ 启发式策略，通过 Arena SDK 的
`arena_create`、`arena_act`、`arena_destroy` 接入。策略只使用观测、公开统计及自身记忆，
不依赖 Python 决策代码，也不包含其他选手的实现。

目录名沿用最初的 v1 研发目录；当前根目录的策略对应已发布的 **v2**。

| Arena 名称 | 策略入口 | 参数 |
| --- | --- | --- |
| `gpt-6-aegis-v1` | [checkpoints/v1/policy.cpp](checkpoints/v1/policy.cpp) | [checkpoints/v1/params.json](checkpoints/v1/params.json) |
| `gpt-6-aegis-v2` | [policy.cpp](policy.cpp) | [params.json](params.json) |

提交完整目录内的源码和同名参数文件即可由 Arena 编译。v1 依赖同目录的 `menace.h`；
v2 还依赖 `world_model.h` 和 `deployment.h`。公共头文件 `arena_policy.h` 由 Arena 提供。
根目录 v2 保留当前的性能优化；`checkpoints/v1` 保留首次发布的策略代码和参数。

v2 在 v1 基础上改进前方集结选点、兵力交付价值、敌将位置估计与部署距离，
保留第 24 半回合附近的开局集中出兵、局部扩张、资源路线搜索及可见威胁防守。
内部搜索有截止时间，Arena 的单步决策上限仍为 10 ms。

`tests/strategy.cpp`、`tests/opening.cpp`、`tests/deployment.cpp` 和
`tests/fog_knowledge.cpp` 为当前 v2 的行为检查，可用 C++17 编译，
包含路径为 `arena/src/generals_arena/policies/sdk`。开局检查接受参数 JSON 路径。

例如，从仓库根目录运行：

```sh
c++ -std=c++17 -O3 -ffp-contract=off \
  -I arena/src/generals_arena/policies/sdk \
  heristic/gpt-6-aegis-v1/tests/strategy.cpp -o /tmp/aegis-strategy
/tmp/aegis-strategy
```

未发布的研究候选没有包含在这两个版本中。
