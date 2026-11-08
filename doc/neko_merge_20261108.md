# neko-dev 合并与 RL 兼容性检查

日期：2026-11-08。目标分支：`codex_dev_clean`；上游：
`origin/neko-dev@f85e36dbda15da7c473682d7332c5d0f44b2f93d`。

合并前拆分两个提交：`916ae35` 原生执行链；`e777e8f` 七对手训练评估。
本次采用真实 merge，保留上游父提交与历史；排除的文件不进入合并结果的源码树，不改写上游历史。

## 合并范围

- 合入原生 CPU 规则适配、Metal 后端、线上接入、桌面入口删除及 Arena 的网页、联赛、统计和回放更新。
- 保留本地显式 `execution="isolated"` 路由；联赛批任务明确指定 `execution="batch"`，兼顾隔离请求与批执行性能。
- 保留七对手评估、固定源码/参数打包、run 内冻结对手及 TensorBoard 的逐对手和 4/3/7 对手均值。
- 排除 Razor `experiments/` 的 1502 个文件，以及依赖历史试验和旧 Arena API 的 66 个 Razor 测试/研究文件。
  保留命名策略版本、参数、必要头文件及当前 Aegis 行为测试。既定评估面板的 Razor v8 保留，其实验性状态仍以策略 README 为准。
- 补齐七对手对应的训练集成测试及独立安装包内容验证。

## 兼容处理

旧 `evaluation.backend="gpu"` 和 Arena `--backend gpu` 在指定 CUDA 推理设备时提示弃用，映射为 CPU 原生环境。
模型推理设备、batch、显式 profile 与采样方式保持原配置；迁移后的训练配置记录实际 backend。

旧 GPU 评估 run 的 checkpoint 可续训到新目录，学习器/optimizer 的恢复流程保持原有实现。
旧评估记录保留在原目录，新评估重新开始，不沿用旧 best/Elo；新 `evaluations/index.json` 的 `migration`
记录原索引路径、SHA-256 和 profile ID。旧校准 profile 仍须按当前执行源码重新校准。

CUDA 显式投降增加设备端入口，直接复用 `rules.h`，支持 lane mask、episode 时限和缓存失效，
不再将整批状态搬往 CPU。普通 CUDA `simulate()` 函数体未改动；PPO、网络结构及 MDP 编码未改动。
Metal 的低频显式离场保留上游实现。

`CoreEngine` 现为原生 CPU 兼容层。搜索模拟器的父子状态隔离、独立分支、终局、截断和规则边界已检查；
与 CPU lane 的比较验证适配一致性，不再称为独立 Python 规则验证。官方客户端固定案例仍提供独立预期结果。

## 验证

验证使用隔离导出的待提交源码，避免依赖主工作区未提交的地图课程等实验改动。

| 检查 | 结果 |
| --- | --- |
| env CPU / 离线 online 协议 | 298 passed，80 skipped，148 subtests |
| Arena | 290 passed，3 skipped，2 subtests |
| CUDA 环境及投降回归 | 43 passed，8 skipped，1347 subtests |
| RL 全量与失败项定向复测 | 全部问题修复；含搜索、训练/续训、迁移、七对手评估和 PPO 状态隔离 |
| 网页 JavaScript | 31 passed |
| 策略源码 | 20 个版本全部编译；4 个 Aegis 行为测试通过 |
| GPU 双座自对弈 smoke | 32 个游戏环境，2 次 PPO 更新，2 次评估共 64 局，0 错误/超时；旧 backend 配置迁移与 TB 指标已验证 |
| wheel 独立安装 | 包含 Metal/online/原生源码及三个固定启发式；16 局评估，0 错误/超时 |

没有 Apple GPU，未验证 Metal 实机执行；线上接入只做离线协议测试，未发起真实线上对局。
本轮未启动长期训练，也未推送远端。其他未提交的 RL 实验改动单独保留。

本机原始日志、排除清单和汇总：`outputs/analysis/neko_merge_20261108/`。
