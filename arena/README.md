# Generals arena

独立的 2–8 人对战与 Elo 排名包 `generals_arena`。支持手写 baseline、不同版本的 CTS
checkpoint，以及通过 `Controller` 接口接入的其他算法。核心只依赖环境和 NumPy；
只有选择 RL 选手时才加载 Torch、RSL 和对应模型。

## 安装和运行

在仓库根目录执行：

```bash
python -m pip install -e ./env -e ./baselines -e ./arena
generals-arena run --config arena/configs/smoke.json --output outputs/arena/smoke
generals-arena leaderboard outputs/arena/smoke
generals-arena leaderboard outputs/arena/smoke --json
```

`smoke.json` 是 3×3 棋盘、4 个策略、每对 2 张图的 24 局快速检查，不用于衡量正式实力。
`baselines.json` 是 9×9 棋盘的双人示例，`multiplayer.json` 是四种 baseline 的四人混战示例。
默认 1500 分、K=32、Elo 标度 400。

```bash
generals-arena run --config arena/configs/multiplayer.json --output outputs/arena/ffa
```

用 `rules.map.players` 设置每局人数 P（2–8），参赛选手总数 N 必须至少为 P。
赛程枚举所有 P 人组合，每张地图循环轮换 P 次座位，让每位选手在每个座位各打一局。
总局数为 `C(N,P) * P * maps_per_group`；双人时等于 `N*(N-1)*maps_per_group`。
循环轮换平衡各选手的绝对座位，不枚举 P! 种排列。`maps_per_pair` 作为旧配置的兼容别名仍可使用，
与 `maps_per_group` 不能同时设置。

不安装包时可设置 `PYTHONPATH=env/src:baselines/src:arena/src`，
通过 `python -m generals_arena` 执行同样的命令。

## 登记 RL 版本

先按 `rl/README.md` 安装训练依赖及仓库内的 RSL 版本，再把选手加入配置的 `entrants`：

```json
{
  "name": "cts-update-1000",
  "kind": "rl",
  "checkpoint": "../../outputs/moe_cts/run_a/checkpoint.pt",
  "device": "cpu"
}
```

所有相对文件路径均相对于输入 JSON 所在目录。默认读取 checkpoint 同目录的
`run_config.json`，也可用 `config` 指定配置路径。选手名称必须唯一。
RL 推理使用局部 student 的 argmax；棋盘尺寸、人数必须与训练配置一致，视野必须是 local。
四人和八人比赛分别需要以相同人数训练的 checkpoint，双人模型不能直接用于多人比赛。
每局独立重置历史，时钟归一化沿用训练时限。模型每次启动只加载一次，比赛不训练或更新参数。

版本身份包含名称、推理配置、checkpoint/config 的 SHA256，以及 RL/RSL 源码哈希。
baseline 身份包含策略名和源码哈希。更换文件位置不改变版本，覆盖同一位置的权重会改变版本。
每个训练快照应使用独立名称，例如 `cts-update-1000`、`cts-update-2000`。

## 追加版本和断点续跑

```bash
generals-arena run --config arena/configs/my_league.json --output outputs/arena/league
# 在同一份配置中增加新名称的选手，或者增大 maps_per_group：
generals-arena run --config arena/configs/my_league.json --output outputs/arena/league --resume
generals-arena rebuild outputs/arena/league
```

首次运行要求输出目录不存在。`--resume` 按比赛 ID 跳过已完成的对局；追加选手后只补充缺失对战，
增加地图数后只补充新地图。多人追加选手时只补充含新选手的组合。
已有选手必须保留原名称和版本，已有比赛记录不会重写。
规则、环境/比赛执行源码、地图种子和 Elo 参数固定；改变这些配置需要新建排名池。
源码升级后若要继续旧排名池，应使用原版本代码及模型工件。旧双人结果仍可使用新版 `rebuild` 重算。
不同每局人数使用不同排名池，不混合计算 Elo。

选手报错会中止本次运行，写入 `failure.json`，已完成对局仍可重算排名；失败对局不计作输或平。
正常异常退出会释放写锁。进程被强制终止后，如残留 `.arena.lock`，先确认没有运行中的写入进程再删除。
损坏、重复或不完整的比赛日志会明确报错，系统不会自动丢弃比赛或编造结果。

## 比赛和评分口径

- 使用可移植 Python `CoreEngine`，支持 macOS/Linux 等 Python 环境；顺序执行 2–8 人比赛。
- 所有存活玩家在同一 tick 增兵前读取各自观测并提交动作；随后增兵并按绝对座位顺序执行，和训练时序一致。
  已被淘汰的玩家在后续 tick 不再调用策略，环境使用 WAIT 占位。
- 所有组合共享同一组地图种子；每张地图轮换座位，同一选手在不同座位使用相同策略随机种子。
- 最后存活者获胜，其余玩家整局记负；达到时限且未产生最终胜者时，所有玩家整局记平，包括已淘汰者。
  自然结束优先于同 tick 时限，无效游戏移动保留引擎语义并单独计数。
- 多人 Elo 采用明确的 winner-only 两两比较：胜者对每位非胜者得 1 分，非胜者对胜者得 0 分，
  非胜者相互得 0.5 分；超时局的所有两两比较均得 0.5 分。淘汰顺序和兵力不用于推定名次。
- 对每个对手 j，期望得分 `Eij = 1 / (1 + 10 ** ((Rj - Ri) / scale))`。
  玩家 i 的更新量为 `K/(P-1) * sum_j(Sij - Eij)`，所有期望使用赛前分数，同时更新并保持总分不变。
  除以对手人数使每局更新幅度不随人数增加；P=2 时与原双人 Elo 完全一致。
- 按 `matches.jsonl` 的持久化顺序逐局更新；Elo 与对手池及比赛顺序有关，不同排名池的分数不能直接横比。
- `wins/losses/draws`、`win_rate`、`score_rate` 按整局统计；`pairwise_score_rate` 是用于 Elo 的平均两两得分率。
  `head_to_head` 记录同场两两比较：两位非胜者之间记平，但各自整局记负。少量对局只适合验证流程。

## 接入其他算法

```json
{
  "name": "search-v1",
  "kind": "custom",
  "factory": "my_agents.search:create_controller",
  "version": "git-commit-or-release-id",
  "kwargs": {"simulations": 100},
  "artifacts": {"checkpoint": "./models/search-v1.bin"}
}
```

工厂签名为 `create_controller(*, player_id, seed, **kwargs) -> Controller`。
`artifacts` 的文件先哈希，再将绝对路径以同名参数传入工厂；不能与 `kwargs` 重复。
自定义算法必须声明代码版本，并由实现者保证该版本标识对应固定实现。

`Controller` 位于 `generals.protocol.controller`，需要 `reset(player_id, seed)`、
`act(observation) -> Action`、`on_result(move_result)`。每局前调用 reset，
实现者应清空历史、搜索树和随机流等局内状态。可选的 `validate_rules(map_config, visibility)`
用于在开赛前检查支持的规则。控制器只收到允许的玩家观测，不接收私有棋盘。

## 数据和模块

输出包含固定规则 `config.json`、版本登记 `entrants.json`、原始结果 `matches.jsonl`，
以及派生的 `leaderboard.json`。每局记录座位、种子、终止原因、最终统计、初末状态摘要和无效动作计数，
另保存各座位的最终存活状态及淘汰 tick；这些字段长度均等于每局人数。
排名可以在不加载模型或保留原 checkpoint 的情况下重算；日志不包含逐步动作录像。

`schedule.py` 负责赛程，`match.py` 负责对局，`entrants.py/registry.py` 负责版本和工厂，
`results.py` 负责记录，`ratings/elo.py` 负责评分，`tournament.py` 负责运行及恢复。
CTS 的模型加载和观测适配留在 `rl/inference.py` 与 `rl/arena_adapter.py`。
现有训练期间评估保留原生批量实现，与 arena 共享地图种子派生和胜负平统计；训练对手池仍由 RL 管理。

```bash
python -m unittest discover -s arena/tests -v
python -m unittest discover -s rl/tests -p test_arena_adapter.py -b -v
```
