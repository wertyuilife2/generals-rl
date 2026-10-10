# 当前正式联赛数据

正式目录：`outputs/arena/native_1000`。网页：http://localhost:8086。

当前源码通过 `generals.make_engine()` 复用 `env/backends/cpu`，baseline 由 C++ 实现。
这批数据包含 1000 场 2–8 人混合对局，地图宽高均为 15–35，允许同一算法重复入场。
所有人数共用一个按名次更新的 rating；random 固定为 0。

| 算法 | Rating |
| --- | ---: |
| aggressive-v1 | 495.5 |
| expansion-v1 | 220.2 |
| random | 0.0 |
| defensive-v1 | −58.7 |

数据库保存完整赛程、结果、评分和压缩录像，累计 1,584,005 tick。
3 个评测任务均已完成，无失败或待运行对局。清理时 SQLite 完整性检查通过，
规则指纹与当前源码匹配；数据文件未改动，评分未重置。

同目录保留 `leaderboard.json`、`verification.json`、`schedule_source.json` 和 `run.log`，
用于记录生成时的分数、验证结果、赛程来源和日志。来源元数据中的旧联赛目录已清理；
当前联赛的运行和回放不依赖这些来源目录。网页会从数据库读取后续更新的实时榜单。

旧联赛、迁移备份、接口测试联赛和旧版运行代码副本已删除；当前正式数据保留完整。
C++ 策略的接入和正式 checkpoint 评测见 [RESEARCH.md](RESEARCH.md)。

```bash
generals-arena serve outputs/arena/native_1000 --port 8086 --no-worker
generals-arena league leaderboard outputs/arena/native_1000
python arena/benchmarks/verify_league.py outputs/arena/native_1000 --workers 1
```
