# Razor 策略

从 `neko-dev` 的 `54145062716ab4e59e30d2a805c2e1218ad29da9` 选择性移植。
v7 提供任务规划、兵力调度和分兵记账；v8 是带区域/战术规划的实验版本，不能视为已验收的 v7 替代品。
仅保留源码和参数，未引入历史 trials、实验报告或线上对局存档。

已按当前 Arena ABI 2 读取迷雾障碍，并修正即时可移动兵数、部分增长预测及交换将军后的记忆。
前瞻搜索仍是启发式估计，其中部分对手回复顺序按座位近似；不是官方规则的第二套精确模拟器。
旧分支的评分与胜率不代表当前规则下的实力，应在新联赛中重新评估。

```bash
generals-arena run --player heristic/gpt-6-razor/v7 --params heristic/gpt-6-razor/v7/params.json \
  --player baseline/aggressive --games 10 --output outputs/razor-test
```

命令的参数绑定和批量评估方式见 [Arena](../../arena/README.md)；
注册正式版本时也必须传入对应 `params.json`，不要把空参数结果与该配置混为一谈。
