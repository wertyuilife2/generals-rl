# Generals PPO self-play

支持 v32、v32-opt 和 Scaled/F03 checkpoint；研究过程与结果集中在 [auto_research.md](../doc/auto_research.md)。

## 安装与训练

在仓库根目录激活 generals-rl 环境，安装本地包：

```bash
python -m pip install -e ./env -e ./rl/vendor/rsl_rl -e ./rl --no-deps --no-build-isolation
python -m generals_rl.train --config rl/configs/v32-opt.json --output outputs/current
```

v32 使用八个历史对手，v32-opt 使用当前 learner 自对弈。共同协议为9×9/P2、H4局部combat历史、K11知识、独立信息critic、StudentPPO、WAIT/MOVE factorization、productive masks和PBRS。当前actor不接收critic特权输入，对手样本不重复计入learner transitions。Scaled支持ViT/ResNet及dense/pyramid；physical minibatch和inference chunk不改变logical PPO预算。

```bash
python -m generals_rl.evaluate_builtin outputs/current/checkpoint.pt --maps-per-ai 50
python rl/tools/benchmark_training.py --config rl/configs/v32-opt.json --help
```

周期评估和训练恢复均使用完整run_config及checkpoint；恢复历史池需保留其签名和配套快照。性能区分采样＋GAE＋PPO的fresh transitions/s与包括评估的全流程wall指标，不能把PPO重复epoch当新样本。

## 维护范围

保留当前PPO/MDP、历史与current对手、分布式训练、独立评估、动作录制与Core重放。已退休旧dense/MoE模型入口、权重迁移/蒸馏、analytic value offset、旧dilation/context和standaloneMap网络；公共CTS runner/storage及policy基类仍供StudentPPO使用。potential reward、Scaled投影、own-general geometry均保留。

未来MCTS使用search.py的PolicyValue和JointTickSimulator；搜索不能把特权全图误当局部信息。

```bash
python -m pytest -q rl/tests rl/tools/tests
```
