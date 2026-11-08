# Aegis 首次 Arena 试测候选

冻结来源：secure-growth-prototype-01。默认开启 secure_growth，显式参数与研发评测一致。
对 Razor v7 两批共 241/300（80.3%），aggressive 299/300（99.7%），全部参与者无超时或错误。
同图对照对 Razor 为 246/300（82%），因此这不是已证明更强的研发默认版；根据用户“平均 80% 可进入 Arena 试测”的授权发布首个 checkpoint。
正式全量成绩以 Arena publication.json 为准。

首次全量评测已完成：1,000 局（1v1 563 局，FFA 437 局），1v1 GoR 2296.94、FFA 1179.11，完成时两榜均第 1。77 次本策略席位决策超时均按投降计入，控制器错误为 0；最大记录耗时 49.972 ms，尚不满足禁止超时的目标。

旧运行进程完成 377 局后停止，其余 623 局使用当前优化版，4 局并行，仍为同一 checkpoint。完整成绩：`outputs/arena/main/commits/e7728d5bcb4f71ebdb923fdda0409b33d6ec0a2695e41e5b891f8ae7a743e36d/publication.json`。
