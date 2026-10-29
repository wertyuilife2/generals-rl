# Official 31.4.5 conformance data

Captured on 2026-10-28 from the deployed [generals.io client](https://generals.io/generals-main-prod-v31.4.5-e5c53eb3.js).

SHA256: `777be342300996f23a97e540248fa9765ab7f59534b0a7b2bc80f738ea325e8f`.
This directory contains numeric inputs, outputs and digests, not the official client source.

## Real replays

Recent ordinary duel/FFA entries were obtained from `https://generals.io/api/replays`, then downloaded through `https://generals.io/api/replay/<id>`.
The replay format version **19** is distinct from client/game release **31.4.5**.

| Replay | Mode | Geometry | Last replay tick |
|---|---|---|---|
| [IZSpNMqIS](https://generals.io/replays/IZSpNMqIS) | 1v1 | 23×18 | 509 |
| [V4hcb6tBF](https://generals.io/replays/V4hcb6tBF) | 1v1, departure | 18×21 | 366 |
| [eerdV1X22](https://generals.io/replays/eerdV1X22) | 1v1, departure | 18×19 | 165 |
| [6CeX6xdG5](https://generals.io/replays/6CeX6xdG5) | 8-player FFA, departure | 31×32 | 396 |
| [WtLgBl5yB](https://generals.io/replays/WtLgBl5yB) | 3-player FFA | 19×20 | 176 |
| [twdUV3dVM](https://generals.io/replays/twdUV3dVM) | 8-player FFA | 31×32 | 912 |

The official deserializer and replay simulator were executed in a local Node VM. Each JSON retains initial terrain/cities/generals, joint actions, departure events, terminal winner/death order and a canonical state digest at every tick. Every 20 ticks and at the end, per-player view digests cover visibility, indistinguishable fog obstacles, terrain, structure, owner and army. Our tests do not require network access or execute downloaded JavaScript.

Canonical state serialization is compact JSON of `[owner, army, structure, general_pos, alive]`, with all neutral/empty owners normalized to −1 for comparison with the official representation; SHA256 hashes the UTF-8 bytes. View serialization is compact JSON of the flattened arrays in the order above. Live games end immediately at a terminal departure boundary; the official replay UI may display one additional render frame. Tests explicitly account for this difference.

## Controlled official-program cases

| File | Cases | Evidence |
|---|---:|---|
| `edge_cases.json` | 108 | Joint action priority, chasing dependencies/cycles, direct opposing moves, general swaps, growth and randomized small boards, executed by the official simulator |
| `city_placement.json` | 20 | Official 1v1 city groups and fairness deficit under a fixed random stream |
| `ffa_cities.json` | 36 | Official live FFA city placement/fallback; 3/4/8 players, multiple requested counts and RNG seeds, including city armies |

The city fixtures use the LCG `x = (1664525*x + 1013904223) mod 2^32`, returning `x/2^32`. FFA neighbor traversal order is also frozen in the fixture so constructor RNG consumption cannot contaminate the comparison.

The live game's AFK/winner cache was separately checked against its `Game.update/checkAfks/killPlayer` implementation. Its same-tick multiple-AFK behavior differs from the replay UI. A dedicated cross-backend test asserts the live behavior, including clearing warnings when a player satisfies the expansion requirement and retaining the cached winner even if that player is subsequently marked dead in the same AFK loop.

No actual mutual-general-swap event was found in the 123 recent ordinary replays inspected. Swap boundary coverage therefore comes from executable official-program fixtures, **not from a recorded real swap game**. Map-generator RNG identity is not promised; distribution and constraints follow the official algorithm, while seeded random streams remain project-owned.

Local acquisition scripts, client bytes and binary replays are archived under ignored `outputs/analysis/official_rules_20261028/`. To refresh evidence, fetch and hash the new deployed client first, regenerate oracle outputs independently of this environment, then use a new rules/fixture version. Never generate expected outputs from the implementation being tested.
