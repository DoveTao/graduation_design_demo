# S5E4 / S5E3 / S5E2 / ORB-SLAM3 comparison

## 执行摘要
S5E4 用 temporal sign disambiguation 改善 signed tdir 风险，但仍是 experimental candidate。

## comparison table
| method | coverage | rot | signed tdir | tdir_abs | anti_parallel_rate | tmag | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S5 official locked result | official evaluator | None | None | None | None | None | 7.352288 | None | None | 0.932379 | official locked result |
| S5E2 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914153 | None | 32.77577273937451 | 51.706643679614736 | 23.713532954995145 | 4.097680633241629 | 3.5557784814144453 | experimental minimal ridge |
| S5E3 traceable dense | 454/454 | 0.9134395040767528 | 135.28985476811536 | 37.14982041104558 | None | 12.109093390318419 | 30.682466118163305 | 10.774498124607831 | 3.9115097570948705 | 2.1345479454214416 | scale calibrated but signed tdir bad |
| S5E4 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914154 | 0.1368653421633554 | 12.109093390318423 | 28.742908168441225 | 13.75936457268861 | 4.0211631491566155 | 2.1345479454214416 | temporal sign disambiguation candidate |
| ORB-SLAM3 external baseline | 273/454 | None | None | None | None | None | 11.46685113827757 | 0.30854441069248173 | 0.224292165986624 | 0.2998258665660257 | partial coverage external baseline |

## interpretation
S5E4 tests signed temporal direction recovery; it remains diagnostic and cannot replace official S5.

## caveats
- S5E4 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
