# S5E6 / S5E5 / S5E4 / S5E3 / S5E2 / ORB-SLAM3 comparison

## 执行摘要
S5E6 不是去堆更大的模型，而是优先修复 S5E5 的 scale outlier 和 path explosion。

## comparison table
| method | coverage | rot | signed tdir | tdir_abs | anti_parallel_rate | tmag median | tmag p95 | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S5 official locked result | official evaluator | None | None | None | None | None | None | 7.352288 | None | None | 0.932379 | official locked result |
| S5E2 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914153 | None | 32.77577273937451 | None | 51.706643679614736 | 23.713532954995145 | 4.097680633241629 | 3.5557784814144453 | minimal ridge baseline |
| S5E3 traceable dense | 454/454 | 0.9134395040767528 | 135.28985476811536 | 37.14982041104558 | None | 12.109093390318419 | 73.00488324816631 | 30.682466118163305 | 10.774498124607831 | 3.9115097570948705 | 2.1345479454214416 | scale calibration |
| S5E4 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914154 | 0.1368653421633554 | 12.109093390318423 | 73.00488324816631 | 28.742908168441225 | 13.75936457268861 | 4.0211631491566155 | 2.1345479454214416 | temporal sign prior |
| S5E5 traceable dense | 454/454 | 0.9134395040767528 | 56.14019055030064 | 52.096460882587564 | 0.10816777041942605 | 11.953588693727822 | 95.44378091660477 | 113.44308639694538 | 66.13733880826825 | 4.111550831251635 | 6.591091747656769 | temporal visual backbone smoke |
| S5E6 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914154 | 0.1368653421633554 | 11.352940388835835 | 63.490479510847564 | 24.922442378420534 | 12.016350787220915 | 4.012221895981248 | 1.880843438039364 | robust scale guard |
| ORB-SLAM3 external baseline | 273/454 | None | None | None | None | None | None | 11.46685113827757 | 0.30854441069248173 | 0.224292165986624 | 0.2998258665660257 | partial coverage external baseline |

## interpretation
S5E6 is a robust scale-guard experiment focused on fixing S5E5 path explosion without giving up full traceability.

## caveats
- S5E6 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
