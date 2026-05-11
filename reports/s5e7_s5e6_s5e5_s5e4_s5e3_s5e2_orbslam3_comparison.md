# S5E7 / S5E6 / S5E5 / S5E4 / S5E3 / S5E2 / ORB-SLAM3 comparison

## 执行摘要
S5E7 必须同时改善方向和尺度，单独 ATE 下降并不足以认定几何提升。

## comparison table
| method | coverage | rot | signed tdir | tdir_abs | anti_parallel_rate | tmag median | tmag p95 | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S5 official locked result | official evaluator | None | None | None | None | None | None | 7.352288 | None | None | 0.932379 | official locked result |
| S5E2 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914153 | None | 32.77577273937451 | None | 51.706643679614736 | 23.713532954995145 | 4.097680633241629 | 3.5557784814144453 | minimal ridge baseline |
| S5E3 traceable dense | 454/454 | 0.9134395040767528 | 135.28985476811536 | 37.14982041104558 | None | 12.109093390318419 | 73.00488324816631 | 30.682466118163305 | 10.774498124607831 | 3.9115097570948705 | 2.1345479454214416 | scale calibration |
| S5E4 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914154 | 0.1368653421633554 | 12.109093390318423 | 73.00488324816631 | 28.742908168441225 | 13.75936457268861 | 4.0211631491566155 | 2.1345479454214416 | temporal sign prior |
| S5E5 traceable dense | 454/454 | 0.9134395040767528 | 56.14019055030064 | 52.096460882587564 | 0.10816777041942605 | 11.953588693727822 | 95.44378091660477 | 113.44308639694538 | 66.13733880826825 | 4.111550831251635 | 6.591091747656769 | temporal visual backbone smoke |
| S5E6 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914154 | 0.1368653421633554 | 11.352940388835835 | 63.490479510847564 | 24.922442378420534 | 12.016350787220915 | 4.012221895981248 | 1.880843438039364 | robust scale guard |
| S5E7 traceable dense | 454/454 | 0.9134395040767528 | 53.62212702240343 | 48.649060747032244 | 0.1545253863134658 | 0.8315272040683075 | 16.590740550562295 | 9.787743578157508 | 3.7889242467975253 | 3.765359444110737 | 0.3237847877095814 | direction-scale calibrated geometry |
| ORB-SLAM3 external baseline | 273/454 | None | None | None | None | None | None | 11.46685113827757 | 0.30854441069248173 | 0.224292165986624 | 0.2998258665660257 | partial coverage external baseline |

## comparison summary
{'vs_s5e6': {'rot_preserved': True, 'signed_tdir_improved_vs_s5e6': False, 'tdir_abs_improved_vs_s5e6': False, 'anti_parallel_improved_vs_s5e6': False, 'tmag_p95_improved_vs_s5e6': True, 'path_ratio_improved_vs_s5e6': True, 'sim3_ate_improved_vs_s5e6': True, 'raw_vs_guarded': {'raw_vs_guarded_tdir_gap': 0.0, 'raw_vs_guarded_tdir_abs_gap': 0.0, 'raw_vs_guarded_tmag_p95_gap': 77.69394478224177, 'raw_vs_guarded_path_ratio_gap': 6.366704946500162}}, 'vs_orbslam3': {'coverage': 'S5E7 454/454 vs ORB-SLAM3 273/454', 'rot_gap': 'unavailable', 'tdir_gap': 'unavailable', 'tmag_gap': 'unavailable', 'se3_ate_gap': 3.4803798361050435, 'sim3_ate_gap': 3.541067278124113, 'path_ratio': 'S5E7=0.3237847877095814; ORB-SLAM3=0.2998258665660257', 'summary': 'ORB-SLAM3 没有可比的 edge-level 指标存档，因此这里只比较轨迹层面的覆盖率与 ATE/path_ratio。'}}

## caveats
- S5E7 是 experimental candidate。
- 不替代 official S5 locked result。
- ORB-SLAM3 的 edge-level rot/tdir/tmag gap 仍然 unavailable。
