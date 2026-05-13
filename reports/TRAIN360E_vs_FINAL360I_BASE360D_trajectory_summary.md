# TRAIN360E vs FINAL360I BASE360D trajectory summary

| model | split | sequence | ATE none RMSE | ATE SE3 RMSE | ATE Sim3 RMSE | trajectory_path_ratio | pred_path_length | gt_path_length | coverage | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TRAIN360E / FINAL360I composed | val | mountains | 70.730716 | 22.754926 | 19.668482 | 3.613210 | 374.779122 | 103.724697 | 1.000000 | pair-level model composed into trajectory |
| TRAIN360E / FINAL360I composed | val | downhill_biking | 456.994928 | 24.112738 | 20.433648 | 1.056649 | 478.330735 | 452.686471 | 1.000000 | pair-level model composed into trajectory |
| TRAIN360E / FINAL360I composed | test | snowmobile | 260.054001 | 128.374524 | 10.023027 | 2.360466 | 768.052406 | 325.381641 | 1.000000 | pair-level model composed into trajectory |
| TRAIN360E / FINAL360I composed | test | ridge_to_lake | 161.158831 | 104.516151 | 40.232291 | 1.287942 | 540.501431 | 419.662906 | 1.000000 | pair-level model composed into trajectory |
| BASE360D | val | downhill_biking | 137.186671 | 114.592544 | 3.330729 | 0.110386 | 49.970371 | 452.686471 | 1.000000 | official sequence VO pipeline |
| BASE360D | val | mountains | 38.332161 | 14.909265 | 0.522377 | 0.409545 | 42.479952 | 103.724697 | 1.000000 | official sequence VO pipeline |
| BASE360D | test | ridge_to_lake | 80.695912 | 66.506846 | 2.352784 | 0.050396 | 21.149423 | 419.662906 | 1.000000 | official sequence VO pipeline |
| BASE360D | test | snowmobile | 117.677369 | 87.859110 | 3.374917 | 0.034876 | 11.347930 | 325.381641 | 0.866265 | official sequence VO pipeline |

- caveat: `FINAL360I` is a pair-level model composed into a sequential trajectory, so drift can accumulate.
- caveat: `BASE360D` is an official sequence-level VO pipeline and is not directly equivalent to pair-only inference.
- caveat: `ATE` and pair-level component metrics answer different questions; both are kept in the final analysis.
