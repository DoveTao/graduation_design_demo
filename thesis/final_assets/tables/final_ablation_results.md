# Final Structural Ablation Results

| Model | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PlainPairVO | 2.421762 | 49.126304 | 0.206835 | 0.876957 | 0.685466 | 1.000000 | reports/ABLDVO2_metrics_test.json |
| NoSphericalGeometry | 2.536085 | 45.103224 | 0.199723 | 0.781094 | 0.614440 | 1.000000 | reports/ABLDVO2_metrics_test.json |
| NoCrossImageInteraction | 3.606171 | 49.758598 | 0.201699 | 0.623008 | 0.490037 | 1.000000 | reports/ABLDVO2_metrics_test.json |
| SingleStagePoseRegression | 2.658405 | 51.409642 | 0.200316 | 1.318604 | 1.024806 | 1.000000 | reports/ABLDVO3_metrics_test.json |
| FINAL360I | 2.330109 | 45.264702 | 0.201699 | 0.834425 | 0.640352 | 1.000000 | reports/ABLDVO2_metrics_test.json, reports/ABLDVO3_metrics_test.json |

Table note: This core thesis ablation table merges the retained `ABLDVO2` module ablations and the `ABLDVO3` single-stage regression ablation.
