# Final Trajectory-level Results

| Model | ATE none | ATE SE3 | ATE Sim3 | trajectory_path_ratio | coverage | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FINAL360M-direct | 251.214513 | 119.934128 | 29.423059 | 1.867511 | 1.000000 | Protocol-correct direct composition from the full-train FINAL360M pair model. |
| FINAL360M-ODOM360A | 132.075060 | 53.831131 | 29.394188 | 1.186821 | 1.000000 | FINAL360M-based lightweight local-window trajectory fusion selected by val only. |
| FINAL360M-ODOM360B | 132.076321 | 53.833900 | 29.395214 | 1.221642 | 1.000000 | FINAL360M-based local pose graph with k-step constraints selected by val only. |
| old TRAIN360E / FINAL360I direct | 222.568564 | 118.603779 | 27.564662 | 1.756343 | 1.000000 | Historical subset-trained direct export reference only. |
| old SEQ360B | 139.227175 | 75.946913 | 27.567211 | 1.350237 | 1.000000 | Historical FINAL360I-based scale smoothing reference only. |
| old ODOM360A | 94.598711 | 52.194464 | 27.580519 | 1.110492 | 1.000000 | Historical FINAL360I-based retained export reference. |
| old ODOM360B | 94.600219 | 52.196647 | 27.581094 | 1.144478 | 1.000000 | Historical FINAL360I-based retained export reference. |
| BASE360D | 103.256252 | 79.295214 | 2.974466 | 0.043618 | 1.000000 | Official external sequence pipeline baseline. |

Table note: `FINAL360M-ODOM360A` is the recommended refreshed FINAL360M-based backend among the newly rerun eval-only variants.
Table note: the old `FINAL360I`-based ODOM360A/B rows are kept only as historical references and are no longer valid final-main evidence.
