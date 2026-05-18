# FINAL360M vs old trajectory backends

| backend | ATE none | ATE SE3 | ATE Sim3 | path_ratio | coverage | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FINAL360M-direct | 251.214513 | 119.934128 | 29.423059 | 1.867511 | 1.000000 | protocol-correct direct composition from full-train FINAL360M |
| FINAL360M-ODOM360A | 132.075060 | 53.831131 | 29.394188 | 1.186821 | 1.000000 | selected by val only; lightweight local-window fusion |
| FINAL360M-ODOM360B | 132.076321 | 53.833900 | 29.395214 | 1.221642 | 1.000000 | selected by val only; local pose graph with k-step constraints |
| old TRAIN360E / FINAL360I direct | 222.568564 | 118.603779 | 27.564662 | 1.756343 | 1.000000 | subset-trained candidate composed into trajectory |
| old SEQ360B | 139.227175 | 75.946913 | 27.567211 | 1.350237 | 1.000000 | old FINAL360I-based scale smoothing |
| old ODOM360A | 94.598711 | 52.194464 | 27.580519 | 1.110492 | 1.000000 | retained export-only historical reference |
| old ODOM360B | 94.600219 | 52.196647 | 27.581094 | 1.144478 | 1.000000 | retained export-only historical reference |
| BASE360D | 103.256252 | 79.295214 | 2.974466 | 0.043618 | 1.000000 | official external sequence pipeline |

- pair-level scale/path improvement transfers to trajectory-level: `false`
- worse FINAL360M signed_tdir appears to hurt trajectory shape / ATE Sim3: `true`
- old ODOM360A conclusion still holds for FINAL360M: `false`
- trajectory backend main result should be refreshed to a FINAL360M-based version: `true`
- recommended FINAL360M trajectory backend: `lightweight_trajectory_fusion`
