# S5E3 / S5E2 / ORB-SLAM3 comparison

## 执行摘要

本报告比较 S5 official locked result、S5E2 traceable dense、S5E3 traceable dense 与 ORB-SLAM3 external baseline。S5E3 仍是 experimental candidate，不替代 official S5 locked result。

## comparison table

| method | coverage | rot | tdir | tdir_abs | tmag | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S5 official locked result | official evaluator | None | None | None | None | 7.352288 | None | None | 0.932379 | official locked result; not external dense |
| S5E2 traceable dense | 454/454 | 0.9134395040767528 | 51.47429479273258 | 46.07968707914153 | 32.77577273937451 | 51.706643679614736 | 23.713532954995145 | 4.097680633241629 | 3.5557784814144453 | experimental minimal ridge; failed metrics |
| S5E3 traceable dense | 454/454 | 0.9134395040767528 | 135.28985476811536 | 37.14982041104558 | 12.109093390318419 | 30.682466118163305 | 10.774498124607831 | 3.9115097570948705 | 2.1345479454214416 | experimental scale calibrated candidate |
| ORB-SLAM3 external baseline | 273/454 | None | None | None | None | 11.46685113827757 | 0.30854441069248173 | 0.224292165986624 | 0.2998258665660257 | partial coverage; external strong baseline |

## interpretation

S5E3 相对 S5E2 的 improvement flags 为 {'rot_improved_or_preserved': True, 'tdir_improved': False, 'tdir_abs_improved': True, 'tmag_improved': True, 'path_ratio_improved': True, 'sim3_ate_improved': True, 'overall_geometry_improved': True}. S5E3 对 ORB-SLAM3 的 se3 ATE gap=10.465953713915349, sim3 ATE gap=3.6872175911082463. full coverage 仍是 S5E3 相对 ORB-SLAM3 的优势，但不得声称替代 official S5。

## recommendations

S5E3 若仍未接近 ORB-SLAM3，应优先升级视觉 backbone 与 direction head；若要使用 ORB-SLAM3 做 teacher，需要另开明确的 distillation 实验。

## caveats

- S5E3 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- ORB-SLAM3 是 external strong baseline。
