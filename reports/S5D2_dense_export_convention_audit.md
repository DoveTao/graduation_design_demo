# S5D2 Dense Export Convention Audit

## Scope

This report audits the S5 dense external trajectory export. It does not change S5 policy, locked metrics, split, or official eval convention, and it does not start a new algorithm experiment.

## Reference Metrics

S5 official locked: ATE = 7.352288, drift = 1.327343, official path_ratio = 0.932379.

S5 dense external: none ATE = 21.681522, SE(3) ATE = 8.231469, Sim(3) ATE = 4.079123, path_ratio = 2.777268.

## Dense Export Summary

- trajectory path: `external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt`
- dense_export_found: `True`
- num poses: `454`
- matched poses: `454`
- timestamp status: strictly increasing `True`, match rate `1.000000`
- quaternion status: close to unit norm `True`

## Path Ratio Audit

- pred path length: `71.746932`
- GT path length: `25.833640`
- dense path_ratio mismatch: `True`
- dense path_ratio: `2.777268`
- scale_factor_to_match_gt: `0.360066`
- suspected_scale_over_application: `True`
- pred tmag mean/median/p90/max: `0.158382` / `0.137201` / `0.201692` / `0.531463`
- GT tmag mean/median/p90/max: `0.057028` / `0.007507` / `0.212513` / `0.740239`
- log tmag error mean/median/p90/max: `2.522650` / `2.856695` / `4.164398` / `5.815660`

## Convention Variant Audit

| variant | ATE_none | ATE_se3 | ATE_sim3 | path_ratio | notes |
|---|---:|---:|---:|---:|---|
| original_xyzw | 21.681522 | 8.231469 | 4.079123 | 2.777268 | ok |
| invert_pose | 21.133302 | 8.438499 | 4.057329 | 14.538626 | ok |
| axis_flip_x | 23.561149 | 8.228929 | 4.078596 | 2.777268 | ok |
| axis_flip_y | 18.589690 | 8.228929 | 4.078596 | 2.777268 | ok |
| axis_flip_z | 21.675145 | 8.228929 | 4.078596 | 2.777268 | ok |
| quaternion_order_wxyz | 21.681522 | 8.231469 | 4.079123 | 2.777268 | ok |
| reverse_position_order | 21.849098 | 6.966184 | 3.634527 | 2.777268 | ok |

## Official vs Dense Scope

- official evaluator scope: test split odometry evaluation via eval_odometry_sequence over selected odometry chains and configured eval k-list
- dense export scope: scene01/seq03 split=None dense adjacent k=1 all-frame stream exported to TUM for external evaluator
- same_pair_source: `unknown`
- path_ratio_definition_match: `False`

## Final Classification

`S5D2-EVAL-SCOPE-MISMATCH`

## Recommendation

Do not compare official S5 locked path_ratio directly with dense external path_ratio; audit scope-specific scale behavior separately.

## Caveats

- no S5 policy change
- no final candidate change
- no new clean-result claim
- ORB-SLAM3 remains external baseline only
