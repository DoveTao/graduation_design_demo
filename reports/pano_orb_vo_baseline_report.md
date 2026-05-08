# Pano-ORB-VO Strong Baseline

## Scope
This is a protocol-compatible strong baseline built on panorama-derived virtual pinhole views.
It is not ORB-SLAM3, and it is not an original fisheye-camera baseline.

## Input
- sequence: `scene01/seq03`
- number of frames: `454`
- virtual camera: `640x480`, `fov=90.0` deg, yaws=`[0.0, 90.0, 180.0, 270.0]` pitch=`0.0`
- virtual intrinsics: `fx=fy=320.000000`, `cx=320.000000`, `cy=240.000000`
- note: these are derived virtual pinhole intrinsics, not original camera parameters.

## Method
- panorama -> virtual pinhole views
- ORB matching
- essential matrix
- recoverPose
- trajectory composition
- main scale policy: `train_median_k1_step_scale` (train-only, non-test-GT tuned)

## Results
| method | alignment | ATE | drift | path_ratio | tracking_success_rate | mean_inliers | median_inliers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pano-ORB-VO-train-scale | none | 10.194695 | 0.172360 | 0.811334 | 1.000000 | 191.668874 | 88.000000 |
| Pano-ORB-VO-train-scale | se3 | 4.137858 | 0.173159 | 0.811334 | 1.000000 | 191.668874 | 88.000000 |
| Pano-ORB-VO-train-scale | sim3 | 4.133322 | 0.233378 | 0.811334 | 1.000000 | 191.668874 | 88.000000 |
| Pano-ORB-VO-oracle-scale | none | 10.610520 | 0.588103 | 4.055540 | 1.000000 | 191.668874 | 88.000000 |
| Pano-ORB-VO-oracle-scale | se3 | 3.559221 | 0.581265 | 4.055540 | 1.000000 | 191.668874 | 88.000000 |
| Pano-ORB-VO-oracle-scale | sim3 | 3.160861 | 1.447171 | 4.055540 | 1.000000 | 191.668874 | 88.000000 |

## Comparison with S5
| method | ATE | drift | path_ratio | notes |
| --- | --- | --- | --- | --- |
| S5_clean_tmag_calibration_policy | 7.352288 | 1.327343 | 0.932379 | Final locked clean candidate. |
| Pano-ORB-VO-train-scale | 4.137858 | 0.173159 | 0.811334 | Protocol-compatible classical panorama VO baseline using derived virtual pinhole views. |

## Interpretation
Pano-ORB-VO is competitive in at least one trajectory metric, but scale/path stability and failure rate still need to be considered together.

## Caveats
- derived virtual pinhole input
- not direct original fisheye baseline
- no loop closure
- monocular scale ambiguity
- scale policy not test-GT tuned unless a diagnostic oracle-scale row is explicitly marked
- Sim(3)-aligned ATE evaluates shape after scale alignment, while path_ratio and none alignment remain scale-sensitive
