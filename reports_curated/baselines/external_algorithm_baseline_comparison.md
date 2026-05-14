# External Algorithm Baseline Comparison

## Scope
This report provides an external literature/algorithm baseline comparison framework for the thesis.
Only results generated on the same exported test sequence and under the same evaluation protocol are used for numeric comparison.
Published results from different datasets are not compared directly against S5.

## Dataset and Protocol
- sequence: `scene01/seq03`
- input modality: `monocular panorama / equirectangular RGB`
- camera intrinsics availability: `external_baselines/dataset/scene01_seq03/camera.yaml` with an equirectangular caveat
- GT trajectory: `external_baselines/dataset/scene01_seq03/groundtruth_tum.txt`
- alignment modes: `none / se3 / sim3`
- metrics: `ATE`, `drift`, `path_ratio`, `tracking_success_rate`

## Baseline Methods
| method | category | input | loop_closure | learning_based | run_status | output_trajectory_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pano-ORB-VO | protocol-compatible classical panorama VO baseline | panorama-derived virtual pinhole | no | no | ready_to_evaluate | external_baselines/results/pano_orb_vo/scene01_seq03_est_tum.txt | Repo-local strong baseline built from panorama-derived virtual pinhole views; not an external published method. |
| ORB-SLAM2 | feature-based SLAM | monocular / stereo / RGB-D | yes | no | pending_external_run | (missing) | Protocol-compatible only if built and run on the exported sequence. |
| ORB-SLAM3 | feature-based visual / visual-inertial SLAM | monocular / stereo / RGB-D / visual-inertial | yes | no | pending_external_run | (missing) | Protocol-compatible only if built and run on the exported sequence. |
| DSO | direct sparse visual odometry | monocular | no | no | pending_external_run | (missing) | Requires an input mode compatible with the exported panorama sequence or a documented adapter. |
| DROID-SLAM | deep learning SLAM | monocular / stereo / RGB-D | implicit / backend-dependent | yes | pending_external_run | (missing) | Protocol-compatible only after a same-sequence trajectory is generated locally. |
| S5 | proposed method | monocular panorama / equirectangular RGB | no | yes | locked_internal_result | internal clean eval / locked metrics | Final clean candidate under the historical protocol. |

## Quantitative Comparison
| method | alignment | ATE | drift | path_ratio | tracking_success_rate | notes |
| --- | --- | --- | --- | --- | --- | --- |
| S5 | historical_clean_eval | 7.352288 | 1.327343 | 0.932379 | 1.000000 | Locked internal clean result on the historical final test sequence. |
| Pano-ORB-VO | none | 10.194695 | 0.172360 | 0.811334 | 1.000000 | Same-sequence evaluation only; do not compare against cross-dataset published numbers. |
| Pano-ORB-VO | se3 | 4.137858 | 0.173159 | 0.811334 | 1.000000 | Same-sequence evaluation only; do not compare against cross-dataset published numbers. |
| Pano-ORB-VO | sim3 | 4.133322 | 0.233378 | 0.811334 | 1.000000 | Same-sequence evaluation only; do not compare against cross-dataset published numbers. |

## Literature Context
- Pano-ORB-VO is a protocol-compatible strong baseline implemented in this repo and should not be described as an external published method.
- ORB-SLAM2/3, DSO, and DROID-SLAM are included as representative external baselines only when they produce trajectories on the exported sequence.
- Published numbers from different datasets remain method context only and are not used for direct numerical comparison against S5.

## Thesis-Ready Interpretation
External algorithms are included as protocol-compatible baselines only when their trajectories are generated on the same test sequence.
Published results from other datasets are used only for method context.
S5 remains the final clean candidate under the locked historical protocol and is not practical-ready.

## Caveats
- external baselines require matching input modality
- monocular baselines have scale ambiguity
- Sim(3)-aligned ATE and scale-sensitive path_ratio answer different questions
- results are not comparable to published numbers from different datasets
