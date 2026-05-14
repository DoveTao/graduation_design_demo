# THESIS360 Experiment Section

## 1. Experimental Protocol
This work is evaluated on the DSET2C canonical split constructed on the 360DVO/HKUST data source. The current repository uses a manifest-native protocol: train/val/test samples are read from canonical JSONL manifests rather than discovered through raw directory globbing. The protocol explicitly avoids random pair splitting and treats sequence identity as a first-class constraint. The current split files are `pair_manifest_train.jsonl`, `pair_manifest_val.jsonl`, and `pair_manifest_test.jsonl` under `external_baselines/results/dset2c_360dvo_canonical/`.

Our experiments use two evaluation layers. First, we report pair-level translation component metrics for relative pose prediction: signed translation-direction mean, anti-parallel rate, translation-magnitude median ratio, and path ratio. These metrics directly evaluate the quality of relative pose components predicted by the match-free panoramic model. Second, we report trajectory-level metrics by composing adjacent relative poses into full sequences and then measuring ATE under none / SE3 / Sim3 alignment as well as trajectory path ratio. This second layer is important because pair-level accuracy does not automatically imply stable long-horizon sequential composition.

The current mainline follows strict research guardrails. No ORB-SLAM3 teacher, HKUST 360DVO teacher, or BASE360 output is used as training input. The method remains match-free and does not introduce explicit feature matching, RANSAC, PnP, or bundle adjustment into the learned relative pose pipeline.

## 2. Compared Methods
The retained comparison set is intentionally compact after repository cleanup. `FINAL360I_struct360b_final_selected` is the current pair-level thesis main model. `TRAIN360E` composes the `FINAL360I` adjacent-pair predictions into full trajectories and evaluates them with ATE. `SEQ360B` is the only retained trajectory-scale variant, adding a lightweight log-scale correction head on top of frozen `FINAL360I` predictions. `BASE360D` denotes the HKUST official 360DVO external baseline, kept as an official sequence VO comparison anchor. `T57b` is a recovered legacy baseline used only as an external reference.

Two additional lines are retained only as status summaries, not as main comparison candidates. `SEQ360A` is a negative ablation with status `no_improvement`, while `STRUCT360C` is an unstable challenger with status `evaluation_failed`.

## 3. Pair-level Metrics
The pair-level metrics focus on translation quality in relative pose prediction. Signed translation-direction mean measures the average angular error between predicted and ground-truth translation vectors; lower is better. Anti-parallel rate measures how often the predicted translation direction flips to the opposite hemisphere; lower is better. Translation-magnitude median ratio compares predicted and ground-truth translation scales, with values closer to 1 indicating better scale calibration. Path ratio sums predicted relative translation magnitudes over the evaluation set and divides by the summed ground-truth magnitudes; values closer to 1 indicate a more balanced overall scale tendency.

These metrics matter because the main contribution of this work is a match-free panoramic pair-level relative pose estimator. Even when full sequential VO remains challenging, improved pair-level direction and scale behavior is still a meaningful contribution for downstream composition or refinement systems.

## 4. Trajectory-level Metrics
Trajectory-level evaluation is performed by composing adjacent relative poses into a sequential path. ATE none measures raw trajectory error without additional rigid alignment, ATE SE3 measures error after rigid alignment, and ATE Sim3 measures error after similarity alignment. Trajectory path ratio compares the predicted and ground-truth path lengths and indicates whether the composed trajectory is globally too short or too long. These metrics address a different question from the pair-level component metrics: they evaluate accumulated sequential drift rather than isolated pair quality.

## 5. Main Results
At the pair level, `FINAL360I` achieves a test signed translation-direction mean of 45.264702, an anti-parallel rate of 0.201699, a translation-magnitude median ratio of 0.834425, and a path ratio of 0.640352. These values are substantially stronger than the retained legacy `T57b` reference (`111.964932`, `0.674672`, `0.175156`, `0.148787`) and also much stronger than the trajectory-derived `BASE360D` component baseline (`128.402578`, `0.814895`, `0.039910`, `0.043542`). However, the `BASE360D` component metrics are not native pair-forward outputs and therefore should only be interpreted as partially comparable component proxies.

At the trajectory level, `TRAIN360E` shows that `FINAL360I` can be composed into complete trajectories with full reported test coverage, but the resulting sequence still drifts: test ATE none / SE3 / Sim3 are 222.568564 / 118.603779 / 27.564662, and the trajectory path ratio is 1.756343. `SEQ360B` partially mitigates this drift by improving the trajectory path ratio to 1.350237 and reducing SE3 ATE to 75.946913, but its Sim3 ATE remains essentially unchanged at 27.567211. This indicates that the variant mainly addresses scale/path drift rather than the deeper trajectory-shape error.

## 6. Analysis
The most robust conclusion is that `FINAL360I` is the strongest retained pair-level model. Its translation component metrics clearly improve over both the legacy `T57b` reference and the trajectory-derived `BASE360D` component baseline. This supports the claim that the main contribution of the method lies in match-free panoramic pair-level relative pose estimation.

`TRAIN360E` provides the necessary cautionary result: strong pair-level metrics do not guarantee strong full-trajectory behavior under direct adjacent-pair composition. The gap between pair-level success and trajectory-level ATE demonstrates that sequential accumulation remains a separate difficulty axis.

`SEQ360B` provides a useful partial ablation. It improves trajectory path ratio (1.7563 to 1.3502) and ATE SE3 (118.60 to 75.95), but not ATE Sim3 (27.56 to 27.57). Therefore, the evidence suggests that scale drift is only one part of the problem; the residual error increasingly looks like trajectory-shape, rotation-accumulation, or direction-accumulation error.

The `BASE360D` comparison should be interpreted carefully. It remains an official sequence VO pipeline, so it is not surprising that it retains advantages on trajectory-level metrics such as test Sim3 ATE. This work should not claim to have fully surpassed the official 360DVO pipeline as a complete VO system.

## 7. Ablations and Diagnostic Attempts
`SEQ360B` is the only retained positive diagnostic variant and should be understood as a partial success: it improves path length behavior and SE3 ATE but does not improve Sim3 trajectory shape. `SEQ360A` is retained only as a brief negative ablation, with status `no_improvement`; the local clip-consistency design did not reduce the main trajectory drift. `STRUCT360C` is retained only as an unstable challenger with status `evaluation_failed` and is not part of the final model selection.

## 8. Limitations
The current results have several clear limitations. First, pair-level success does not yet translate into full VO pipeline superiority. Second, direct sequential composition still accumulates drift. Third, ATE Sim3 remains difficult to improve, even when path/scale behavior is partially corrected. Fourth, `BASE360D` remains a mature official sequence VO pipeline and still holds advantages in trajectory-level evaluation. Fifth, the present work deliberately excludes explicit matching, RANSAC, PnP, and BA, so it does not yet include a learned-to-global optimization bridge.

## 9. Summary
In summary, the strongest claim supported by the retained evidence is that this work contributes an effective match-free panoramic pair-level relative pose estimator. `FINAL360I` is the current thesis main model for pair-level relative pose estimation. `TRAIN360E` shows that direct trajectory composition is feasible but still drifts. `SEQ360B` shows that scale correction helps path and SE3 behavior but does not solve trajectory shape. Future work should therefore move toward sequence-level refinement rather than overclaiming full VO pipeline superiority.

