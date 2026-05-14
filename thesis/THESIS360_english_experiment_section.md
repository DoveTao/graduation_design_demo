# THESIS360 English Experiment Section

## Experimental Setup
All experiments are reported on the DSET2C canonical split built on top of the 360DVO/HKUST data source. The repository uses a manifest-native protocol in which train/val/test samples are read from canonical JSONL manifests, rather than discovered through raw directory globbing. Random pair splitting is explicitly disallowed. This protocol is important because the thesis evaluates both pair-level relative pose quality and sequence-level trajectory composition under a stable sequence-aware data split.

The evaluation is divided into two layers. Pair-level evaluation uses translation component metrics, including signed translation-direction mean, anti-parallel rate, translation-magnitude median ratio, and path ratio. Trajectory-level evaluation composes adjacent relative poses into full trajectories and then reports ATE under none, SE3, and Sim3 alignment, together with the trajectory path ratio. These two layers should not be conflated: pair-level metrics measure local relative pose quality, whereas trajectory metrics measure accumulated sequential drift.

The method remains strictly match-free. No explicit matching, RANSAC, PnP, or bundle adjustment is introduced into the learned pair-level pipeline. No ORB-SLAM3 teacher, HKUST 360DVO teacher, or BASE360 outputs are used as training inputs.

## Compared Methods
The current visible comparison set is intentionally compact. `FINAL360I_struct360b_final_selected` is the retained pair-level thesis main model. `TRAIN360E` composes `FINAL360I` adjacent-pair predictions into full trajectories for ATE evaluation. `SEQ360B` is the only retained sequence-scale variant and focuses on lightweight log-scale correction. `BASE360D` is the retained HKUST official 360DVO external baseline. `T57b` is a recovered legacy external reference.

Two additional experiments are retained only as status summaries: `SEQ360A` (`no_improvement`) and `STRUCT360C` (`evaluation_failed`). They are not included in the main comparison table as successful competing methods.

## Results and Discussion
`FINAL360I` achieves strong pair-level test metrics: signed translation-direction mean 45.264702, anti-parallel rate 0.201699, translation-magnitude median ratio 0.834425, and path ratio 0.640352. These results are much stronger than the recovered `T57b` reference (111.964932, 0.674672, 0.175156, 0.148787) and also much stronger than the trajectory-derived `BASE360D` component baseline (128.402578, 0.814895, 0.039910, 0.043542). This supports the central thesis claim: the main contribution of the work lies in match-free panoramic pair-level relative pose estimation.

However, the comparison with `BASE360D` is only partially homogeneous, because its component numbers are recovered from official trajectory outputs rather than produced by the same pair-level forward interface. Moreover, trajectory-level results show that pair-level quality does not automatically transfer into sequence-level stability. `TRAIN360E` reports test ATE none / SE3 / Sim3 of 222.568564 / 118.603779 / 27.564662, with a trajectory path ratio of 1.756343, indicating substantial drift under direct adjacent-pair composition.

`SEQ360B` partially mitigates this issue. It improves the trajectory path ratio to 1.350237 and lowers SE3 ATE to 75.946913, but its Sim3 ATE remains almost unchanged at 27.567211. This indicates that the variant mainly corrects scale/path drift rather than trajectory-shape error. By contrast, `BASE360D`, as an official sequence VO pipeline, still provides a much stronger trajectory-level anchor, especially on Sim3 ATE (2.974466). Therefore, this thesis should not claim that the pair-level method fully surpasses official 360DVO as a complete VO system.

## Limitations
The most important limitation is conceptual: pair-level superiority is not the same as end-to-end VO superiority. Direct sequential composition still accumulates drift, Sim3 trajectory error remains hard to improve, and the official `BASE360D` pipeline remains stronger or at least not replaceable on full trajectory evaluation. These limitations are consistent with the current match-free pair-level design, which intentionally excludes explicit matching and global geometric optimization.

