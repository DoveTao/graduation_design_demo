# FINAL360I thesis-ready result paragraph



## English

On the DSET2C canonical split, FINAL360I keeps the STRUCT360B match-free design as the main candidate and performs a final retrain/model-selection pass without any architecture change, explicit matching, RANSAC, PnP, or bundle adjustment. The model remains LoFTR-inspired only at the hierarchy level: it uses a coarse-to-fine pose refinement pipeline without producing correspondence lists. Validation alone is used for model selection, and test is reserved for final confirmation. The recommended thesis main model is `FINAL360I_struct360b_final_selected`, which is compared against the recovered T57b ERP baseline and the HKUST official 360DVO baseline under the same DSET2C canonical split while preserving balanced translation direction and scale/path behavior.



## Chinese

在 DSET2C canonical split 上，FINAL360I 以 STRUCT360B 作为主候选模型，在不修改网络结构的前提下完成最终重训与模型选择；整个流程不使用 explicit matching、不输出 correspondence list，也不使用 RANSAC、PnP 或 BA。该方法仅在层级设计上借鉴 LoFTR 的 coarse-to-fine 思路，但具体实现仍然是无匹配的姿态残差细化。模型选择严格只基于 validation，test 仅用于最终确认。当前推荐的论文主模型为 `FINAL360I_struct360b_final_selected`，并已在相同 DSET2C canonical split 下与恢复出的 T57b ERP baseline 以及 HKUST official 360DVO baseline 完成对比，重点体现 translation direction 与 scale/path 的平衡表现。
