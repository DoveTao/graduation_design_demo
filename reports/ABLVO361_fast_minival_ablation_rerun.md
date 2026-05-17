# ABLVO361_fast_minival_ablation_rerun

## 1. Executive summary
- ablation executed true/false: `true`
- models trained: `['ABLVO360_PlainPairVO', 'ABLVO360_NoSphericalGeometry', 'ABLVO360_NoCrossImageInteraction']`
- models skipped: `['ABLVO360_SingleStagePoseRegression']`
- selected full baseline: `FINAL360I_struct360b_final_selected`
- classification: `completed`

## 2. Experimental setup
- train/val/test manifests: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- training epochs: `3`
- seed: `0`
- optimizer: `AdamW`
- mini-val subset count: `512`
- test used for selection: `false`
- subset info: `{'subset_used': True, 'subset_count': 64, 'original_count': 12154, 'subset_seed': 0}`
- mini-val summary: `{'subset_used': True, 'subset_count': 512, 'original_count': 5342, 'seed': 0, 'group_count': 18, 'groups_selected': 18, 'tmag_bucket_edges': [0.03, 0.08, 0.2, 0.5], 'rot_bucket_edges_deg': [5.0, 15.0, 30.0, 60.0, 120.0], 'sequence_count_selected': 2, 'top_sequence_counts': {'mountains': 292, 'downhill_biking': 220}, 'selected_tmag_bucket_histogram': {'2': 77, '3': 158, '4': 276, '1': 1}, 'selected_rot_bucket_histogram': {'0': 511, '1': 1}}`
- system info: `{'device': 'cuda', 'cuda_available': True, 'torch_version': '2.9.1+cu128', 'gpu_name': 'NVIDIA GeForce RTX 3060 Laptop GPU'}`
- runtime sec: `3271.36`

## 3. Validation selection table
- {'model_name': 'ABLVO360_PlainPairVO', 'executed': True, 'best_epoch': 2, 'minival_score': 154.46562296958135, 'val_score': 155.6863916217744, 'val_signed_tdir_mean': 97.43632540216363, 'val_anti_parallel_rate': 0.5990265818045676, 'val_rot_mean': 5.106030519581698, 'val_tmag_median_ratio': 1.5596525244134307, 'val_path_ratio': 0.7189428088908575, 'selected_checkpoint': '/home/dovetao/graduation_design_demo/checkpoints/ABLVO360_PlainPairVO/best_val.pt', 'test_used_for_selection': False}
- {'model_name': 'ABLVO360_NoSphericalGeometry', 'executed': True, 'best_epoch': 1, 'minival_score': 148.95501441779967, 'val_score': 150.61218441417589, 'val_signed_tdir_mean': 95.07216829006609, 'val_anti_parallel_rate': 0.5969674279296143, 'val_rot_mean': 1.2811739897736893, 'val_tmag_median_ratio': 1.28714391857561, 'val_path_ratio': 0.6265666595036484, 'selected_checkpoint': '/home/dovetao/graduation_design_demo/checkpoints/ABLVO360_NoSphericalGeometry/best_val.pt', 'test_used_for_selection': False}
- {'model_name': 'ABLVO360_NoCrossImageInteraction', 'executed': True, 'best_epoch': 1, 'minival_score': 149.86214029366334, 'val_score': 151.62345786813603, 'val_signed_tdir_mean': 92.73668722829896, 'val_anti_parallel_rate': 0.6282291276675402, 'val_rot_mean': 2.3684908600934813, 'val_tmag_median_ratio': 1.324321369080562, 'val_path_ratio': 0.6110020325994311, 'selected_checkpoint': '/home/dovetao/graduation_design_demo/checkpoints/ABLVO360_NoCrossImageInteraction/best_val.pt', 'test_used_for_selection': False}
- {'model_name': 'ABLVO360_SingleStagePoseRegression', 'executed': False, 'skip_reason': 'resource_limited_optional_group_D', 'test_used_for_selection': False}

## 4. Test results
| model | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABLVO360_PlainPairVO | 48.68811067066032 | 0.20071118135124458 | 0.6458483659657839 | 0.5082019066288274 | 1.0 |
| ABLVO360_NoSphericalGeometry | 47.32440502655299 | 0.20090873172659027 | 0.6150483309923794 | 0.48372296542035503 | 1.0 |
| ABLVO360_NoCrossImageInteraction | 52.43926329996059 | 0.20229158435401026 | 0.5504093571251074 | 0.4328696029867613 | 1.0 |
| FINAL360I_full_model | 45.264702006380205 | 0.20169893322797314 | 0.8344251368086006 | 0.6403519796204528 | 1.0 |

## 5. Compliance checklist
- training_executed = true
- test_used_for_selection = false
- full_model_checkpoint_modified = false
- metrics_modified_existing = false
- checkpoints_committed = false
- raw_data_committed = false
- explicit_matching_used = false
- ransac_used = false
- pnp_used = false
- ba_used = false
