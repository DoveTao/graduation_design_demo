import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestS5E15Static(unittest.TestCase):
    def test_files_exist(self):
        for rel in [
            'configs/s5e15_scale_deunderfit_antiparallel.yaml',
            'tools/audit_s5e15_under_scale_and_antiparallel.py',
            'tools/train_s5e15_scale_antiparallel_refinement.py',
            'tools/export_s5e15_adjacent_dense_predictions.py',
            'tools/evaluate_s5e15_traceable_dense.py',
            'tools/compare_s5e15_s5e14_s5e13_s5e9_orbslam3.py',
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_paths_and_classifications(self):
        cfg = (ROOT / 'configs/s5e15_scale_deunderfit_antiparallel.yaml').read_text(encoding='utf-8')
        for t in [
            'reports/s5e15_scale_deunderfit_antiparallel_report.md',
            'reports/s5e15_s5e14_s5e13_s5e9_orbslam3_comparison.md',
            'checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json',
            'train_split_prior',
            '没有使用 eval GT 做 scale/sign calibration',
        ]:
            self.assertIn(t, cfg)

        txt = (ROOT / 'tools/evaluate_s5e15_traceable_dense.py').read_text(encoding='utf-8')
        for k in [
            'S5E15_SCALE_DEUNDERFIT_IMPROVED',
            'S5E15_SCALE_IMPROVED_ANTIPARALLEL_STILL_BAD',
            'S5E15_ANTIPARALLEL_IMPROVED_SCALE_STILL_BAD',
            'S5E15_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT',
            'rot_mean_deg', 'signed_tdir_mean_deg', 'anti_parallel_rate', 'tmag_median_ratio', 'path_ratio', 'ate',
        ]:
            self.assertIn(k, txt)

    def test_guard_and_no_policy_modification_terms(self):
        txt = (ROOT / 'tools/export_s5e15_adjacent_dense_predictions.py').read_text(encoding='utf-8')
        for k in ['uses_eval_gt_for_scale', 'uses_eval_gt_for_sign', 'uses_train_prior_scale', 'anti_parallel_guard_applied', 'scale_factor_source']:
            self.assertIn(k, txt)
        merged = '\n'.join((ROOT / p).read_text(encoding='utf-8') for p in [
            'tools/audit_s5e15_under_scale_and_antiparallel.py',
            'tools/train_s5e15_scale_antiparallel_refinement.py',
            'tools/export_s5e15_adjacent_dense_predictions.py',
            'tools/evaluate_s5e15_traceable_dense.py',
            'tools/compare_s5e15_s5e14_s5e13_s5e9_orbslam3.py',
        ])
        self.assertNotIn('final_clean_candidate_manifest.json', merged)
        self.assertNotIn('S5_clean_tmag_calibration_policy.json', merged)


if __name__ == '__main__':
    unittest.main()
