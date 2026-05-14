import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestS5E16Static(unittest.TestCase):
    def test_files(self):
        for rel in [
            'configs/s5e16_confidence_calibrated_sign_scale_router.yaml',
            'tools/audit_s5e16_router_inputs.py',
            'tools/train_s5e16_confidence_router.py',
            'tools/export_s5e16_adjacent_dense_predictions.py',
            'tools/evaluate_s5e16_traceable_dense.py',
            'tools/compare_s5e16_s5e15_s5e14_s5e13_s5e9_orbslam3.py',
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_terms(self):
        cfg = (ROOT / 'configs/s5e16_confidence_calibrated_sign_scale_router.yaml').read_text(encoding='utf-8')
        for t in [
            'reports/s5e16_confidence_calibrated_sign_scale_router_report.md',
            'reports/s5e16_s5e15_s5e14_s5e13_s5e9_orbslam3_comparison.md',
            'checkpoints/S5E16_confidence_calibrated_sign_scale_router.json',
            '没有使用 eval GT 做 router/scale/sign calibration',
        ]:
            self.assertIn(t, cfg)

        exp = (ROOT / 'tools/export_s5e16_adjacent_dense_predictions.py').read_text(encoding='utf-8')
        for t in ['direction_route', 'scale_route', 'sign_guard_route', 'router_confidence', 'fallback_prior']:
            self.assertIn(t, exp)

        eva = (ROOT / 'tools/evaluate_s5e16_traceable_dense.py').read_text(encoding='utf-8')
        for t in [
            'S5E16_ROUTER_IMPROVED_GEOMETRY',
            'S5E16_SCALE_IMPROVED_ANTIPARALLEL_STILL_BAD',
            'S5E16_ANTIPARALLEL_IMPROVED_SCALE_STILL_GAP',
            'S5E16_ROUTER_NO_IMPROVEMENT',
            'rot_mean_deg', 'signed_tdir_mean_deg', 'anti_parallel_rate', 'tmag_median_ratio', 'path_ratio', 'ate',
            'commits_created', 'working_tree_clean'
        ]:
            self.assertIn(t, eva)

    def test_no_forbidden_mods(self):
        merged = '\n'.join((ROOT / p).read_text(encoding='utf-8') for p in [
            'tools/audit_s5e16_router_inputs.py',
            'tools/train_s5e16_confidence_router.py',
            'tools/export_s5e16_adjacent_dense_predictions.py',
            'tools/evaluate_s5e16_traceable_dense.py',
            'tools/compare_s5e16_s5e15_s5e14_s5e13_s5e9_orbslam3.py',
        ])
        self.assertNotIn('final_clean_candidate_manifest.json', merged)
        self.assertNotIn('S5_clean_tmag_calibration_policy.json', merged)


if __name__ == '__main__':
    unittest.main()
