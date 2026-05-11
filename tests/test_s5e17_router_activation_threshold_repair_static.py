import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class T(unittest.TestCase):
    def test_files(self):
        for r in ['configs/s5e17_router_activation_threshold_repair.yaml','tools/audit_s5e17_router_activation_failure.py','tools/tune_s5e17_router_thresholds_without_eval_gt.py','tools/export_s5e17_adjacent_dense_predictions.py','tools/evaluate_s5e17_traceable_dense.py','tools/compare_s5e17_s5e16_s5e15_orbslam3.py']:
            self.assertTrue((ROOT/r).exists(),r)
    def test_terms(self):
        txt=(ROOT/'tools/evaluate_s5e17_traceable_dense.py').read_text(encoding='utf-8')
        for t in ['S5E17_ROUTER_ACTIVATION_REPAIRED_GEOMETRY_IMPROVED','S5E17_ROUTER_ACTIVATION_REPAIRED_NO_GEOMETRY_IMPROVEMENT','S5E17_ROUTER_THRESHOLD_REPAIR_DIAGNOSTIC_ONLY','high_confidence_edges','medium_confidence_edges','low_confidence_edges','sign_guarded_edges','train_prior_scale_edges','rot_mean_deg','signed_tdir_mean_deg','anti_parallel_rate','tmag_median_ratio','path_ratio','ate','commits_created']:
            self.assertIn(t,txt)
        cfg=(ROOT/'configs/s5e17_router_activation_threshold_repair.yaml').read_text(encoding='utf-8')
        self.assertIn('没有使用 eval GT 选择 router threshold',cfg)
    def test_no_forbidden(self):
        merged='\n'.join((ROOT/p).read_text(encoding='utf-8') for p in ['tools/audit_s5e17_router_activation_failure.py','tools/tune_s5e17_router_thresholds_without_eval_gt.py','tools/export_s5e17_adjacent_dense_predictions.py','tools/evaluate_s5e17_traceable_dense.py'])
        self.assertNotIn('final_clean_candidate_manifest.json',merged)
if __name__=='__main__': unittest.main()
