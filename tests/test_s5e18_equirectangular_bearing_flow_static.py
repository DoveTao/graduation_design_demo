import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class T(unittest.TestCase):
    def test_files(self):
        for r in ['configs/s5e18_equirectangular_bearing_flow.yaml','tools/extract_s5e18_spherical_bearing_flow_features.py','tools/audit_s5e18_bearing_flow_direction_signal.py','tools/export_s5e18_adjacent_dense_predictions.py','tools/evaluate_s5e18_traceable_dense.py','tools/compare_s5e18_s5e17_s5e15_orbslam3.py']:
            self.assertTrue((ROOT/r).exists(),r)
    def test_terms(self):
        txt=(ROOT/'tools/extract_s5e18_spherical_bearing_flow_features.py').read_text(encoding='utf-8')
        for t in ['equirectangular','spherical','bearing','lon','lat','pinhole_intrinsics_used']:
            self.assertIn(t,txt)
        ev=(ROOT/'tools/evaluate_s5e18_traceable_dense.py').read_text(encoding='utf-8')
        for t in ['rot_mean_deg','tdir_mean_deg','anti_parallel_rate','tmag_median_ratio','path_ratio','ate']:
            self.assertIn(t,ev)
        cfg=(ROOT/'configs/s5e18_equirectangular_bearing_flow.yaml').read_text(encoding='utf-8')
        self.assertIn('没有使用 eval GT 做 calibration',cfg)
    def test_no_forbidden(self):
        merged='\n'.join((ROOT/p).read_text(encoding='utf-8') for p in ['tools/extract_s5e18_spherical_bearing_flow_features.py','tools/audit_s5e18_bearing_flow_direction_signal.py','tools/export_s5e18_adjacent_dense_predictions.py'])
        self.assertNotIn('final_clean_candidate_manifest.json',merged)
if __name__=='__main__': unittest.main()
