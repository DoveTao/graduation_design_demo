from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestStruct1BScaleGuardRepairStatic(unittest.TestCase):
    def test_required_files_exist(self):
        for rel in [
            "configs/struct1b_scale_guard_repair.yaml",
            "tools/train_struct1b_scale_guard_repair.py",
            "tools/export_struct1b_adjacent_dense_predictions.py",
            "tools/evaluate_struct1b_traceable_dense.py",
            "tools/audit_struct1b_scale_guard.py",
            "tools/compare_struct1b_struct1_s5e15_orbslam3.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_terms_present(self):
        cfg = (ROOT / "configs/struct1b_scale_guard_repair.yaml").read_text(encoding="utf-8")
        tool = (ROOT / "tools/train_struct1b_scale_guard_repair.py").read_text(encoding="utf-8")
        eval_text = (ROOT / "tools/evaluate_struct1b_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "hard_scale_guard_enabled: true",
            "w_path_candidates",
            "w_tmag_candidates",
            "record_raw_log_tmag: true",
            "record_bounded_log_tmag: true",
            "record_final_tmag: true",
            "preserve_geometry_token_pose_solver",
            "redesign_tdir_head: false",
        ]:
            self.assertIn(token, cfg)
        for token in [
            "raw_log_tmag",
            "bounded_log_tmag",
            "final_tmag",
            "hard_scale_guard_enabled",
            "prior_min_ratio",
            "prior_max_ratio",
        ]:
            self.assertIn(token, tool)
        for token in [
            "STRUCT1B_SCALE_REPAIR_SUCCESS_TDIR_STILL_BAD",
            "STRUCT1B_SCALE_PARTIAL_REPAIR",
            "STRUCT1B_SCALE_GUARD_NOT_EFFECTIVE",
            "STRUCT1B_SCALE_REPAIR_AND_GLOBAL_IMPROVEMENT",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, eval_text)


if __name__ == "__main__":
    unittest.main()
