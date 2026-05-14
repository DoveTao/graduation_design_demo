import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "configs" / "s5e4_temporal_direction_head.yaml",
    ROOT / "tools" / "train_s5e4_temporal_direction_candidate.py",
    ROOT / "tools" / "export_s5e4_adjacent_dense_predictions.py",
    ROOT / "tools" / "evaluate_s5e4_traceable_dense.py",
    ROOT / "tools" / "compare_s5e4_s5e3_s5e2_orbslam3.py",
]


class TestS5E4TemporalDirectionHeadStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(p.read_text(encoding="utf-8") for p in FILES)

    def test_files_exist(self):
        for p in FILES:
            self.assertTrue(p.exists(), p)

    def test_paths_referenced(self):
        t = self._text()
        self.assertIn("reports/s5e4_temporal_direction_head_report.md", t)
        self.assertIn("reports/s5e4_s5e3_s5e2_orbslam3_comparison.md", t)
        self.assertIn("checkpoints/S5E4_temporal_direction_head_candidate.json", t)
        self.assertIn("external_baselines/results/s5e4_traceable_dense", t)

    def test_allowed_classifications(self):
        t = self._text()
        for x in ["S5E4_SIGNED_TDIR_IMPROVED", "S5E4_TDIR_AND_TMAG_IMPROVED", "S5E4_ROT_PRESERVED_TDIR_PARTIAL", "S5E4_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT", "S5E4_TRAINING_BLOCKED", "S5E4_EXPORT_BLOCKED", "S5E4_ERROR"]:
            self.assertIn(x, t)

    def test_metrics_and_losses(self):
        t = self._text()
        for x in ["rot", "signed_tdir", "tdir_abs", "anti_parallel_rate", "tmag", "ATE", "path_ratio", "anti_parallel_penalty"]:
            self.assertIn(x, t)

    def test_no_gt_leakage_and_known_refs(self):
        t = self._text()
        self.assertIn("uses_scene01_seq03_for_training", t)
        self.assertIn("scene01/seq03 GT is used only for evaluation", t)
        for x in ["135.28985476811536", "12.109093390318419", "51.47429479273258", "32.77577273937451", "0.30854441069248173", "0.224292165986624"]:
            self.assertIn(x, t)

    def test_no_policy_manifest_split_or_evaluator_mutation(self):
        t = self._text()
        for bad in ["S5_clean_tmag_calibration_policy.json').write_text", "final_clean_candidate_manifest.json').write_text", "train_test_split.write", "s6_final_clean_candidate_lockdown_audit.py').write_text"]:
            self.assertNotIn(bad, t)
        self.assertIn("official_s5_unchanged", t)
        self.assertIn("not_official_replacement", t)
        self.assertIn("S5 locked metrics/policy unchanged", t)


if __name__ == "__main__":
    unittest.main()
