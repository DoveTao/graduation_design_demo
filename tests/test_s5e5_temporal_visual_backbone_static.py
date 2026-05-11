import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "configs" / "s5e5_temporal_visual_backbone_geometry.yaml",
    ROOT / "tools" / "s5e5_temporal_visual_lib.py",
    ROOT / "tools" / "train_s5e5_temporal_visual_candidate.py",
    ROOT / "tools" / "export_s5e5_adjacent_dense_predictions.py",
    ROOT / "tools" / "evaluate_s5e5_traceable_dense.py",
    ROOT / "tools" / "compare_s5e5_s5e4_s5e3_s5e2_orbslam3.py",
]


class TestS5E5TemporalVisualBackboneStatic(unittest.TestCase):
    def _text(self):
        return "\n".join(p.read_text(encoding="utf-8") for p in FILES)

    def test_files_exist(self):
        for p in FILES:
            self.assertTrue(p.exists(), p)

    def test_paths_and_classifications(self):
        t = self._text()
        for x in [
            "reports/s5e5_temporal_visual_backbone_report.md",
            "reports/s5e5_s5e4_s5e3_s5e2_orbslam3_comparison.md",
            "checkpoints/S5E5_temporal_visual_backbone_geometry_candidate.json",
            "external_baselines/results/s5e5_traceable_dense",
            "S5E5_GEOMETRY_IMPROVED",
            "S5E5_TDIR_IMPROVED_TMAG_STILL_BAD",
            "S5E5_TMAG_IMPROVED_TDIR_STILL_BAD",
            "S5E5_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
            "S5E5_TRAINING_BLOCKED",
            "S5E5_EXPORT_BLOCKED",
            "S5E5_ERROR",
        ]:
            self.assertIn(x, t)

    def test_metrics_losses_and_visual_terms(self):
        t = self._text()
        for x in [
            "rot",
            "signed_tdir",
            "tdir_abs",
            "anti_parallel_rate",
            "tmag",
            "ATE",
            "path_ratio",
            "anti_parallel_penalty",
            "tdir_abs_aux",
            "tmag_log",
            "temporal_visual_backbone",
            "ordered image pair",
        ]:
            self.assertIn(x, t)

    def test_no_gt_leakage_and_known_refs(self):
        t = self._text()
        self.assertIn("uses_scene01_seq03_for_training", t)
        self.assertIn("scene01/seq03 GT is used only for evaluation", t)
        for x in [
            "51.47429479273258",
            "46.07968707914154",
            "12.109093390318423",
            "2.1345479454214416",
            "0.30854441069248173",
            "0.224292165986624",
        ]:
            self.assertIn(x, t)

    def test_no_mutation_of_locked_assets(self):
        t = self._text()
        for bad in [
            "S5_clean_tmag_calibration_policy.json').write_text",
            "final_clean_candidate_manifest.json').write_text",
            "train_test_split.write",
            "s6_final_clean_candidate_lockdown_audit.py').write_text",
        ]:
            self.assertNotIn(bad, t)
        self.assertIn("official_s5_locked_metrics_unchanged", t)
        self.assertIn("not_official_replacement", t)


if __name__ == "__main__":
    unittest.main()
