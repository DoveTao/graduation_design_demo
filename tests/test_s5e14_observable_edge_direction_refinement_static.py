import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestS5E14Static(unittest.TestCase):
    def test_s5e14_required_files_exist(self):
        paths = [
            "configs/s5e14_observable_edge_direction_refinement.yaml",
            "tools/audit_s5e14_s5e13_direction_errors.py",
            "tools/train_s5e14_observable_direction_refinement.py",
            "tools/export_s5e14_adjacent_dense_predictions.py",
            "tools/evaluate_s5e14_traceable_dense.py",
            "tools/compare_s5e14_s5e13_s5e12_s5e9_orbslam3.py",
        ]
        for rel in paths:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_paths_and_classifications_and_terms_present(self):
        cfg = (ROOT / "configs/s5e14_observable_edge_direction_refinement.yaml").read_text(encoding="utf-8")
        for t in [
            "reports/s5e14_observable_edge_direction_refinement_report.md",
            "reports/s5e14_s5e13_s5e12_s5e9_orbslam3_comparison.md",
            "checkpoints/S5E14_observable_edge_direction_refinement_candidate.json",
            "strict essential geometry",
            "scene01/seq03 GT",
        ]:
            self.assertIn(t, cfg)

        eval_text = (ROOT / "tools/evaluate_s5e14_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "S5E14_DIRECTION_REFINEMENT_IMPROVED",
            "S5E14_OBSERVABLE_IMPROVED_OVERALL_STILL_BAD",
            "S5E14_UNDERSCALE_REDUCED_TDIR_NOT_IMPROVED",
            "S5E14_INFERENCE_GATING_ONLY_NO_TRAIN_SIGNAL",
            "S5E14_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
            "overall",
            "observable_edges",
            "reliable_original_edges",
            "reliable_strict_edges",
            "unobservable_edges",
            "small_motion_edges",
            "anti_parallel_rate",
            "tmag_p95_ratio",
            "path_ratio",
            "ate",
        ]:
            self.assertIn(token, eval_text)

        export_text = (ROOT / "tools/export_s5e14_adjacent_dense_predictions.py").read_text(encoding="utf-8")
        for token in [
            "reliable_strict",
            "inference_blend_weight",
            "fallback_direction_source",
            "uses_eval_gt_for_calibration",
            "strict_essential_geometry_used",
        ]:
            self.assertIn(token, export_text)

    def test_caveats_and_protection_terms_present(self):
        report_text = (ROOT / "tools/evaluate_s5e14_traceable_dense.py").read_text(encoding="utf-8")
        for token in [
            "no GT leakage",
            "no eval GT calibration",
            "official S5 locked result",
            "S5 locked metrics/policy unchanged",
            "strict essential geometry",
        ]:
            self.assertIn(token, report_text)

    def test_no_forbidden_modification_terms_in_s5e14_scripts(self):
        merged = "\n".join(
            (ROOT / p).read_text(encoding="utf-8")
            for p in [
                "tools/audit_s5e14_s5e13_direction_errors.py",
                "tools/train_s5e14_observable_direction_refinement.py",
                "tools/export_s5e14_adjacent_dense_predictions.py",
                "tools/evaluate_s5e14_traceable_dense.py",
                "tools/compare_s5e14_s5e13_s5e12_s5e9_orbslam3.py",
            ]
        )
        self.assertNotIn("final_clean_candidate_manifest.json", merged)
        self.assertIn("train/test split", (ROOT / "configs/s5e14_observable_edge_direction_refinement.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
