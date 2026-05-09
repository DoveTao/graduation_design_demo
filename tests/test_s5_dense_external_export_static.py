from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_SCRIPT = REPO_ROOT / "tools" / "export_s5_dense_trajectory.py"
COMPARISON_SCRIPT = REPO_ROOT / "tools" / "run_same_evaluator_comparison.py"
EXPORT_REPORT = REPO_ROOT / "reports" / "s5_dense_external_export_report.md"
COMPARISON_REPORT = REPO_ROOT / "reports" / "s5_orbslam3_same_evaluator_comparison.md"
EXPORT_JSON = REPO_ROOT / "checkpoints" / "S5D_dense_external_export.json"
COMPARISON_JSON = REPO_ROOT / "checkpoints" / "S5D_same_evaluator_comparison_results.json"


class S5DenseExternalExportStaticTest(unittest.TestCase):
    def test_scripts_exist(self) -> None:
        self.assertTrue(EXPORT_SCRIPT.is_file())
        self.assertTrue(COMPARISON_SCRIPT.is_file())

    def test_output_paths_are_referenced(self) -> None:
        text = (
            EXPORT_SCRIPT.read_text(encoding="utf-8")
            + COMPARISON_SCRIPT.read_text(encoding="utf-8")
            + EXPORT_REPORT.read_text(encoding="utf-8")
            + COMPARISON_REPORT.read_text(encoding="utf-8")
        )
        self.assertIn("external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt", text)
        self.assertIn("checkpoints/S5D_dense_external_export.json", text)
        self.assertIn("checkpoints/S5D_same_evaluator_comparison_results.json", text)

    def test_all_three_alignments_are_referenced(self) -> None:
        text = COMPARISON_SCRIPT.read_text(encoding="utf-8") + COMPARISON_REPORT.read_text(encoding="utf-8")
        self.assertIn("none", text)
        self.assertIn("se3", text)
        self.assertIn("sim3", text)
        self.assertIn("--alignment", text)

    def test_allowed_classifications_are_present(self) -> None:
        text = (
            EXPORT_SCRIPT.read_text(encoding="utf-8")
            + COMPARISON_SCRIPT.read_text(encoding="utf-8")
            + EXPORT_JSON.read_text(encoding="utf-8")
            + COMPARISON_JSON.read_text(encoding="utf-8")
        )
        for classification in [
            "S5D_DENSE_EXPORT_READY",
            "S5D_SPARSE_EXPORT_ONLY",
            "S5D_DENSE_EXPORT_UNAVAILABLE",
            "S5D_EXPORT_REJECTED_GT_LEAKAGE_RISK",
            "S5D_EXPORT_ERROR",
            "S5D_SAME_EVALUATOR_COMPARISON_COMPLETE",
            "S5D_SAME_EVALUATOR_COMPARISON_PARTIAL",
            "S5D_SAME_EVALUATOR_COMPARISON_BLOCKED",
            "S5D_SAME_EVALUATOR_COMPARISON_ERROR",
        ]:
            self.assertIn(classification, text)

    def test_gt_leakage_checks_are_implemented(self) -> None:
        text = EXPORT_SCRIPT.read_text(encoding="utf-8") + EXPORT_JSON.read_text(encoding="utf-8")
        self.assertIn("gt_leakage_check_passed", text)
        self.assertIn("gt_used_to_generate_prediction", text)
        self.assertIn("np.allclose", text)
        payload = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
        self.assertFalse(payload["safety_checks"]["gt_used_to_generate_prediction"])

    def test_s5_external_metrics_come_from_evaluator_outputs(self) -> None:
        script = COMPARISON_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("tools/evaluate_external_baseline_trajectory.py", script)
        self.assertIn("eval_alignment_{alignment}.json", script)
        for generated_s5_metric in [
            "21.681522044975264",
            "8.231468716451547",
            "4.07912293550008",
            "2.777267572676944",
        ]:
            self.assertNotIn(generated_s5_metric, script)

    def test_orbslam3_metrics_are_loaded_not_fabricated(self) -> None:
        script = COMPARISON_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("eval_alignment_{alignment}.json", script)
        self.assertIn("ORB1d_orbslam3_fisheye_evaluation_results.json", script)
        for known_metric in [
            "11.46685113827757",
            "0.30854441069248173",
            "0.224292165986624",
            "0.6013215859030837",
        ]:
            self.assertNotIn(known_metric, script)

    def test_scripts_do_not_modify_s5_policy_manifest_split_or_official_evaluator(self) -> None:
        text = EXPORT_SCRIPT.read_text(encoding="utf-8") + COMPARISON_SCRIPT.read_text(encoding="utf-8")
        forbidden_write_targets = [
            "S5_clean_tmag_calibration_policy.json\"",
            "final_clean_candidate_manifest.json\"",
            "final_clean_candidate_split",
            "tools/eval_clean_policy.py\"",
            "scripts/eval_s5_clean_policy.sh\"",
        ]
        for target in forbidden_write_targets:
            self.assertNotIn(f"{target}.write_text", text)
        self.assertNotIn("open(\"checkpoints/S5_clean_tmag_calibration_policy.json\", \"w", text)
        self.assertNotIn("open(\"checkpoints/final_clean_candidate_manifest.json\", \"w", text)

    def test_reports_distinguish_official_locked_from_external_comparison(self) -> None:
        text = EXPORT_REPORT.read_text(encoding="utf-8") + COMPARISON_REPORT.read_text(encoding="utf-8")
        self.assertIn("S5 official locked metrics are unchanged", text)
        self.assertIn("external same-evaluator comparison", text)
        self.assertIn("does not replace the official S5 locked result", text)


if __name__ == "__main__":
    unittest.main()
