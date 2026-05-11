import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "evaluate_component_pose_diagnostics.py"


class TestComponentPoseDiagnosticsStatic(unittest.TestCase):
    def test_tool_exists(self):
        self.assertTrue(TOOL.exists(), str(TOOL))

    def test_required_paths_and_outputs_are_referenced(self):
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("reports/s5_orbslam3_component_diagnostics.md", text)
        self.assertIn("checkpoints/S5D2_component_diagnostics.json", text)
        self.assertIn("s5_full_component_diagnostics.json", text)
        self.assertIn("orbslam3_tracked_component_diagnostics.json", text)
        self.assertIn("s5_on_orb_matched_component_diagnostics.json", text)
        self.assertIn("orbslam3_on_matched_component_diagnostics.json", text)

    def test_allowed_classifications_and_metric_names_exist(self):
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("S5D2_COMPONENT_DIAGNOSTICS_COMPLETE", text)
        self.assertIn("S5D2_COMPONENT_DIAGNOSTICS_PARTIAL", text)
        self.assertIn("S5D2_COMPONENT_DIAGNOSTICS_BLOCKED", text)
        self.assertIn("S5D2_COMPONENT_DIAGNOSTICS_ERROR", text)
        self.assertIn("rot_mean_deg", text)
        self.assertIn("tdir_mean_deg", text)
        self.assertIn("tdir_mean_cosine", text)
        self.assertIn("tmag_mean_log_error", text)
        self.assertIn("tmag_mean_ratio", text)

    def test_formulas_and_concepts_exist(self):
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("inverse(T_i) @ T_j", text)
        self.assertIn("arccos", text)
        self.assertIn("log(norm(t_est) / norm(t_gt))", text)
        self.assertIn("qx qy qz qw", text)

    def test_no_protected_file_modification_strings(self):
        text = TOOL.read_text(encoding="utf-8")
        self.assertNotIn("S5_clean_tmag_calibration_policy.json\", \"w", text)
        self.assertNotIn("final_clean_candidate_manifest.json\", \"w", text)
        self.assertNotIn("split = ", text)
        self.assertNotIn("official evaluator modified", text.lower())

    def test_report_caveats_present(self):
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("does not replace the official S5 locked result", text)
        self.assertIn("same_input_protocol", text)
        self.assertIn("fitted KB8 compatibility approximation", text)
        self.assertIn("partial coverage", text)


if __name__ == "__main__":
    unittest.main()
