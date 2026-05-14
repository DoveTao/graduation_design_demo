from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestGen1ExternalPanoDatasetFeasibilityStatic(unittest.TestCase):
    def test_audit_tool_exists(self):
        self.assertTrue((ROOT / "tools/audit_gen1_external_pano_datasets.py").exists())

    def test_required_terms_present(self):
        text = (ROOT / "tools/audit_gen1_external_pano_datasets.py").read_text(encoding="utf-8")
        for token in [
            "reports/gen1_external_pano_dataset_feasibility_audit.md",
            "checkpoints/GEN1_external_pano_dataset_feasibility_audit.json",
            "360DVO Dataset",
            "Stanford 2D-3D-S",
            "360-Indoor",
            "ERP",
            "pose",
            "timestamp",
            "TUM",
            "relative_pose_A_to_B_in_B",
            "adapter_schema",
            "DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT",
            "GEN1_360DVO_PRIMARY_CANDIDATE",
            "GEN1_2D3DS_PRIMARY_CANDIDATE",
            "GEN1_EXTERNAL_DATASET_FEASIBLE",
            "GEN1_EXTERNAL_DATASET_DIFFICULT_BUT_POSSIBLE",
            "GEN1_NO_SUITABLE_EXTERNAL_DATASET_FOUND",
            "GEN1_AUDIT_ERROR",
            "不训练模型",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
