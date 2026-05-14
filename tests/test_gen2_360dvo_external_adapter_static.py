from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestGen2360DVOExternalAdapterStatic(unittest.TestCase):
    def test_config_and_tools_exist(self):
        for rel in [
            "configs/gen2_360dvo_adapter.yaml",
            "tools/inspect_gen2_360dvo_dataset.py",
            "tools/build_gen2_360dvo_pair_manifest.py",
            "tools/audit_gen2_360dvo_motion_distribution.py",
            "tools/evaluate_gen2_360dvo_external_smoke.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_required_terms_present(self):
        text = "\n".join(
            [
                (ROOT / "configs/gen2_360dvo_adapter.yaml").read_text(encoding="utf-8"),
                (ROOT / "tools/inspect_gen2_360dvo_dataset.py").read_text(encoding="utf-8"),
                (ROOT / "tools/build_gen2_360dvo_pair_manifest.py").read_text(encoding="utf-8"),
                (ROOT / "tools/audit_gen2_360dvo_motion_distribution.py").read_text(encoding="utf-8"),
                (ROOT / "tools/evaluate_gen2_360dvo_external_smoke.py").read_text(encoding="utf-8"),
            ]
        )
        for token in [
            "360DVO",
            "pair_manifest",
            "R_BA",
            "t_BA_B",
            "tdir_B",
            "DATA2",
            "no_training",
            "GEN2_360DVO_ADAPTER_READY",
            "GEN2_360DVO_DATA_NOT_AVAILABLE",
            "GEN2_360DVO_POSE_CONVENTION_BLOCKED",
            "GEN2_360DVO_SMOKE_EVAL_READY",
            "GEN2_360DVO_EXTERNAL_EVAL_BLOCKED",
            "GEN2_360DVO_NOT_SUITABLE_AFTER_INSPECTION",
            "GEN2_ERROR",
            "s5_locked_metrics_policy_unchanged",
            "reports/gen2_360dvo_external_adapter_smoke_eval.md",
            "checkpoints/GEN2_360DVO_external_adapter_smoke_eval.json",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
