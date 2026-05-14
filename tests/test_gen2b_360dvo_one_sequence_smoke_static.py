from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestGen2B360DVOOneSequenceSmokeStatic(unittest.TestCase):
    def test_tools_exist(self):
        for rel in [
            "tools/download_gen2b_360dvo_miniseq.py",
            "tools/summarize_gen2b_360dvo_smoke.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_required_terms_present(self):
        text = "\n".join(
            [
                (ROOT / "tools/download_gen2b_360dvo_miniseq.py").read_text(encoding="utf-8"),
                (ROOT / "tools/summarize_gen2b_360dvo_smoke.py").read_text(encoding="utf-8"),
            ]
        )
        for token in [
            "chris1004336379/360DVO",
            "one-sequence",
            "pair manifest",
            "DATA2",
            "no training",
            "fine-tune",
            "checkpoints/GEN2B_360DVO_one_sequence_smoke_audit.json",
            "reports/gen2b_360dvo_one_sequence_smoke_audit.md",
            "GEN2B_360DVO_ONE_SEQUENCE_READY",
            "GEN2B_360DVO_MOTION_AUDIT_READY",
            "GEN2B_360DVO_EXTERNAL_EVAL_READY",
            "GEN2B_360DVO_DOWNLOAD_BLOCKED",
            "GEN2B_360DVO_POSE_CONVENTION_BLOCKED",
            "GEN2B_360DVO_ADAPTER_NOT_READY",
            "GEN2B_360DVO_NOT_SUITABLE",
            "GEN2B_ERROR",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
