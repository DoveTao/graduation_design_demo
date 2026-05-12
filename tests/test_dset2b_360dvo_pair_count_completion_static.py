from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestDset2b360DvoPairCountCompletionStatic(unittest.TestCase):
    def test_config_exists(self):
        self.assertTrue((ROOT / "configs/dset2b_360dvo_pair_count_completion.yaml").exists())

    def test_tools_exist(self):
        for rel in [
            "tools/list_dset2b_360dvo_remote_sequences.py",
            "tools/download_dset2b_360dvo_sequences.py",
            "tools/audit_dset2b_360dvo_sequence_completeness.py",
            "tools/build_dset2b_360dvo_train360_manifest.py",
            "tools/audit_dset2b_train360_readiness.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_required_terms_present(self):
        text = "\n".join(
            (ROOT / rel).read_text(encoding="utf-8")
            for rel in [
                "configs/dset2b_360dvo_pair_count_completion.yaml",
                "tools/list_dset2b_360dvo_remote_sequences.py",
                "tools/download_dset2b_360dvo_sequences.py",
                "tools/audit_dset2b_360dvo_sequence_completeness.py",
                "tools/build_dset2b_360dvo_train360_manifest.py",
                "tools/audit_dset2b_train360_readiness.py",
            ]
        )
        for token in [
            "chris1004336379/360DVO",
            "sequence_completeness",
            "train360_ready",
            "no_training",
            "no_finetune",
            "snapshot_download",
            "HfApi",
            "valid_adjacent_pairs_possible",
            "valid_kstep_pairs_possible",
            "small_motion_fraction",
            "very_small_motion_fraction",
            "s5_locked_metrics_policy_unchanged",
            "DSET2B_TRAIN360_READY",
            "DSET2B_PAIR_COUNT_STILL_INSUFFICIENT",
            "DSET2B_DOWNLOAD_BLOCKED",
            "DSET2B_POSE_LIMITED_DATASET",
            "DSET2B_ADAPTER_LIMIT_BUG_SUSPECTED",
            "DSET2B_ERROR",
        ]:
            self.assertIn(token, text)

    def test_report_and_checkpoint_have_policy_statements(self):
        combined = "\n".join(
            (ROOT / rel).read_text(encoding="utf-8")
            for rel in [
                "tools/audit_dset2b_train360_readiness.py",
                "configs/dset2b_360dvo_pair_count_completion.yaml",
            ]
        )
        self.assertIn("S5 locked metrics/policy were not changed.", combined)
        self.assertIn("no_gt_calibration", combined)


if __name__ == "__main__":
    unittest.main()
