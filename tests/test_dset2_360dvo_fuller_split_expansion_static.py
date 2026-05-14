from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestDset2360DvoFullerSplitExpansionStatic(unittest.TestCase):
    def test_files_exist(self):
        for rel in [
            "configs/dset2_360dvo_fuller_split_expansion.yaml",
            "tools/download_dset2_360dvo_fuller_sequences.py",
            "tools/build_dset2_360dvo_fuller_manifest.py",
            "tools/audit_dset2_360dvo_fuller_motion_distribution.py",
            "tools/summarize_dset2_360dvo_fuller_split.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_terms_present(self):
        text = "\n".join(
            [
                (ROOT / "configs/dset2_360dvo_fuller_split_expansion.yaml").read_text(encoding="utf-8"),
                (ROOT / "tools/download_dset2_360dvo_fuller_sequences.py").read_text(encoding="utf-8"),
                (ROOT / "tools/build_dset2_360dvo_fuller_manifest.py").read_text(encoding="utf-8"),
                (ROOT / "tools/audit_dset2_360dvo_fuller_motion_distribution.py").read_text(encoding="utf-8"),
                (ROOT / "tools/summarize_dset2_360dvo_fuller_split.py").read_text(encoding="utf-8"),
            ]
        )
        for token in [
            "360DVO",
            "fuller split",
            "train_sequences",
            "val_sequences",
            "test_sequences",
            "split_by_sequence",
            "no_training",
            "no_finetune",
            "min_total_sequences",
            "min_train_pairs",
            "DSET2_360DVO_FULLER_SPLIT_READY",
            "DSET2_360DVO_TRAIN360_READY",
            "DSET2_360DVO_INSUFFICIENT_SEQUENCES",
            "DSET2_360DVO_DOWNLOAD_BLOCKED",
            "DSET2_360DVO_SPLIT_IMBALANCED",
            "DSET2_360DVO_MANIFEST_BLOCKED",
            "DSET2_ERROR",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
