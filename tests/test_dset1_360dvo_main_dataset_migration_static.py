from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestDset1360DvoMainDatasetMigrationStatic(unittest.TestCase):
    def test_files_exist(self):
        for rel in [
            "configs/dset1_360dvo_main_dataset_migration.yaml",
            "tools/download_dset1_360dvo_sequences.py",
            "tools/build_dset1_360dvo_multi_sequence_manifest.py",
            "tools/audit_dset1_360dvo_motion_distribution.py",
            "tools/compare_dset1_360dvo_vs_legacy_dataset.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_terms_present(self):
        text = "\n".join(
            [
                (ROOT / "configs/dset1_360dvo_main_dataset_migration.yaml").read_text(encoding="utf-8"),
                (ROOT / "tools/download_dset1_360dvo_sequences.py").read_text(encoding="utf-8"),
                (ROOT / "tools/build_dset1_360dvo_multi_sequence_manifest.py").read_text(encoding="utf-8"),
                (ROOT / "tools/audit_dset1_360dvo_motion_distribution.py").read_text(encoding="utf-8"),
                (ROOT / "tools/compare_dset1_360dvo_vs_legacy_dataset.py").read_text(encoding="utf-8"),
            ]
        )
        for token in [
            "360DVO",
            "multi sequence",
            "split_by_sequence",
            "train",
            "val",
            "test",
            "no_training",
            "no_finetune",
            "DSET1_360DVO_MAIN_DATASET_READY",
            "DSET1_360DVO_MULTI_SEQUENCE_READY",
            "DSET1_360DVO_SMOKE_ONLY_READY",
            "DSET1_360DVO_DOWNLOAD_BLOCKED",
            "DSET1_360DVO_INSUFFICIENT_SEQUENCES",
            "DSET1_LEGACY_DATASET_QUALITY_RISK_CONFIRMED",
            "DSET1_DATASET_MIGRATION_BLOCKED",
            "DSET1_ERROR",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
