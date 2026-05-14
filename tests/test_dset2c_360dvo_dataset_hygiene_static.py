from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestDset2c360DvoDatasetHygieneStatic(unittest.TestCase):
    def test_config_exists(self):
        self.assertTrue((ROOT / "configs/dset2c_360dvo_dataset_hygiene.yaml").exists())

    def test_tools_exist(self):
        for rel in [
            "tools/audit_dset2c_360dvo_local_tree.py",
            "tools/build_dset2c_canonical_train360_manifest.py",
            "tools/validate_dset2c_canonical_manifest.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_terms_present(self):
        text = "\n".join(
            (ROOT / rel).read_text(encoding="utf-8")
            for rel in [
                "configs/dset2c_360dvo_dataset_hygiene.yaml",
                "tools/audit_dset2c_360dvo_local_tree.py",
                "tools/build_dset2c_canonical_train360_manifest.py",
                "tools/validate_dset2c_canonical_manifest.py",
            ]
        )
        for token in [
            "QUARANTINE_EMPTY_IMAGE_DIR",
            "QUARANTINE_IMAGE_MISSING",
            "QUARANTINE_POSE_MISSING",
            "QUARANTINE_TIMESTAMP_MISSING",
            "QUARANTINE_LOW_PAIR_COUNT",
            "QUARANTINE_UNKNOWN_FORMAT",
            "USABLE_FOR_TRAIN360",
            "USABLE_PARTIAL_HIGH_YIELD",
            "canonical_manifest",
            "train360_ready_after_hygiene",
            "no_training",
            "no_finetune",
            "DSET2C_CANONICAL_MANIFEST_READY",
            "DSET2C_TRAIN360_READY_AFTER_HYGIENE",
            "DSET2C_TRAIN360_NOT_READY_AFTER_CLEANING",
            "DSET2C_DSET2B_MANIFEST_REFERENCES_BAD_SEQUENCES",
            "DSET2C_LOCAL_TREE_DIRTY_BUT_CANONICAL_READY",
            "DSET2C_DATASET_HYGIENE_ERROR",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)

    def test_policy_statement_present(self):
        text = (ROOT / "tools/validate_dset2c_canonical_manifest.py").read_text(encoding="utf-8")
        self.assertIn("S5 locked metrics/policy were not changed.", text)


if __name__ == "__main__":
    unittest.main()
