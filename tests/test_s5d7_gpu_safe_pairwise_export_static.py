import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
E = REPO_ROOT / "tools" / "export_final_s5_pairwise_vectors_gpu_safe.py"
A = REPO_ROOT / "tools" / "audit_s5d7_validation_oom.py"


class TestS5D7GpuSafePairwiseExportStatic(unittest.TestCase):
    def test_tools_exist(self):
        self.assertTrue(E.exists())
        self.assertTrue(A.exists())

    def test_paths_and_classes(self):
        t = E.read_text(encoding="utf-8") + "\n" + A.read_text(encoding="utf-8")
        self.assertIn("reports/s5d7_gpu_safe_validation_and_pairwise_export_hook.md", t)
        self.assertIn("checkpoints/S5D7_gpu_safe_validation_and_pairwise_export_hook.json", t)
        self.assertIn("logs/s5d6_s6_lockdown_eval_only.log", t)
        for k in [
            "S5D7_VALIDATION_RECOVERED_PAIRWISE_EXPORTED",
            "S5D7_VALIDATION_RECOVERED_PAIRWISE_UNAVAILABLE",
            "S5D7_VALIDATION_BLOCKED_CUDA_OOM",
            "S5D7_PAIRWISE_EXPORT_ONLY",
            "S5D7_ERROR",
        ]:
            self.assertIn(k, t)

    def test_output_paths(self):
        t = E.read_text(encoding="utf-8") + "\n" + A.read_text(encoding="utf-8")
        self.assertIn("final_s5_pairwise_vectors_scene01_seq03.jsonl", t)
        self.assertIn("final_s5_pairwise_vectors_scene01_seq03.npz", t)
        self.assertIn("final_s5_pairwise_export_metadata.json", t)

    def test_caveat_and_safety(self):
        t = E.read_text(encoding="utf-8") + "\n" + A.read_text(encoding="utf-8")
        self.assertIn("diagnostic only", t)
        self.assertIn("gt_used_to_generate_predictions", t)
        self.assertIn("not_replaced_by_s5d7", t)


if __name__ == "__main__":
    unittest.main()
