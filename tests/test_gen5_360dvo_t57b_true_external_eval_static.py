from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestGen5360DvoT57bTrueExternalEvalStatic(unittest.TestCase):
    def test_files_exist(self):
        for rel in [
            "tools/export_gen5_360dvo_t57b_predictions.py",
            "tools/evaluate_gen5_360dvo_t57b_external_eval.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_terms_present(self):
        text = "\n".join(
            [
                (ROOT / "tools/export_gen5_360dvo_t57b_predictions.py").read_text(encoding="utf-8"),
                (ROOT / "tools/evaluate_gen5_360dvo_t57b_external_eval.py").read_text(encoding="utf-8"),
            ]
        )
        for token in [
            "T57b_no_dt_multiscale_tmag_head_400",
            "360DVO",
            "rot_mean_deg",
            "signed_tdir_mean_deg",
            "tdir_abs_mean_deg",
            "anti_parallel_rate",
            "tmag_median_ratio",
            "path_ratio",
            "by_sequence",
            "by_k",
            "uses_360dvo_gt_for_calibration",
            "False",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
