from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestData3LegacyPanoImageQualityStatic(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue((ROOT / "tools/audit_data3_legacy_pano_image_quality.py").exists())

    def test_terms_present(self):
        text = (ROOT / "tools/audit_data3_legacy_pano_image_quality.py").read_text(encoding="utf-8")
        for token in [
            "low_texture_fraction",
            "seam_score_vertical",
            "seam_score_horizontal",
            "blur_score_laplacian",
            "exposure_discontinuity",
            "quality_vs_s5e15_signed_tdir",
            "quality_vs_struct1b_tmag_ratio",
            "diagnostic",
            "DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED",
            "DATA3_LEGACY_IMAGE_QUALITY_RISK_SUSPECTED",
            "DATA3_LEGACY_IMAGE_QUALITY_NOT_MAJOR",
            "DATA3_LEGACY_IMAGES_NOT_AVAILABLE",
            "DATA3_INSUFFICIENT_ARTIFACTS",
            "DATA3_ERROR",
            "s5_locked_metrics_policy_unchanged",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
