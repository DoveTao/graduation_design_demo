from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestS5E12RealCorrespondenceFeatureGateStatic(unittest.TestCase):
    def test_config_exists(self):
        self.assertTrue((ROOT / "configs/s5e12_real_correspondence_feature_gate.yaml").exists())

    def test_required_tools_exist(self):
        for rel in [
            "tools/s5e12_extract_real_correspondence_features.py",
            "tools/s5e12_audit_correspondence_quality.py",
            "tools/s5e12_audit_essential_geometry.py",
            "tools/s5e12_observability_gate.py",
            "tools/export_s5e12_feature_gate_artifacts.py",
            "tools/evaluate_s5e12_feature_gate.py",
            "tools/compare_s5e12_s5e11_s5e10_s5e9_s5e8_s5e7_s5e6_s5e5_s5e4_s5e3_s5e2_orbslam3.py",
            "tools/s5e12_feature_dependency_check.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_real_correspondence_terms_present(self):
        txt = (ROOT / "tools/s5e12_extract_real_correspondence_features.py").read_text(encoding="utf-8")
        for term in [
            "ORB_create",
            "BFMatcher",
            "findFundamentalMat",
            "calcOpticalFlowPyrLK",
            "correspondence_failure_reason",
            "flow_failure_reason",
        ]:
            self.assertIn(term, txt)

    def test_no_fake_essential_geometry(self):
        txt = (ROOT / "tools/s5e12_audit_essential_geometry.py").read_text(encoding="utf-8")
        self.assertIn("strict_essential_geometry_available", txt)
        self.assertIn("pinhole_intrinsics_missing_or_non_pinhole_camera_model", txt)

    def test_classifications_present(self):
        txt = (ROOT / "tools/evaluate_s5e12_feature_gate.py").read_text(encoding="utf-8")
        for term in [
            "S5E12_FEATURE_GATE_PASSED",
            "S5E12_DEPENDENCY_READY",
            "S5E12_DEPENDENCY_BLOCKED",
            "S5E12_INTRINSICS_MISSING",
            "S5E12_CORRESPONDENCE_TOO_SPARSE",
            "S5E12_OBSERVABILITY_STILL_LIMITED",
            "S5E12_ESSENTIAL_GEOMETRY_AVAILABLE",
            "S5E12_PROXY_ONLY_REPEATED",
            "S5E12_REGRESSION",
        ]:
            self.assertIn(term, txt)

    def test_chinese_report_requirement_documented(self):
        txt = (ROOT / "configs/s5e12_real_correspondence_feature_gate.yaml").read_text(encoding="utf-8")
        self.assertIn("report_language: zh", txt)


if __name__ == "__main__":
    unittest.main()
