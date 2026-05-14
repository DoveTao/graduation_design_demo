from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestArch2PParameterSensitivityStatic(unittest.TestCase):
    def test_required_files_exist(self):
        for rel in [
            "configs/arch2p_parameter_sweep.yaml",
            "tools/run_arch2p_parameter_sweep.py",
            "tools/audit_arch2p_sweep_validity.py",
            "tools/summarize_arch2p_parameter_sweep.py",
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_sweep_terms_present(self):
        text = (ROOT / "configs/arch2p_parameter_sweep.yaml").read_text(encoding="utf-8")
        for token in [
            "delta_alpha:",
            "delta_clip_norm:",
            "high_confidence_quantile:",
            "softcorr_temperature:",
            "w_tdir:",
            "w_path:",
            "w_delta_l2:",
            "train_steps:",
            "50.3530",
            "3.9682",
        ]:
            self.assertIn(token, text)

    def test_rule_terms_present(self):
        text = (ROOT / "configs/arch2p_parameter_sweep.yaml").read_text(encoding="utf-8")
        for token in [
            "uses_eval_gt_for_training: false",
            "uses_eval_gt_for_gate: false",
            "uses_orbslam3_teacher: false",
        ]:
            self.assertIn(token, text)

    def test_final_classifications_present(self):
        text = (ROOT / "tools/summarize_arch2p_parameter_sweep.py").read_text(encoding="utf-8")
        for token in [
            "ARCH2P_FOUND_IMPROVED_CONFIG",
            "ARCH2P_SUBSET_ONLY_NO_GLOBAL_GAIN",
            "ARCH2P_NO_CONFIG_BEATS_S5E15",
            "ARCH2P_SWEEP_INVALID",
            "ARCH2P_SWEEP_BLOCKED",
            "ARCH2P_ERROR",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
