from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_PY = REPO_ROOT / "tools" / "verify_final_s5_candidate.py"
VERIFY_SH = REPO_ROOT / "scripts" / "verify_final_candidate.sh"


class VerifyFinalCandidateCliTest(unittest.TestCase):
    def test_verify_cli_help_exits_successfully(self) -> None:
        proc = subprocess.run([sys.executable, str(VERIFY_PY), "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--expected-ate", proc.stdout)
        self.assertIn("--tol-path-ratio", proc.stdout)

    def test_verify_shell_script_contains_locked_metrics(self) -> None:
        text = VERIFY_SH.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("--expected-ate 7.352288", text)
        self.assertIn("--expected-drift 1.327343", text)
        self.assertIn("--expected-path-ratio 0.932379", text)


if __name__ == "__main__":
    unittest.main()
