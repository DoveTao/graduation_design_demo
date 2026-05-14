from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HEALTH_SH = REPO_ROOT / "scripts" / "project_health_check.sh"


class ProjectHealthCheckScriptStaticTest(unittest.TestCase):
    def test_health_check_script_exists(self) -> None:
        self.assertTrue(HEALTH_SH.is_file())

    def test_health_check_script_contains_expected_commands(self) -> None:
        text = HEALTH_SH.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("bash scripts/verify_final_candidate.sh", text)
        self.assertIn("-m unittest discover -s tests -q", text)

    def test_health_check_script_does_not_require_pytest(self) -> None:
        text = HEALTH_SH.read_text(encoding="utf-8")
        self.assertNotIn("pytest -q\n", text)
        self.assertIn("pytest is available, optional command: pytest -q", text)


if __name__ == "__main__":
    unittest.main()
