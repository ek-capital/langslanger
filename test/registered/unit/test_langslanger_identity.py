import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import langslanger
import sglang
from sglang.cli.main import main as sglang_main
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLangSlangerIdentity(CustomTestCase):
    def test_public_api_is_an_additive_alias(self):
        self.assertEqual(langslanger.__all__, sglang.__all__)
        self.assertEqual(langslanger.__version__, sglang.__version__)
        for name in sglang.__all__:
            self.assertIs(getattr(langslanger, name), getattr(sglang, name))

    def test_console_scripts_preserve_sglang_and_add_langslanger(self):
        repo_root = Path(__file__).resolve().parents[3]
        packaging_dir = repo_root / "python"
        for pyproject_path in packaging_dir.glob("pyproject*.toml"):
            with self.subTest(pyproject=pyproject_path.name):
                pyproject = pyproject_path.read_text()
                self.assertIn('sglang = "sglang.cli.main:main"', pyproject)
                self.assertIn('langslanger = "langslanger.cli:main"', pyproject)

        default_pyproject = (packaging_dir / "pyproject.toml").read_text()
        self.assertIn('killall_sglang = "sglang.cli.killall:main"', default_pyproject)
        self.assertIn(
            'killall_langslanger = "sglang.cli.killall:main"',
            default_pyproject,
        )

    @patch("sglang.cli.main.get_git_commit_hash", return_value="abcdef123456")
    def test_version_brand_follows_the_invoked_command(self, _git_hash):
        outputs = {}
        for program_name in ("sglang", "langslanger"):
            stdout = io.StringIO()
            with patch.object(sys, "argv", [program_name, "version"]):
                with contextlib.redirect_stdout(stdout):
                    sglang_main()
            outputs[program_name] = stdout.getvalue()

        self.assertIn("sglang version:", outputs["sglang"])
        self.assertNotIn("sglang compatibility: enabled", outputs["sglang"])
        self.assertIn("langslanger version:", outputs["langslanger"])
        self.assertIn("sglang compatibility: enabled", outputs["langslanger"])


if __name__ == "__main__":
    unittest.main()
