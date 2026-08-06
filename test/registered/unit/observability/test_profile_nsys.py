import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.observability.profile_nsys import (
    build_nsys_command,
    export_nsys_sqlite,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestProfileNsys(unittest.TestCase):
    def test_builds_node_level_api_triggered_command(self):
        command = build_nsys_command("run", ["python", "-m", "sglang.launch_server"])
        self.assertIn("--capture-range=cudaProfilerApi", command)
        self.assertIn("--cuda-graph-trace=node", command)
        self.assertIn("--trace-fork-before-exec=true", command)
        self.assertEqual(command[-3:], ["python", "-m", "sglang.launch_server"])

    @patch("sglang.srt.observability.profile_nsys.subprocess.run")
    @patch(
        "sglang.srt.observability.profile_nsys.shutil.which", return_value="/bin/nsys"
    )
    def test_exports_report_to_sqlite(self, _which, run):
        with tempfile.TemporaryDirectory() as temporary_dir:
            report = Path(temporary_dir) / "run.nsys-rep"
            report.touch()
            output = export_nsys_sqlite(report)
        self.assertEqual(output.name, "run.sqlite")
        run.assert_called_once()
        self.assertIn("--type=sqlite", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
