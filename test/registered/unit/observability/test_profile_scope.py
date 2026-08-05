import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.observability.profile_scope import (
    batch_bucket,
    profile_scope,
    record_profile_step,
    start_profile_recording,
    stop_profile_recording,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestProfileScope(unittest.TestCase):
    def tearDown(self):
        stop_profile_recording()

    def test_batch_bucket(self):
        self.assertEqual(batch_bucket(0), "0")
        self.assertEqual(batch_bucket(1), "1")
        self.assertEqual(batch_bucket(2), "2")
        self.assertEqual(batch_bucket(9), "9-16")
        self.assertEqual(batch_bucket(16), "9-16")

    def test_disabled_step_recording_does_not_inspect_metadata(self):
        record_profile_step("disabled", tensor_like=object())

    @patch("sglang.srt.observability.profile_scope.profile_range")
    def test_records_stable_scope_and_host_metadata(self, mock_profile_range):
        mock_profile_range.return_value.__enter__.return_value = None
        with tempfile.TemporaryDirectory() as temporary_dir:
            start_profile_recording(
                output_dir=temporary_dir,
                profile_id="run",
                profile_prefix="",
                stage="decode",
                ps=ParallelState.trivial(),
            )
            record_profile_step("model_forward_start", batch_size=9)
            with profile_scope("model.forward", batch_bucket="9-16"):
                pass
            path = stop_profile_recording()

            records = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual(
                [record["event"] for record in records],
                ["clock_sync", "model_forward_start", "scope_start", "scope_end"],
            )
            self.assertEqual(records[2]["scope"], "model.forward")
            self.assertEqual(records[2]["batch_bucket"], "9-16")

    def test_rejects_tensor_like_metadata_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            start_profile_recording(
                output_dir=temporary_dir,
                profile_id="run",
                profile_prefix="",
                stage=None,
                ps=ParallelState.trivial(),
            )
            with self.assertRaises(TypeError):
                record_profile_step("bad", tensor=object())


if __name__ == "__main__":
    unittest.main()
