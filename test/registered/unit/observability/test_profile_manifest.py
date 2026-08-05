import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.observability.profile_manifest import (
    profile_rank_label,
    write_profile_manifest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclass
class _Args:
    model_path: str = "model"
    api_key: str = "do-not-write"
    max_total_tokens: int = 1024
    extra: Path = Path("config.json")


class TestProfileManifest(unittest.TestCase):
    def test_rank_label_only_includes_enabled_parallel_dimensions(self):
        ps = ParallelState.trivial(
            tp_rank=1,
            tp_size=2,
            dp_rank=3,
            dp_size=4,
            pp_rank=0,
            pp_size=1,
            moe_ep_rank=2,
            moe_ep_size=8,
        )
        self.assertEqual(profile_rank_label(ps), "TP-1-DP-3-EP-2")

    @patch(
        "sglang.srt.observability.profile_manifest._source_checkout",
        return_value={"git_commit": "abc", "dirty": False},
    )
    @patch(
        "sglang.srt.observability.profile_manifest._hardware",
        return_value={"hostname": "test-host", "gpu_id": 0},
    )
    def test_writes_atomic_manifest_with_artifact_hashes(self, _hardware, _source):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            trace_path = output_dir / "run-TP-0.trace.json.gz"
            trace_path.write_bytes(b"trace")

            manifest_path = write_profile_manifest(
                output_dir=output_dir,
                profile_id="run",
                profile_prefix="",
                stage=None,
                ps=ParallelState.trivial(),
                activities=["CPU", "GPU"],
                profiler_options={"record_shapes": False},
                server_args=_Args(),
                started_at_ns=1_000_000_000,
                stopped_at_ns=2_000_000_000,
                artifact_paths=[trace_path],
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["profile_id"], "run")
            self.assertEqual(manifest["duration_ns"], 1_000_000_000)
            self.assertEqual(manifest["launch"]["api_key"], "<redacted>")
            self.assertEqual(manifest["launch"]["max_total_tokens"], 1024)
            self.assertEqual(manifest["launch"]["extra"], "config.json")
            self.assertEqual(
                manifest["artifacts"],
                [
                    {
                        "path": trace_path.name,
                        "sha256": hashlib.sha256(b"trace").hexdigest(),
                        "size_bytes": 5,
                    }
                ],
            )
            self.assertEqual(list(output_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
