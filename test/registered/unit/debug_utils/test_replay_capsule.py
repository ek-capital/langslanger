import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.debug_utils.replay_capsule import (
    ReplayCaptureConfig,
    begin_replay_capture,
    configure_replay_capture,
)
from sglang.srt.debug_utils.replay_runner import run_replay
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _candidate(inputs, read_only_state, mutable_state, metadata):
    mutable_state["accumulator"].add_(inputs["x"])
    outputs = {"y": inputs["x"] * read_only_state["scale"] + metadata["bias"]}
    return outputs, mutable_state


class TestReplayCapsule(unittest.TestCase):
    def tearDown(self):
        configure_replay_capture(None)

    def test_capture_is_explicit_bounded_and_replayable(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = ReplayCaptureConfig(
                output_dir=Path(temporary_dir),
                scopes=frozenset({"model.test"}),
                max_cases_per_scope_rank=1,
            )
            configure_replay_capture(config)
            value = torch.tensor([1.0, 2.0])
            scale = torch.tensor([2.0, 3.0])
            accumulator = torch.tensor([0.0, 0.0])
            capture = begin_replay_capture(
                "model.test",
                inputs={"x": value},
                read_only_state={"scale": scale},
                mutable_state={"accumulator": accumulator},
                metadata={"bias": 1.0},
                required_topology={"world_size": 1},
            )
            self.assertTrue(capture.capturing)
            output, mutated_state = _candidate(
                {"x": value},
                {"scale": scale},
                {"accumulator": accumulator},
                {"bias": 1.0},
            )
            capsule = capture.finish(outputs=output, mutable_state=mutated_state)

            self.assertTrue((capsule / "COMPLETE").is_file())
            manifest = json.loads((capsule / "manifest.json").read_text())
            self.assertFalse(manifest["authoritative_serving_timing"])
            self.assertEqual(
                manifest["capture_warning"],
                "perturbed_run_not_performance_evidence",
            )

            second = begin_replay_capture(
                "model.test",
                inputs={"x": value},
                required_topology={"world_size": 1},
            )
            self.assertFalse(second.capturing)
            self.assertEqual(second.skip_reason, "case_limit_reached")

            result = run_replay(
                capsule,
                candidate=_candidate,
                device="cpu",
                warmup=1,
                iterations=3,
            )
            self.assertTrue(result["correctness"]["passed"])
            self.assertTrue(result["correctness"]["mutable_state"]["passed"])
            self.assertFalse(result["authoritative_serving_timing"])
            self.assertEqual(len(result["candidate_timing"]["samples_ms"]), 3)

            paired = run_replay(
                capsule,
                candidate=_candidate,
                reference=_candidate,
                device="cpu",
                warmup=0,
                iterations=2,
            )
            self.assertEqual(paired["timing_method"]["pair_order"], "alternating")
            self.assertEqual(len(paired["reference_timing"]["samples_ms"]), 2)

    def test_capture_rejects_unselected_scope_without_cloning(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            configure_replay_capture(
                ReplayCaptureConfig(
                    output_dir=Path(temporary_dir),
                    scopes=frozenset({"selected"}),
                )
            )
            capture = begin_replay_capture(
                "not-selected",
                inputs={"not even inspected": object()},
                required_topology={"world_size": 1},
            )
            self.assertFalse(capture.capturing)
            self.assertEqual(capture.skip_reason, "scope_not_selected")


if __name__ == "__main__":
    unittest.main()
