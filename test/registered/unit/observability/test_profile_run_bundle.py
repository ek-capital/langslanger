import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from sglang.benchmark.profile_run_bundle import (
    canonical_sha256,
    resolve_profile_id,
    summarize_speculative_outputs,
    workload_sha256,
    write_profile_run_bundle,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@dataclass
class _Output:
    success: bool = True
    spec_metrics_present: bool = True
    spec_accept_rate: float | None = None
    spec_accept_length: float | None = None
    spec_cap_length: float | None = None
    spec_block_accept_length: float | None = None
    spec_num_correct_drafts: int | None = None
    spec_num_proposed_drafts: int | None = None
    spec_verify_ct: int | None = None
    spec_correct_drafts_histogram: list[int] = field(default_factory=list)
    spec_cap_lens_histogram: list[int] = field(default_factory=list)


class TestProfileRunBundle(unittest.TestCase):
    def test_profile_id_is_safe_and_explicit(self):
        self.assertEqual(resolve_profile_id("speed-run.1"), "speed-run.1")
        self.assertTrue(resolve_profile_id(None).startswith("bench-"))
        with self.assertRaises(ValueError):
            resolve_profile_id("../escape")

    def test_workload_fingerprint_is_ordered(self):
        requests = [
            {"prompt": "a", "prompt_len": 1, "output_len": 2},
            {"prompt": "b", "prompt_len": 1, "output_len": 2},
        ]
        self.assertNotEqual(
            workload_sha256(requests), workload_sha256(reversed(requests))
        )

    def test_speculative_summary_keeps_distribution_and_weighted_rate(self):
        summary = summarize_speculative_outputs(
            [
                _Output(
                    spec_accept_rate=0.5,
                    spec_accept_length=2.0,
                    spec_num_correct_drafts=2,
                    spec_num_proposed_drafts=4,
                    spec_verify_ct=2,
                    spec_correct_drafts_histogram=[0, 2],
                    spec_cap_lens_histogram=[0, 0, 2],
                ),
                _Output(
                    spec_accept_rate=0.75,
                    spec_accept_length=3.0,
                    spec_num_correct_drafts=6,
                    spec_num_proposed_drafts=8,
                    spec_verify_ct=2,
                    spec_correct_drafts_histogram=[0, 1, 1],
                    spec_cap_lens_histogram=[0, 0, 1, 1],
                ),
                _Output(success=False),
            ]
        )
        self.assertTrue(summary["available"])
        self.assertEqual(summary["acceptance_rate"]["p50"], 0.625)
        self.assertEqual(summary["draft_totals"]["weighted_acceptance_rate"], 2 / 3)
        self.assertEqual(summary["correct_drafts_histogram"], [0, 3, 1])
        self.assertEqual(summary["cap_length_histogram"], [0, 0, 3, 1])

    def test_bundle_joins_profile_workload_and_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "run.profile.json"
            benchmark = {"duration": 1.5, "completed": 2}
            write_profile_run_bundle(
                output_path,
                profile_id="run-1",
                evidence_class="deployment-proxy",
                workload={"ordered_requests_sha256": "abc"},
                profile_targets=[{"worker": "combined", "output_dir": "/tmp/run-1"}],
                benchmark_record=benchmark,
                speculative={"available": False},
            )
            bundle = json.loads(output_path.read_text())
            self.assertEqual(bundle["profile_id"], "run-1")
            self.assertEqual(
                bundle["benchmark"]["record_sha256"], canonical_sha256(benchmark)
            )
            self.assertEqual(bundle["evidence_class"], "deployment-proxy")

    def test_bundle_rejects_unknown_evidence_class(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(ValueError):
                write_profile_run_bundle(
                    Path(temporary_dir) / "run.profile.json",
                    profile_id="run-1",
                    evidence_class="trust-me",
                    workload={},
                    profile_targets=[],
                    benchmark_record={},
                    speculative={"available": False},
                )


if __name__ == "__main__":
    unittest.main()
