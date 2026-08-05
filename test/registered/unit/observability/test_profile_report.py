import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.observability.profile_report import (
    analyze_profile,
    interval_intersection_us,
    interval_union_us,
    write_profile_report,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestProfileReport(unittest.TestCase):
    def test_interval_union(self):
        self.assertEqual(interval_union_us([(0, 5), (3, 7), (10, 12)]), 9)
        self.assertEqual(interval_union_us([]), 0)

    def test_interval_intersection(self):
        self.assertEqual(interval_intersection_us([(0, 5), (8, 12)], [(3, 10)]), 4)

    def test_analyzes_attributed_overlapping_gpu_events(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_dir = Path(temporary_dir)
            trace_path = profile_dir / "run-TP-0.trace.json.gz"
            steps_path = profile_dir / "run-TP-0.steps.jsonl"
            manifest_path = profile_dir / "run-TP-0.manifest.json"

            trace = {
                "traceEvents": [
                    _event("spec.verify", "user_annotation", 0, 20, 1),
                    _event("aten::op", "cpu_op", 1, 4, 7),
                    _event("kernel_a", "kernel", 5, 5, 7, pid=2, tid=3),
                    _event("kernel_b", "kernel", 8, 5, 7, pid=2, tid=4),
                    _event("kernel_unattributed", "kernel", 15, 2, 99, pid=2, tid=3),
                ]
            }
            with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
                json.dump(trace, handle)
            steps = [
                {
                    "event": "scope_start",
                    "scope": "spec.verify",
                    "batch_size": 9,
                    "batch_bucket": "9-16",
                    "scheduler_iteration": 3,
                },
                {"event": "scope_end", "scope": "spec.verify"},
                {
                    "event": "scheduler_result",
                    "scheduler_iteration": 3,
                    "committed_tokens": 18,
                },
                {
                    "event": "implementation",
                    "implementation_id": "impl-1",
                    "scope": "spec.verify",
                    "implementation": "synthetic.verify",
                    "conditions": {"operation": "verify"},
                    "expected_symbols": ["kernel_"],
                    "sources": [
                        {
                            "path": "kernel.cu",
                            "sha256": "sha256",
                            "git_blob": "blob",
                        }
                    ],
                    "loaded_libraries": [],
                    "attribution_source": "explicit_dispatch_declaration",
                },
            ]
            steps_path.write_text(
                "".join(json.dumps(record) + "\n" for record in steps)
            )
            manifest = {
                "schema_version": 1,
                "profile_id": "run",
                "rank_label": "TP-0",
                "parallel": {"tp_rank": 0, "dp_rank": 0},
                "artifacts": [_artifact(trace_path), _artifact(steps_path)],
            }
            manifest_path.write_text(json.dumps(manifest))

            report = analyze_profile(profile_dir)
            row = report["rank_bucket_scopes"][0]
            self.assertEqual(row["scope"], "spec.verify")
            self.assertEqual(row["batch_bucket"], "9-16")
            self.assertEqual(row["committed_tokens"], 18)
            self.assertEqual(row["gpu_elapsed_ms"], 0.008)
            self.assertEqual(row["summed_gpu_work_ms"], 0.010)
            self.assertEqual(row["overlap_ms"], 0.002)
            self.assertAlmostEqual(
                row["request_weighted_gpu_ms_per_committed_token"], 0.004
            )
            self.assertAlmostEqual(
                report["coverage"]["gpu_event_attribution_ratio"], 2 / 3
            )
            self.assertIsNone(report["scopes"][0]["distributed_critical_path_ms"])
            self.assertEqual(row["implementation_ids"], ["impl-1"])
            self.assertEqual(
                report["implementations"][0]["observed_matching_symbols"],
                ["kernel_a", "kernel_b"],
            )

            json_path, markdown_path = write_profile_report(profile_dir)
            self.assertTrue(json_path.is_file())
            self.assertIn("interval unions", markdown_path.read_text())

    def test_deduplicates_tokens_and_reports_rank_skew(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_dir = Path(temporary_dir)
            for rank, kernel_duration in ((0, 3), (1, 7)):
                trace_path = profile_dir / f"run-TP-{rank}.trace.json.gz"
                steps_path = profile_dir / f"run-TP-{rank}.steps.jsonl"
                manifest_path = profile_dir / f"run-TP-{rank}.manifest.json"
                trace = {
                    "traceEvents": [
                        _event("spec.verify", "user_annotation", 0, 20, rank + 1),
                        _event("aten::op", "cpu_op", 1, 2, 10 + rank),
                        _event(
                            f"kernel_{rank}",
                            "kernel",
                            5,
                            kernel_duration,
                            10 + rank,
                            pid=2,
                            tid=3,
                        ),
                    ]
                }
                with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
                    json.dump(trace, handle)
                records = [
                    {
                        "event": "scope_start",
                        "scope": "spec.verify",
                        "batch_size": 2,
                        "batch_bucket": "2",
                        "scheduler_iteration": 8,
                    },
                    {
                        "event": "scheduler_result",
                        "scheduler_iteration": 8,
                        "committed_tokens": 5,
                    },
                ]
                steps_path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "profile_id": "run",
                            "rank_label": f"TP-{rank}",
                            "run_fingerprint": "same-run",
                            "process": {
                                "global_rank": rank,
                                "world_size": 2,
                            },
                            "parallel": {"tp_rank": rank, "dp_rank": 0},
                            "artifacts": [_artifact(trace_path), _artifact(steps_path)],
                        }
                    )
                )

            report = analyze_profile(profile_dir)
            self.assertEqual(report["normalization"]["committed_tokens"], 5)
            self.assertTrue(report["coverage"]["rank_manifests"]["complete"])
            self.assertEqual(report["scopes"][0]["slowest_observed_rank"], "TP-1")
            self.assertIsNone(report["ranks"][1]["distributed_critical_path_ms"])

    def test_rejects_missing_v2_rank_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_dir = Path(temporary_dir)
            (profile_dir / "run-TP-0.manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "profile_id": "run",
                        "rank_label": "TP-0",
                        "run_fingerprint": "same-run",
                        "process": {"global_rank": 0, "world_size": 2},
                        "artifacts": [],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "missing=\\[1\\]"):
                analyze_profile(profile_dir)

    def test_rejects_v2_rank_without_profile_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_dir = Path(temporary_dir)
            (profile_dir / "run-TP-0.manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "profile_id": "run",
                        "rank_label": "TP-0",
                        "run_fingerprint": "same-run",
                        "process": {"global_rank": 0, "world_size": 1},
                        "artifacts": [],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "incomplete_ranks"):
                analyze_profile(profile_dir)

    def test_reports_aligned_distributed_steps_and_collectives(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_dir = Path(temporary_dir)
            for rank, collective_duration in ((0, 3), (1, 7)):
                trace_path = profile_dir / f"run-TP-{rank}.trace.json.gz"
                steps_path = profile_dir / f"run-TP-{rank}.steps.jsonl"
                manifest_path = profile_dir / f"run-TP-{rank}.manifest.json"
                trace = {
                    "traceEvents": [
                        _event("profile.clock_sync", "user_annotation", 0, 0.2, 1),
                        _event("runtime.decode", "user_annotation", 10, 20, 2),
                        _event("model.collective", "user_annotation", 14, 10, 3),
                        _event("aten::compute", "cpu_op", 11, 2, 10),
                        _event("aten::collective", "cpu_op", 14, 2, 20),
                        _event("compute_kernel", "kernel", 12, 4, 10, pid=2, tid=3),
                        _event(
                            "nccl_kernel",
                            "kernel",
                            15,
                            collective_duration,
                            20,
                            pid=2,
                            tid=4,
                        ),
                    ]
                }
                with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
                    json.dump(trace, handle)
                records = [
                    {
                        "event": "clock_sync",
                        "monotonic_before_ns": 999_900,
                        "monotonic_after_ns": 1_000_100,
                        "uncertainty_ns": 200,
                    },
                    {
                        "event": "scope_start",
                        "scope": "runtime.decode",
                        "scheduler_iteration": 8,
                        "worker": "target",
                        "forward_mode": "decode",
                        "batch_bucket": "2",
                        "batch_size": 2,
                    },
                    {
                        "event": "scope_start",
                        "scope": "model.collective",
                        "collective_group": "tp",
                        "collective_sequence": 1,
                        "collective_id": "tp:1",
                        "operation": "all_reduce",
                        "backend": "group_coordinator",
                        "scheduler_iteration": 8,
                        "group_ranks": [0, 1],
                        "input_bytes": 64,
                        "output_bytes": 0,
                    },
                    {
                        "event": "scheduler_result",
                        "scheduler_iteration": 8,
                        "committed_tokens": 4,
                    },
                ]
                steps_path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "profile_id": "run",
                            "rank_label": f"TP-{rank}",
                            "run_fingerprint": "same-run",
                            "stage": "decode",
                            "process": {
                                "global_rank": rank,
                                "world_size": 2,
                                "hostname": "one-host",
                            },
                            "parallel": {
                                "tp_rank": rank,
                                "tp_size": 2,
                                "dp_rank": 0,
                                "dp_size": 1,
                            },
                            "artifacts": [_artifact(trace_path), _artifact(steps_path)],
                        }
                    )
                )

            report = analyze_profile(profile_dir)
            self.assertTrue(report["distributed_steps"]["available"])
            step = report["distributed_steps"]["steps"][0]
            self.assertEqual(step["critical_rank"], "TP-1")
            self.assertAlmostEqual(step["critical_path_ms"], 0.010)
            collective = report["communication"]["collectives"][0]
            self.assertTrue(collective["complete"])
            self.assertEqual(collective["critical_rank"], "TP-1")
            self.assertAlmostEqual(collective["critical_path_ms"], 0.007)
            self.assertAlmostEqual(
                collective["critical_path_ms_per_committed_token"], 0.00175
            )
            rank_zero = report["communication"]["ranks"][0]
            self.assertAlmostEqual(rank_zero["compute_communication_overlap_ms"], 0.001)


def _event(name, category, start, duration, external_id, pid=1, tid=1):
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": start,
        "dur": duration,
        "pid": pid,
        "tid": tid,
        "args": {"External id": external_id},
    }


def _artifact(path: Path):
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    unittest.main()
