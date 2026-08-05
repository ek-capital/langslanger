"""Offline, all-rank analysis for LangSlanger serving profiles.

The analyzer intentionally uses only the Python standard library so profile
artifacts can be inspected away from a configured SGLang runtime.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROFILE_REPORT_SCHEMA_VERSION = 3
_GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}


@dataclass
class _TraceEvent:
    name: str
    category: str
    start_us: float
    end_us: float
    pid: Any
    tid: Any
    external_id: str | None

    @property
    def duration_us(self) -> float:
        return max(0.0, self.end_us - self.start_us)


@dataclass
class _ScopeInvocation:
    invocation_id: int
    name: str
    event: _TraceEvent
    metadata: dict[str, Any]
    gpu_events: list[_TraceEvent] = field(default_factory=list)
    exclusive_gpu_events: list[_TraceEvent] = field(default_factory=list)
    aligned_gpu_intervals: list[tuple[float, float]] = field(default_factory=list)
    aligned_exclusive_gpu_intervals: list[tuple[float, float]] = field(
        default_factory=list
    )


@dataclass
class _TraceClockAlignment:
    anchors: list[tuple[float, float]]
    max_uncertainty_us: float

    def convert_us(self, trace_us: float) -> float:
        if len(self.anchors) == 1 or trace_us <= self.anchors[0][0]:
            trace_anchor, monotonic_anchor = self.anchors[0]
            return monotonic_anchor + trace_us - trace_anchor
        if trace_us >= self.anchors[-1][0]:
            trace_anchor, monotonic_anchor = self.anchors[-1]
            return monotonic_anchor + trace_us - trace_anchor
        for (trace_start, monotonic_start), (trace_end, monotonic_end) in zip(
            self.anchors, self.anchors[1:]
        ):
            if trace_start <= trace_us <= trace_end:
                ratio = (trace_us - trace_start) / (trace_end - trace_start)
                return monotonic_start + ratio * (monotonic_end - monotonic_start)
        raise AssertionError("trace timestamp did not fall between clock anchors")

    def interval(self, event: _TraceEvent) -> tuple[float, float]:
        return self.convert_us(event.start_us), self.convert_us(event.end_us)


def analyze_profile(profile_dir: str | Path, profile_id: str | None = None):
    """Analyze matching per-rank manifests and return a JSON-compatible report."""
    profile_dir = Path(profile_dir).expanduser().resolve()
    manifests = _load_manifests(profile_dir, profile_id)
    if not manifests:
        raise ValueError(f"no profile manifests found in {profile_dir}")

    resolved_ids = {manifest["profile_id"] for _, manifest in manifests}
    if len(resolved_ids) != 1:
        raise ValueError(
            "multiple profile IDs found; pass --profile-id: "
            + ", ".join(sorted(resolved_ids))
        )
    profile_id = next(iter(resolved_ids))

    warnings: list[str] = []
    rank_manifest_coverage = _validate_manifest_set(manifests)
    warnings.extend(rank_manifest_coverage.pop("warnings"))
    integrity = {"verified": 0, "failed": 0, "missing": 0}
    all_invocations: list[tuple[dict[str, Any], _ScopeInvocation]] = []
    scheduler_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    implementation_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    incomplete_rank_artifacts: list[str] = []
    total_gpu_events = 0
    attributed_gpu_events: set[tuple[str, int]] = set()
    total_gpu_work_us = 0.0
    unattributed_gpu_work_us = 0.0
    trace_clock_alignments: list[dict[str, Any]] = []

    for manifest_path, manifest in manifests:
        artifacts = _resolve_artifacts(profile_dir, manifest_path, manifest, integrity)
        trace_paths = [path for path in artifacts if ".trace.json" in path.name]
        step_paths = [path for path in artifacts if path.name.endswith(".steps.jsonl")]
        if not trace_paths:
            warnings.append(f"{manifest_path.name}: no Kineto trace artifact")
            incomplete_rank_artifacts.append(f"{manifest_path.name}:trace")
            continue
        if not step_paths:
            warnings.append(f"{manifest_path.name}: no step sidecar artifact")
            incomplete_rank_artifacts.append(f"{manifest_path.name}:steps")
            continue

        sidecar_records = _read_jsonl(step_paths[0])
        clock_sync_records = [
            record for record in sidecar_records if record.get("event") == "clock_sync"
        ]
        scope_metadata = defaultdict(list)
        for record in sidecar_records:
            if record.get("event") == "scope_start":
                scope_metadata[record["scope"]].append(record)
            elif record.get("event") == "scheduler_result":
                scheduler_results.append((manifest, record))
            elif record.get("event") == "implementation":
                implementation_records.append((manifest, record))

        for trace_path in trace_paths:
            trace_events = _read_trace(trace_path)
            gpu_events = [event for event in trace_events if _is_gpu_event(event)]
            cpu_events = [event for event in trace_events if not _is_gpu_event(event)]
            clock_alignment = _trace_clock_alignment(
                cpu_events, clock_sync_records, manifest, warnings
            )
            if clock_alignment is not None:
                trace_clock_alignments.append(
                    {
                        "rank_label": manifest.get("rank_label", "unknown"),
                        "stage": manifest.get("stage"),
                        "anchors": len(clock_alignment.anchors),
                        "max_uncertainty_us": clock_alignment.max_uncertainty_us,
                    }
                )
            total_gpu_events += len(gpu_events)
            total_gpu_work_us += sum(event.duration_us for event in gpu_events)

            invocations = _scope_invocations(cpu_events, scope_metadata, warnings)
            external_to_scopes = _external_id_scopes(cpu_events, invocations)
            for gpu_index, gpu_event in enumerate(gpu_events):
                matched = external_to_scopes.get(gpu_event.external_id, [])
                if not matched:
                    unattributed_gpu_work_us += gpu_event.duration_us
                    continue
                attributed_gpu_events.add((str(trace_path), gpu_index))
                for invocation in matched:
                    invocation.gpu_events.append(gpu_event)
                    if clock_alignment is not None:
                        invocation.aligned_gpu_intervals.append(
                            clock_alignment.interval(gpu_event)
                        )
                exclusive_invocation = min(
                    matched, key=lambda inv: inv.event.duration_us
                )
                exclusive_invocation.exclusive_gpu_events.append(gpu_event)
                if clock_alignment is not None:
                    exclusive_invocation.aligned_exclusive_gpu_intervals.append(
                        clock_alignment.interval(gpu_event)
                    )
            all_invocations.extend((manifest, invocation) for invocation in invocations)

    committed_by_replica_iteration = _deduplicate_scheduler_results(
        scheduler_results, warnings
    )
    implementation_ids = _implementation_ids_by_rank_scope(implementation_records)
    groups = _group_scope_metrics(
        all_invocations, committed_by_replica_iteration, implementation_ids
    )
    distributed_steps = _distributed_step_report(
        all_invocations,
        manifests,
        committed_by_replica_iteration,
        trace_clock_alignments,
        warnings,
    )
    communication = _communication_report(
        all_invocations,
        committed_by_replica_iteration,
        distributed_steps,
        trace_clock_alignments,
        warnings,
    )
    scopes = _aggregate_scope_metrics(groups, distributed_steps.get("steps", []))
    ranks = _aggregate_rank_metrics(groups)
    implementations = _implementation_report(implementation_records, groups)

    if integrity["failed"]:
        warnings.append(f"{integrity['failed']} artifacts failed checksum validation")
    if integrity["missing"]:
        warnings.append(f"{integrity['missing']} manifest artifacts are missing")
    if rank_manifest_coverage["strict"] and (
        integrity["failed"] or integrity["missing"] or incomplete_rank_artifacts
    ):
        raise ValueError(
            "profile artifact integrity failed: "
            f"failed={integrity['failed']} missing={integrity['missing']} "
            f"incomplete_ranks={incomplete_rank_artifacts}"
        )

    coverage = (
        len(attributed_gpu_events) / total_gpu_events if total_gpu_events else None
    )
    return {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_id": profile_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timing_basis": {
            "trace_unit": "microseconds",
            "gpu_elapsed": "union of attributed GPU event intervals on each rank",
            "summed_gpu_work": "sum of attributed GPU event durations; may overlap",
            "request_weighted_gpu_ms_per_committed_token": (
                "sum of per-invocation GPU interval-union milliseconds multiplied "
                "by host batch size, divided by committed tokens"
            ),
            "gpu_service_ms_per_committed_token": (
                "per-rank GPU interval-union milliseconds divided by committed tokens"
            ),
            "distributed_critical_path": (
                "span from earliest participating-rank GPU work to latest completion, "
                "using paired single-host monotonic clock markers"
                if distributed_steps["available"]
                else None
            ),
        },
        "coverage": {
            "manifests": len(manifests),
            "rank_manifests": rank_manifest_coverage,
            "clock_alignment": {
                "rank_traces": trace_clock_alignments,
                "aligned_rank_traces": len(trace_clock_alignments),
            },
            "artifact_integrity": integrity,
            "gpu_events": total_gpu_events,
            "attributed_gpu_events": len(attributed_gpu_events),
            "gpu_event_attribution_ratio": coverage,
            "gpu_work_ms": total_gpu_work_us / 1000.0,
            "unattributed_gpu_work_ms": unattributed_gpu_work_us / 1000.0,
        },
        "normalization": {
            "committed_tokens": sum(committed_by_replica_iteration.values()),
            "unique_replica_iterations": len(committed_by_replica_iteration),
            "replica_key": "dp_rank plus scheduler_iteration",
        },
        "scopes": scopes,
        "rank_bucket_scopes": groups,
        "ranks": ranks,
        "distributed_steps": distributed_steps,
        "communication": communication,
        "implementations": implementations,
        "warnings": sorted(set(warnings)),
    }


def write_profile_report(
    profile_dir: str | Path,
    *,
    profile_id: str | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    report = analyze_profile(profile_dir, profile_id)
    output_dir = Path(output_dir or profile_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_output_stem(report["profile_id"])
    json_path = output_dir / f"{stem}.report.json"
    markdown_path = output_dir / f"{stem}.report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown_report(report))
    return json_path, markdown_path


def interval_union_us(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the measure of a union of half-open intervals."""
    sorted_intervals = sorted((start, end) for start, end in intervals if end > start)
    if not sorted_intervals:
        return 0.0
    total = 0.0
    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def interval_intersection_us(
    left: Iterable[tuple[float, float]], right: Iterable[tuple[float, float]]
) -> float:
    """Return the intersection measure of two interval unions."""
    left_intervals = _merged_intervals(left)
    right_intervals = _merged_intervals(right)
    total = 0.0
    left_index = right_index = 0
    while left_index < len(left_intervals) and right_index < len(right_intervals):
        left_start, left_end = left_intervals[left_index]
        right_start, right_end = right_intervals[right_index]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _merged_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted((start, end) for start, end in intervals if end > start):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _load_manifests(profile_dir: Path, profile_id: str | None):
    manifests = []
    for path in sorted(profile_dir.glob("*.manifest.json")):
        manifest = json.loads(path.read_text())
        if profile_id is None or manifest.get("profile_id") == profile_id:
            manifests.append((path, manifest))
    return manifests


def _validate_manifest_set(manifests):
    """Fail closed when a v2 profile is missing or mixes participating ranks."""
    schema_versions = {manifest.get("schema_version", 1) for _, manifest in manifests}
    if schema_versions == {1}:
        return {
            "strict": False,
            "complete": None,
            "stages": [],
            "warnings": ["legacy v1 manifests do not prove all-rank completeness"],
        }
    if schema_versions != {2}:
        raise ValueError(
            "profile mixes manifest schema versions: "
            + ", ".join(str(version) for version in sorted(schema_versions))
        )

    by_stage = defaultdict(list)
    for path, manifest in manifests:
        by_stage[manifest.get("stage")].append((path, manifest))

    stage_coverage = []
    for stage, stage_manifests in sorted(
        by_stage.items(), key=lambda item: str(item[0])
    ):
        world_sizes = {
            manifest.get("process", {}).get("world_size")
            for _, manifest in stage_manifests
        }
        if len(world_sizes) != 1 or None in world_sizes:
            raise ValueError(
                f"stage {stage or 'all'} has inconsistent or missing world_size: "
                f"{sorted(str(size) for size in world_sizes)}"
            )
        world_size = next(iter(world_sizes))
        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError(
                f"stage {stage or 'all'} has invalid world_size={world_size}"
            )

        fingerprints = {
            manifest.get("run_fingerprint") for _, manifest in stage_manifests
        }
        if len(fingerprints) != 1 or None in fingerprints:
            raise ValueError(
                f"stage {stage or 'all'} has inconsistent run fingerprints"
            )

        ranks = [
            manifest.get("process", {}).get("global_rank")
            for _, manifest in stage_manifests
        ]
        non_integer_ranks = [rank for rank in ranks if not isinstance(rank, int)]
        if non_integer_ranks:
            raise ValueError(
                f"stage {stage or 'all'} has missing global ranks: {non_integer_ranks}"
            )
        observed = set(ranks)
        duplicate_ranks = sorted(rank for rank in observed if ranks.count(rank) > 1)
        expected = set(range(world_size))
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        labels = [manifest.get("rank_label") for _, manifest in stage_manifests]
        duplicate_labels = sorted(
            str(label) for label in set(labels) if labels.count(label) > 1
        )
        if missing or unexpected or duplicate_ranks or duplicate_labels:
            raise ValueError(
                f"stage {stage or 'all'} rank coverage failed: missing={missing} "
                f"unexpected={unexpected} duplicate_ranks={duplicate_ranks} "
                f"duplicate_labels={duplicate_labels}"
            )
        stage_coverage.append(
            {
                "stage": stage,
                "world_size": world_size,
                "observed_global_ranks": sorted(observed),
                "run_fingerprint": next(iter(fingerprints)),
                "complete": True,
            }
        )

    return {
        "strict": True,
        "complete": True,
        "stages": stage_coverage,
        "warnings": [],
    }


def _resolve_artifacts(
    profile_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    integrity: dict[str, int],
) -> list[Path]:
    paths = []
    for artifact in manifest.get("artifacts", []):
        path = (manifest_path.parent / artifact["path"]).resolve()
        try:
            path.relative_to(profile_dir)
        except ValueError as exc:
            raise ValueError(f"artifact escapes profile directory: {path}") from exc
        if not path.is_file():
            integrity["missing"] += 1
            continue
        expected = artifact.get("sha256")
        if expected and _sha256(path) != expected:
            integrity["failed"] += 1
            continue
        integrity["verified"] += 1
        paths.append(path)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_trace(path: Path) -> list[_TraceEvent]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_events = (
        payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    )
    events = []
    for raw in raw_events:
        if raw.get("ph") != "X" or "ts" not in raw or "dur" not in raw:
            continue
        start_us = float(raw["ts"])
        duration_us = max(0.0, float(raw["dur"]))
        events.append(
            _TraceEvent(
                name=str(raw.get("name", "")),
                category=str(raw.get("cat", "")),
                start_us=start_us,
                end_us=start_us + duration_us,
                pid=raw.get("pid"),
                tid=raw.get("tid"),
                external_id=_external_id(raw.get("args", {})),
            )
        )
    return events


def _trace_clock_alignment(cpu_events, clock_records, manifest, warnings):
    markers = sorted(
        (event for event in cpu_events if event.name == "profile.clock_sync"),
        key=lambda event: event.start_us,
    )
    records = [
        record
        for record in clock_records
        if record.get("monotonic_before_ns") is not None
        and record.get("monotonic_after_ns") is not None
    ]
    rank = manifest.get("rank_label", "unknown")
    if not markers or not records:
        warnings.append(f"{rank}: no paired trace/monotonic clock markers")
        return None
    if len(markers) != len(records):
        warnings.append(
            f"{rank}: clock marker count mismatch ({len(markers)} vs {len(records)})"
        )

    anchors = []
    uncertainties = []
    for marker, record in zip(markers, records):
        trace_midpoint_us = (marker.start_us + marker.end_us) / 2
        monotonic_midpoint_us = (
            float(record["monotonic_before_ns"]) + float(record["monotonic_after_ns"])
        ) / 2000
        anchors.append((trace_midpoint_us, monotonic_midpoint_us))
        uncertainties.append(
            float(record.get("uncertainty_ns", 0)) / 1000 + marker.duration_us
        )
    anchors.sort()
    unique_anchors = []
    for anchor in anchors:
        if unique_anchors and anchor[0] == unique_anchors[-1][0]:
            continue
        unique_anchors.append(anchor)
    if not unique_anchors:
        return None
    return _TraceClockAlignment(unique_anchors, max(uncertainties, default=0.0))


def _invocation_intervals(
    invocation: _ScopeInvocation, *, exclusive: bool
) -> list[tuple[float, float]]:
    aligned = (
        invocation.aligned_exclusive_gpu_intervals
        if exclusive
        else invocation.aligned_gpu_intervals
    )
    if aligned:
        return aligned
    events = invocation.exclusive_gpu_events if exclusive else invocation.gpu_events
    return [(event.start_us, event.end_us) for event in events]


def _distributed_step_report(
    invocations, manifests, committed, trace_clock_alignments, warnings
):
    hostnames = {
        manifest.get("process", {}).get("hostname")
        or manifest.get("hardware", {}).get("hostname")
        for _, manifest in manifests
    }
    hostnames.discard(None)
    if len(hostnames) != 1:
        return {
            "available": False,
            "reason": "distributed forward timing currently requires one host",
            "steps": [],
        }
    if len(trace_clock_alignments) < len(manifests):
        return {
            "available": False,
            "reason": "not every rank trace has paired clock markers",
            "steps": [],
        }
    if any(
        manifest.get("parallel", {}).get("attn_dp_size", 1) > 1
        or manifest.get("parallel", {}).get("pp_size", 1) > 1
        for _, manifest in manifests
    ):
        return {
            "available": False,
            "reason": (
                "attention-DP and pipeline layouts require an explicit shared "
                "forward epoch; use distributed collective rows meanwhile"
            ),
            "steps": [],
        }

    expected_by_replica = defaultdict(set)
    for _, manifest in manifests:
        parallel = manifest.get("parallel", {})
        replica = parallel.get("dp_rank", 0) or 0
        expected_by_replica[replica].add(manifest.get("rank_label", "unknown"))

    grouped = defaultdict(list)
    forward_scopes = {
        "runtime.prefill",
        "runtime.decode",
        "spec.verify",
        "spec.draft_extend",
    }
    for manifest, invocation in invocations:
        if invocation.name not in forward_scopes:
            continue
        metadata = invocation.metadata
        scheduler_iteration = metadata.get("scheduler_iteration")
        if scheduler_iteration is None or not invocation.aligned_gpu_intervals:
            continue
        parallel = manifest.get("parallel", {})
        replica = parallel.get("dp_rank", 0) or 0
        key = (
            invocation.name,
            manifest.get("stage"),
            replica,
            scheduler_iteration,
            metadata.get("worker"),
            metadata.get("forward_mode"),
            metadata.get("batch_bucket", "unknown"),
        )
        grouped[key].append((manifest, invocation))

    steps = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        scope, stage, replica, iteration, worker, forward_mode, bucket = key
        by_rank = defaultdict(list)
        for manifest, invocation in values:
            by_rank[manifest.get("rank_label", "unknown")].extend(
                invocation.aligned_gpu_intervals
            )
        expected = expected_by_replica[replica]
        if set(by_rank) != expected:
            warnings.append(
                f"{scope} iteration {iteration}: incomplete aligned participants "
                f"observed={sorted(by_rank)} expected={sorted(expected)}"
            )
            continue
        rank_rows = []
        for rank, intervals in sorted(by_rank.items()):
            rank_rows.append(
                {
                    "rank_label": rank,
                    "start_us": min(start for start, _ in intervals),
                    "end_us": max(end for _, end in intervals),
                    "gpu_elapsed_ms": interval_union_us(intervals) / 1000,
                }
            )
        step_start = min(row["start_us"] for row in rank_rows)
        step_end = max(row["end_us"] for row in rank_rows)
        critical_rank = max(rank_rows, key=lambda row: row["end_us"])
        for row in rank_rows:
            row["start_skew_ms"] = (row.pop("start_us") - step_start) / 1000
            row["completion_skew_ms"] = (step_end - row.pop("end_us")) / 1000
        representative_manifest = values[0][0]
        committed_tokens = committed.get(
            _replica_key(representative_manifest, iteration), 0
        )
        critical_path_ms = (step_end - step_start) / 1000
        steps.append(
            {
                "scope": scope,
                "stage": stage,
                "replica": replica,
                "scheduler_iteration": iteration,
                "worker": worker,
                "forward_mode": forward_mode,
                "batch_bucket": bucket,
                "committed_tokens": committed_tokens,
                "critical_path_ms": critical_path_ms,
                "critical_path_ms_per_committed_token": _divide(
                    critical_path_ms, committed_tokens
                ),
                "critical_rank": critical_rank["rank_label"],
                "rank_details": rank_rows,
            }
        )

    return {
        "available": bool(steps),
        "reason": None if steps else "no complete aligned distributed forward steps",
        "alignment": "single-host monotonic clock markers",
        "max_clock_marker_uncertainty_us": max(
            (record["max_uncertainty_us"] for record in trace_clock_alignments),
            default=None,
        ),
        "steps": steps,
    }


def _communication_report(
    invocations, committed, distributed_steps, trace_clock_alignments, warnings
):
    rank_comm = defaultdict(list)
    rank_compute = defaultdict(list)
    grouped_collectives = defaultdict(list)
    for manifest, invocation in invocations:
        rank = manifest.get("rank_label", "unknown")
        intervals = _invocation_intervals(invocation, exclusive=True)
        if invocation.name == "model.collective":
            rank_comm[rank].extend(intervals)
            metadata = invocation.metadata
            key = (
                manifest.get("stage"),
                metadata.get("collective_group"),
                metadata.get("collective_sequence"),
                metadata.get("operation", "unknown"),
                metadata.get("backend", "unknown"),
            )
            grouped_collectives[key].append((manifest, invocation))
        else:
            rank_compute[rank].extend(intervals)

    total_committed_tokens = sum(committed.values())
    distributed_critical_path_ms = sum(
        step["critical_path_ms"] for step in distributed_steps.get("steps", [])
    )
    rank_rows = []
    for rank in sorted(set(rank_comm) | set(rank_compute)):
        comm_intervals = rank_comm[rank]
        compute_intervals = rank_compute[rank]
        comm_us = interval_union_us(comm_intervals)
        compute_us = interval_union_us(compute_intervals)
        overlap_us = interval_intersection_us(comm_intervals, compute_intervals)
        exposed_ms = max(0.0, comm_us - overlap_us) / 1000
        rank_rows.append(
            {
                "rank_label": rank,
                "communication_elapsed_ms": comm_us / 1000,
                "compute_elapsed_ms": compute_us / 1000,
                "compute_communication_overlap_ms": overlap_us / 1000,
                "exposed_communication_ms": exposed_ms,
                "exposed_communication_ms_per_committed_token": _divide(
                    exposed_ms, total_committed_tokens
                ),
                "exposed_communication_pct_of_distributed_time": (
                    100 * exposed_ms / distributed_critical_path_ms
                    if distributed_critical_path_ms
                    else None
                ),
            }
        )

    global_rank_labels = {}
    for manifest, _ in invocations:
        global_rank = manifest.get("process", {}).get("global_rank")
        if isinstance(global_rank, int):
            global_rank_labels[global_rank] = manifest.get("rank_label", "unknown")
    single_host_aligned = (
        bool(trace_clock_alignments)
        and len(
            {
                manifest.get("process", {}).get("hostname")
                or manifest.get("hardware", {}).get("hostname")
                for manifest, _ in invocations
            }
        )
        == 1
    )

    collective_rows = []
    for key, values in sorted(
        grouped_collectives.items(), key=lambda item: str(item[0])
    ):
        stage, group, sequence, operation, backend = key
        by_rank = defaultdict(list)
        expected_global_ranks = set()
        input_bytes = []
        output_bytes = []
        relevant_replica_iterations = set()
        for manifest, invocation in values:
            metadata = invocation.metadata
            by_rank[manifest.get("rank_label", "unknown")].extend(
                invocation.aligned_exclusive_gpu_intervals
                or invocation.aligned_gpu_intervals
            )
            expected_global_ranks.update(metadata.get("group_ranks") or [])
            input_bytes.append(int(metadata.get("input_bytes") or 0))
            output_bytes.append(int(metadata.get("output_bytes") or 0))
            if metadata.get("scheduler_iteration") is not None:
                relevant_replica_iterations.add(
                    _replica_key(manifest, metadata["scheduler_iteration"])
                )
        expected_labels = {
            global_rank_labels[rank]
            for rank in expected_global_ranks
            if rank in global_rank_labels
        }
        complete = bool(expected_labels) and set(by_rank) == expected_labels
        aligned = single_host_aligned and all(by_rank.values())
        rank_ends = {
            rank: max(end for _, end in intervals)
            for rank, intervals in by_rank.items()
            if intervals
        }
        row = {
            "stage": stage,
            "collective_group": group,
            "collective_sequence": sequence,
            "operation": operation,
            "backend": backend,
            "expected_rank_labels": sorted(expected_labels),
            "observed_rank_labels": sorted(by_rank),
            "complete": complete,
            "input_bytes_per_rank_max": max(input_bytes, default=0),
            "output_bytes_per_rank_max": max(output_bytes, default=0),
            "committed_tokens": sum(
                committed.get(replica_iteration, 0)
                for replica_iteration in relevant_replica_iterations
            ),
            "critical_path_ms": None,
            "critical_path_ms_per_committed_token": None,
            "critical_rank": None,
        }
        if complete and aligned and rank_ends:
            starts = [start for intervals in by_rank.values() for start, _ in intervals]
            ends = [end for intervals in by_rank.values() for _, end in intervals]
            row["critical_path_ms"] = (max(ends) - min(starts)) / 1000
            row["critical_path_ms_per_committed_token"] = _divide(
                row["critical_path_ms"], row["committed_tokens"]
            )
            row["critical_rank"] = max(rank_ends, key=rank_ends.get)
        elif not complete:
            warnings.append(
                f"collective {group}:{sequence} has incomplete rank coverage"
            )
        collective_rows.append(row)

    operation_rows = []
    by_operation = defaultdict(list)
    for row in collective_rows:
        if row["critical_path_ms"] is not None:
            by_operation[row["operation"]].append(row)
    total_collective_critical_ms = sum(
        row["critical_path_ms"] for rows in by_operation.values() for row in rows
    )
    for operation, rows in sorted(by_operation.items()):
        critical_path_ms = sum(row["critical_path_ms"] for row in rows)
        operation_rows.append(
            {
                "operation": operation,
                "collectives": len(rows),
                "critical_path_ms": critical_path_ms,
                "share_of_collective_critical_path_pct": (
                    100 * critical_path_ms / total_collective_critical_ms
                    if total_collective_critical_ms
                    else None
                ),
            }
        )

    return {
        "timing_basis": (
            "single-host monotonic clock aligned"
            if single_host_aligned
            else "per-rank trace intervals only"
        ),
        "ranks": rank_rows,
        "operations": operation_rows,
        "collectives": collective_rows,
    }


def _external_id(args: dict[str, Any]) -> str | None:
    for key, value in args.items():
        normalized = key.lower().replace(" ", "").replace("_", "")
        if normalized == "externalid" and value is not None:
            return str(value)
    return None


def _is_gpu_event(event: _TraceEvent) -> bool:
    category = event.category.lower()
    return category in _GPU_CATEGORIES or category.startswith("gpu_")


def _scope_invocations(cpu_events, scope_metadata, warnings):
    invocations = []
    invocation_id = 0
    for scope_name, metadata_records in scope_metadata.items():
        trace_scopes = sorted(
            (event for event in cpu_events if event.name == scope_name),
            key=lambda event: (event.start_us, event.end_us),
        )
        if len(trace_scopes) != len(metadata_records):
            warnings.append(
                f"scope {scope_name}: trace/sidecar invocation count mismatch "
                f"({len(trace_scopes)} vs {len(metadata_records)})"
            )
        for event, metadata in zip(trace_scopes, metadata_records):
            invocation_id += 1
            invocations.append(
                _ScopeInvocation(invocation_id, scope_name, event, metadata)
            )
    return invocations


def _external_id_scopes(cpu_events, invocations):
    by_thread = defaultdict(list)
    for invocation in invocations:
        key = (invocation.event.pid, invocation.event.tid)
        by_thread[key].append(invocation)

    result = defaultdict(list)
    for event in cpu_events:
        if event.external_id is None:
            continue
        for invocation in by_thread[(event.pid, event.tid)]:
            if (
                invocation.event.start_us <= event.start_us
                and event.end_us <= invocation.event.end_us
            ):
                result[event.external_id].append(invocation)
    return result


def _replica_key(manifest: dict[str, Any], scheduler_iteration: Any):
    parallel = manifest.get("parallel", {})
    return (parallel.get("dp_rank", 0), scheduler_iteration)


def _deduplicate_scheduler_results(results, warnings):
    committed = {}
    for manifest, record in results:
        key = _replica_key(manifest, record.get("scheduler_iteration"))
        value = int(record.get("committed_tokens", 0))
        if key in committed and committed[key] != value:
            warnings.append(
                f"committed-token mismatch for replica iteration {key}: "
                f"{committed[key]} vs {value}"
            )
            continue
        committed[key] = value
    return committed


def _group_scope_metrics(invocations, committed, implementation_ids):
    grouped = defaultdict(list)
    for manifest, invocation in invocations:
        metadata = invocation.metadata
        key = (
            invocation.name,
            manifest.get("rank_label", "unknown"),
            metadata.get("batch_bucket", "unknown"),
        )
        grouped[key].append((manifest, invocation))

    rows = []
    for (scope, rank_label, bucket), values in sorted(grouped.items()):
        all_events = [event for _, inv in values for event in inv.gpu_events]
        exclusive_events = [
            event for _, inv in values for event in inv.exclusive_gpu_events
        ]
        elapsed_us = interval_union_us(
            (event.start_us, event.end_us) for event in all_events
        )
        summed_us = sum(event.duration_us for event in all_events)
        exclusive_elapsed_us = interval_union_us(
            (event.start_us, event.end_us) for event in exclusive_events
        )
        exclusive_summed_us = sum(event.duration_us for event in exclusive_events)
        invocation_elapsed_us = [
            interval_union_us(
                (event.start_us, event.end_us) for event in inv.gpu_events
            )
            for _, inv in values
        ]
        relevant_iterations = {
            _replica_key(manifest, inv.metadata.get("scheduler_iteration"))
            for manifest, inv in values
            if inv.metadata.get("scheduler_iteration") is not None
        }
        committed_tokens = sum(committed.get(key, 0) for key in relevant_iterations)
        request_weighted_us = sum(
            elapsed * int(inv.metadata.get("batch_size") or 0)
            for elapsed, (_, inv) in zip(invocation_elapsed_us, values)
        )
        symbol_work = defaultdict(float)
        for event in all_events:
            symbol_work[event.name] += event.duration_us
        rows.append(
            {
                "scope": scope,
                "rank_label": rank_label,
                "batch_bucket": bucket,
                "invocations": len(values),
                "committed_tokens": committed_tokens,
                "gpu_elapsed_ms": elapsed_us / 1000.0,
                "exclusive_gpu_elapsed_ms": exclusive_elapsed_us / 1000.0,
                "summed_gpu_work_ms": summed_us / 1000.0,
                "exclusive_summed_gpu_work_ms": exclusive_summed_us / 1000.0,
                "overlap_ms": max(0.0, summed_us - elapsed_us) / 1000.0,
                "gpu_service_ms_per_committed_token": _divide(
                    elapsed_us / 1000.0, committed_tokens
                ),
                "request_weighted_gpu_ms_per_committed_token": _divide(
                    request_weighted_us / 1000.0, committed_tokens
                ),
                "implementation_ids": implementation_ids.get((rank_label, scope), []),
                "top_gpu_symbols": [
                    {"symbol": symbol, "summed_gpu_work_ms": duration / 1000.0}
                    for symbol, duration in sorted(
                        symbol_work.items(), key=lambda item: item[1], reverse=True
                    )[:10]
                ],
            }
        )
    return rows


def _implementation_ids_by_rank_scope(records):
    result = defaultdict(set)
    for manifest, record in records:
        result[(manifest.get("rank_label", "unknown"), record.get("scope"))].add(
            record["implementation_id"]
        )
    return {key: sorted(value) for key, value in result.items()}


def _implementation_report(records, groups):
    grouped_records = defaultdict(list)
    for manifest, record in records:
        grouped_records[record["implementation_id"]].append((manifest, record))

    symbols_by_rank_scope = defaultdict(dict)
    for group in groups:
        key = (group["rank_label"], group["scope"])
        for item in group["top_gpu_symbols"]:
            symbols_by_rank_scope[key][item["symbol"]] = (
                symbols_by_rank_scope[key].get(item["symbol"], 0.0)
                + item["summed_gpu_work_ms"]
            )

    implementations = []
    for implementation_id, values in sorted(grouped_records.items()):
        first = dict(values[0][1])
        first.pop("event", None)
        first.pop("record_id", None)
        first.pop("monotonic_time_ns", None)
        observed = defaultdict(float)
        observed_on_ranks = set()
        for manifest, record in values:
            rank = manifest.get("rank_label", "unknown")
            observed_on_ranks.add(rank)
            for symbol, duration in symbols_by_rank_scope[
                (rank, record.get("scope"))
            ].items():
                observed[symbol] += duration
        expected = [value.lower() for value in first.get("expected_symbols", [])]
        first["observed_on_ranks"] = sorted(observed_on_ranks)
        first["observed_matching_symbols"] = [
            symbol
            for symbol in sorted(observed)
            if any(fragment in symbol.lower() for fragment in expected)
        ]
        first["top_observed_gpu_symbols"] = [
            {"symbol": symbol, "summed_gpu_work_ms": duration}
            for symbol, duration in sorted(
                observed.items(), key=lambda item: item[1], reverse=True
            )[:10]
        ]
        implementations.append(first)
    return implementations


def _aggregate_scope_metrics(groups, distributed_steps):
    by_scope = defaultdict(list)
    for group in groups:
        by_scope[group["scope"]].append(group)
    distributed_by_scope = defaultdict(list)
    for step in distributed_steps:
        distributed_by_scope[step["scope"]].append(step)
    rows = []
    for scope, scope_groups in sorted(by_scope.items()):
        slowest = max(scope_groups, key=lambda row: row["gpu_elapsed_ms"])
        step_rows = distributed_by_scope[scope]
        critical_path_ms = (
            sum(step["critical_path_ms"] for step in step_rows) if step_rows else None
        )
        critical_path_tokens = sum(step["committed_tokens"] for step in step_rows)
        rows.append(
            {
                "scope": scope,
                "rank_bucket_groups": len(scope_groups),
                "sum_rank_gpu_elapsed_ms": sum(
                    row["gpu_elapsed_ms"] for row in scope_groups
                ),
                "max_rank_bucket_gpu_elapsed_ms": slowest["gpu_elapsed_ms"],
                "slowest_observed_rank": slowest["rank_label"],
                "slowest_observed_batch_bucket": slowest["batch_bucket"],
                "distributed_critical_path_ms": critical_path_ms,
                "distributed_critical_path_ms_per_committed_token": (
                    _divide(critical_path_ms, critical_path_tokens)
                    if critical_path_ms is not None
                    else None
                ),
            }
        )
    return rows


def _aggregate_rank_metrics(groups):
    by_rank = defaultdict(list)
    for group in groups:
        by_rank[group["rank_label"]].append(group)
    rows = []
    for rank, rank_groups in sorted(by_rank.items()):
        slowest = max(rank_groups, key=lambda row: row["gpu_elapsed_ms"])
        rows.append(
            {
                "rank_label": rank,
                "max_scope_bucket_gpu_elapsed_ms": slowest["gpu_elapsed_ms"],
                "slowest_observed_scope": slowest["scope"],
                "slowest_observed_batch_bucket": slowest["batch_bucket"],
                "distributed_critical_path_ms": None,
            }
        )
    return rows


def _divide(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _markdown_report(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    ratio = coverage["gpu_event_attribution_ratio"]
    ratio_text = "n/a" if ratio is None else f"{ratio:.1%}"
    distributed = report["distributed_steps"]
    distributed_text = (
        "Distributed step spans use paired single-host monotonic clock markers."
        if distributed["available"]
        else "Distributed critical path unavailable: " + distributed["reason"] + "."
    )
    lines = [
        f"# Profile report: {report['profile_id']}",
        "",
        "Timing basis: GPU elapsed values are interval unions per rank. Summed GPU "
        "work may overlap and is never presented as wall-clock latency. "
        + distributed_text,
        "",
        f"GPU event attribution coverage: {ratio_text}. Committed tokens: "
        f"{report['normalization']['committed_tokens']}.",
        "",
        "| Scope | Rank | Batch bucket | GPU elapsed ms | Exclusive GPU ms | Summed work ms | "
        "Service ms/committed token | Request-weighted ms/committed token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rank_bucket_scopes"]:
        lines.append(
            "| {scope} | {rank} | {bucket} | {elapsed:.3f} | {exclusive:.3f} | {work:.3f} | "
            "{service} | {weighted} |".format(
                scope=row["scope"],
                rank=row["rank_label"],
                bucket=row["batch_bucket"],
                elapsed=row["gpu_elapsed_ms"],
                exclusive=row["exclusive_gpu_elapsed_ms"],
                work=row["summed_gpu_work_ms"],
                service=_format_optional(row["gpu_service_ms_per_committed_token"]),
                weighted=_format_optional(
                    row["request_weighted_gpu_ms_per_committed_token"]
                ),
            )
        )
    if distributed["available"]:
        lines.extend(
            [
                "",
                "## Distributed steps",
                "",
                "| Scope | Iteration | Batch bucket | Critical path ms | ms/committed token | Critical rank |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for step in distributed["steps"]:
            lines.append(
                "| {scope} | {iteration} | {bucket} | {critical:.4f} | {per_token} | {rank} |".format(
                    scope=step["scope"],
                    iteration=step["scheduler_iteration"],
                    bucket=step["batch_bucket"],
                    critical=step["critical_path_ms"],
                    per_token=_format_optional(
                        step["critical_path_ms_per_committed_token"]
                    ),
                    rank=step["critical_rank"],
                )
            )
    communication_rows = report["communication"]["ranks"]
    if communication_rows:
        lines.extend(
            [
                "",
                "## Communication exposure",
                "",
                "| Rank | Communication ms | Compute ms | Overlap ms | Exposed communication ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in communication_rows:
            lines.append(
                "| {rank} | {communication:.4f} | {compute:.4f} | {overlap:.4f} | {exposed:.4f} |".format(
                    rank=row["rank_label"],
                    communication=row["communication_elapsed_ms"],
                    compute=row["compute_elapsed_ms"],
                    overlap=row["compute_communication_overlap_ms"],
                    exposed=row["exposed_communication_ms"],
                )
            )
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.4f}"


def _safe_output_stem(profile_id: str) -> str:
    stem = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in profile_id
    ).strip(".")
    if not stem:
        raise ValueError("profile ID has no safe output filename characters")
    return stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", help="Directory containing rank manifests")
    parser.add_argument(
        "--profile-id", help="Profile ID when the directory has several"
    )
    parser.add_argument(
        "--output-dir", help="Report destination; defaults to profile_dir"
    )
    args = parser.parse_args(argv)
    json_path, markdown_path = write_profile_report(
        args.profile_dir, profile_id=args.profile_id, output_dir=args.output_dir
    )
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
