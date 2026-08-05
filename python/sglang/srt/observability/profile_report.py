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

PROFILE_REPORT_SCHEMA_VERSION = 2
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
                min(
                    matched, key=lambda inv: inv.event.duration_us
                ).exclusive_gpu_events.append(gpu_event)
            all_invocations.extend((manifest, invocation) for invocation in invocations)

    committed_by_replica_iteration = _deduplicate_scheduler_results(
        scheduler_results, warnings
    )
    implementation_ids = _implementation_ids_by_rank_scope(implementation_records)
    groups = _group_scope_metrics(
        all_invocations, committed_by_replica_iteration, implementation_ids
    )
    scopes = _aggregate_scope_metrics(groups)
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
            "distributed_critical_path": None,
        },
        "coverage": {
            "manifests": len(manifests),
            "rank_manifests": rank_manifest_coverage,
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


def _aggregate_scope_metrics(groups):
    by_scope = defaultdict(list)
    for group in groups:
        by_scope[group["scope"]].append(group)
    rows = []
    for scope, scope_groups in sorted(by_scope.items()):
        slowest = max(scope_groups, key=lambda row: row["gpu_elapsed_ms"])
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
                "distributed_critical_path_ms": None,
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
    lines = [
        f"# Profile report: {report['profile_id']}",
        "",
        "Timing basis: GPU elapsed values are interval unions per rank. Summed GPU "
        "work may overlap and is never presented as wall-clock latency. This report "
        "does not infer a distributed critical path.",
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
