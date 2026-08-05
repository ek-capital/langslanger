"""Small, stable evidence bundle helpers for profiled serving benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROFILE_RUN_BUNDLE_SCHEMA_VERSION = 1
PROFILE_EVIDENCE_CLASSES = {
    "diagnostic",
    "deployment-proxy",
    "production-equivalent",
}
_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def resolve_profile_id(value: str | None) -> str:
    """Return a filename-safe run identifier that is explicit on the wire."""
    if value is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        value = f"bench-{timestamp}-{uuid.uuid4().hex[:8]}"
    if not _PROFILE_ID.fullmatch(value):
        raise ValueError(
            "profile ID must start with an alphanumeric character and contain "
            "only alphanumerics, '.', '_', or '-' (maximum 128 characters)"
        )
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def workload_sha256(requests: Iterable[Any]) -> str:
    """Fingerprint the exact ordered requests without storing their contents."""
    digest = hashlib.sha256()
    for request in requests:
        if isinstance(request, dict):
            # Trace datasets such as Mooncake retain their source-specific keys
            # until request generation, so hash the full source row.
            record = request
        else:
            record = {
                "prompt": _read(request, "prompt"),
                "prompt_len": _read(request, "prompt_len"),
                "output_len": _read(request, "output_len"),
                "image_data": _read(request, "image_data"),
                "extra_request_body": _read(request, "extra_request_body", {}),
                "timestamp": _read(request, "timestamp"),
                "routing_key": _read(request, "routing_key"),
            }
        digest.update(
            json.dumps(
                record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_speculative_outputs(outputs: Iterable[Any]) -> dict[str, Any]:
    """Summarize per-request speculative metrics without hiding distributions."""
    samples = []
    correct_total = 0
    proposed_total = 0
    verify_total = 0
    correct_histogram: list[int] = []
    cap_histogram: list[int] = []

    for request_index, output in enumerate(outputs):
        if not _read(output, "success", False) or not _read(
            output, "spec_metrics_present", False
        ):
            continue
        correct = _optional_int(_read(output, "spec_num_correct_drafts"))
        proposed = _optional_int(_read(output, "spec_num_proposed_drafts"))
        verify = _optional_int(_read(output, "spec_verify_ct"))
        if correct is not None:
            correct_total += correct
        if proposed is not None:
            proposed_total += proposed
        if verify is not None:
            verify_total += verify
        _sum_histogram(
            correct_histogram, _read(output, "spec_correct_drafts_histogram", [])
        )
        _sum_histogram(cap_histogram, _read(output, "spec_cap_lens_histogram", []))
        samples.append(
            {
                "request_index": request_index,
                "acceptance_rate": _optional_float(_read(output, "spec_accept_rate")),
                "accept_length": _optional_float(_read(output, "spec_accept_length")),
                "cap_length": _optional_float(_read(output, "spec_cap_length")),
                "block_accept_length": _optional_float(
                    _read(output, "spec_block_accept_length")
                ),
                "correct_drafts": correct,
                "proposed_drafts": proposed,
                "verify_steps": verify,
            }
        )

    if not samples:
        return {
            "available": False,
            "reason": "no successful response carried speculative metrics",
            "requests_with_metrics": 0,
            "request_samples": [],
        }

    return {
        "available": True,
        "requests_with_metrics": len(samples),
        "acceptance_rate": _distribution(samples, "acceptance_rate"),
        "accept_length": _distribution(samples, "accept_length"),
        "cap_length": _distribution(samples, "cap_length"),
        "block_accept_length": _distribution(samples, "block_accept_length"),
        "draft_totals": {
            "correct": correct_total,
            "proposed": proposed_total,
            "verify_steps": verify_total,
            "weighted_acceptance_rate": (
                correct_total / proposed_total if proposed_total else None
            ),
        },
        "correct_drafts_histogram": correct_histogram,
        "cap_length_histogram": cap_histogram,
        "request_samples": samples,
    }


def write_profile_run_bundle(
    output_path: str | Path,
    *,
    profile_id: str,
    evidence_class: str,
    workload: dict[str, Any],
    profile_targets: list[dict[str, Any]],
    benchmark_record: dict[str, Any],
    speculative: dict[str, Any],
) -> Path:
    """Write one join record for workload, benchmark, and profile artifacts."""
    if evidence_class not in PROFILE_EVIDENCE_CLASSES:
        raise ValueError(f"unknown profile evidence class: {evidence_class}")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": PROFILE_RUN_BUNDLE_SCHEMA_VERSION,
        "profile_id": resolve_profile_id(profile_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": evidence_class,
        "workload": workload,
        "profile": {
            "targets": profile_targets,
            "report_artifact_names": [
                f"{profile_id}.report.json",
                f"{profile_id}.report.md",
            ],
        },
        "benchmark": {
            "record_sha256": canonical_sha256(benchmark_record),
            "record": benchmark_record,
        },
        "speculative_decoding": speculative,
    }
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    return output_path


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _sum_histogram(target: list[int], values: Iterable[Any]) -> None:
    values = list(values or [])
    if len(target) < len(values):
        target.extend([0] * (len(values) - len(target)))
    for index, value in enumerate(values):
        target[index] += int(value)


def _distribution(samples: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = sorted(
        value for sample in samples if (value := sample.get(key)) is not None
    )
    if not values:
        return None
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": values[0],
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": values[-1],
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
