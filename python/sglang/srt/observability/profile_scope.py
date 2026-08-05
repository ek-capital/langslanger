"""Stable semantic profiler scopes and structured per-forward sidecars."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.observability.profile_manifest import profile_rank_label
from sglang.srt.utils.nvtx_utils import profile_range

PROFILE_STEP_SCHEMA_VERSION = 1
_LOCK = threading.Lock()
_RECORDER: _ProfileStepRecorder | None = None


class _ProfileStepRecorder:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        profile_id: str,
        profile_prefix: str,
        stage: str | None,
        ps: ParallelState,
    ) -> None:
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{profile_prefix}-" if profile_prefix else ""
        suffix = f"-{stage}" if stage else ""
        self.path = output_dir / (
            f"{prefix}{profile_id}-{profile_rank_label(ps)}{suffix}.steps.jsonl"
        )
        self._handle = self.path.open("w", encoding="utf-8", buffering=64 * 1024)
        self._record_id = 0
        self._lock = threading.Lock()
        self._closed = False
        self.write(
            {
                "event": "clock_sync",
                "schema_version": PROFILE_STEP_SCHEMA_VERSION,
                "profile_id": profile_id,
                "rank_label": profile_rank_label(ps),
                "stage": stage,
                "wall_time_ns": time.time_ns(),
                "monotonic_time_ns": time.perf_counter_ns(),
            }
        )

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            self._record_id += 1
            record = {
                "record_id": self._record_id,
                "monotonic_time_ns": time.perf_counter_ns(),
                **record,
            }
            self._handle.write(
                json.dumps(record, separators=(",", ":"), sort_keys=True)
            )
            self._handle.write("\n")

    def close(self) -> Path:
        with self._lock:
            if self._closed:
                return self.path
            self._closed = True
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
        return self.path


def start_profile_recording(
    *,
    output_dir: str | Path,
    profile_id: str,
    profile_prefix: str,
    stage: str | None,
    ps: ParallelState,
) -> None:
    """Start the sidecar paired with the active profiler on this rank."""
    global _RECORDER
    with _LOCK:
        if _RECORDER is not None:
            raise RuntimeError("profile step recording is already active")
        _RECORDER = _ProfileStepRecorder(
            output_dir=output_dir,
            profile_id=profile_id,
            profile_prefix=profile_prefix,
            stage=stage,
            ps=ps,
        )


def stop_profile_recording() -> Path | None:
    """Flush the active sidecar, returning its path when recording was active."""
    global _RECORDER
    with _LOCK:
        recorder, _RECORDER = _RECORDER, None
        return recorder.close() if recorder is not None else None


def record_profile_step(event: str, **metadata: Any) -> None:
    """Append host metadata without inspecting tensors or synchronizing a device."""
    recorder = _RECORDER
    if recorder is None:
        return
    recorder.write({"event": event, **_jsonable_metadata(metadata)})


def profile_scope(name: str, **metadata: Any):
    """Emit a stable trace range plus timestamped per-invocation metadata."""
    recorder = _RECORDER
    if recorder is None:
        return profile_range(name)
    return _profile_scope_recorded(recorder, name, metadata)


@contextmanager
def _profile_scope_recorded(
    recorder: _ProfileStepRecorder, name: str, metadata: dict[str, Any]
) -> Iterator[None]:
    recorder.write(
        {"event": "scope_start", "scope": name, **_jsonable_metadata(metadata)}
    )
    try:
        with profile_range(name):
            yield
    finally:
        recorder.write(
            {"event": "scope_end", "scope": name, **_jsonable_metadata(metadata)}
        )


def profile_forward_scope(forward_mode: Any) -> str:
    """Map detailed forward modes to stable, architecture-level phase names."""
    if forward_mode.is_target_verify():
        return "spec.verify"
    if forward_mode.is_draft_extend_v2():
        return "spec.draft_extend"
    if forward_mode.is_prefill():
        return "runtime.prefill"
    if forward_mode.is_decode():
        return "runtime.decode"
    return "runtime.other"


def batch_bucket(batch_size: int) -> str:
    """Power-of-two batch bucket with human-readable inclusive bounds."""
    if batch_size <= 0:
        return "0"
    upper = 1 << (batch_size - 1).bit_length()
    lower = 1 if upper == 1 else (upper // 2) + 1
    return str(upper) if lower == upper else f"{lower}-{upper}"


def _jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            result[key] = value
        elif isinstance(value, (list, tuple)) and all(
            item is None or isinstance(item, (bool, int, float, str)) for item in value
        ):
            result[key] = list(value)
        else:
            raise TypeError(
                f"profile metadata {key!r} must already be host-resident JSON data"
            )
    return result
