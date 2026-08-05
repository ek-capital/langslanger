"""Stable semantic profiler scopes and structured per-forward sidecars."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType
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
        self._once_keys: set[str] = set()
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
            self._write_unlocked(record)

    def close(self) -> Path:
        with self._lock:
            if self._closed:
                return self.path
            self._closed = True
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
        return self.path

    def claim_once(self, key: str) -> bool:
        with self._lock:
            if self._closed or key in self._once_keys:
                return False
            self._once_keys.add(key)
            return True

    def _write_unlocked(self, record: dict[str, Any]) -> None:
        if self._closed:
            return
        self._record_id += 1
        record = {
            "record_id": self._record_id,
            "monotonic_time_ns": time.perf_counter_ns(),
            **record,
        }
        self._handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        self._handle.write("\n")


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


def record_profile_impl(
    scope: str,
    implementation: str,
    *,
    source_objects: tuple[Any, ...] = (),
    source_files: tuple[str | Path, ...] = (),
    expected_symbols: tuple[str, ...] = (),
    loaded_modules: tuple[str, ...] = (),
    conditions: dict[str, Any] | None = None,
) -> None:
    """Record an explicit dispatch decision once per distinct implementation.

    Source and library inspection happens only while profile recording is active.
    Callers should pass the callable/class actually selected by the dispatcher,
    plus explicit lower-level sources or symbols when the wrapper alone is not
    enough to identify compiled code.
    """
    recorder = _RECORDER
    if recorder is None:
        return

    declaration = {
        "scope": scope,
        "implementation": implementation,
        "source_objects": [_object_label(obj) for obj in source_objects],
        "source_files": [str(path) for path in source_files],
        "expected_symbols": sorted(set(expected_symbols)),
        "loaded_modules": sorted(set(loaded_modules)),
        "conditions": _jsonable_metadata(conditions or {}),
    }
    declaration_key = (
        "implementation:"
        + hashlib.sha256(
            json.dumps(declaration, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    if not recorder.claim_once(declaration_key):
        return

    sources = {
        record["path"]: record
        for record in (
            *(_provenance_for_object(obj) for obj in source_objects),
            *(_provenance_for_path(path) for path in source_files),
        )
        if record is not None
    }
    libraries = {}
    for module_prefix in loaded_modules:
        for record in _loaded_module_provenance(module_prefix):
            libraries[(record["module"], record.get("path"))] = record

    record = {
        "event": "implementation",
        "scope": scope,
        "implementation": implementation,
        "conditions": declaration["conditions"],
        "expected_symbols": sorted(set(expected_symbols)),
        "sources": [sources[path] for path in sorted(sources)],
        "loaded_libraries": [
            libraries[key] for key in sorted(libraries, key=lambda item: str(item))
        ],
        "attribution_source": "explicit_dispatch_declaration",
    }
    identity_payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
    record["implementation_id"] = hashlib.sha256(
        identity_payload.encode("utf-8")
    ).hexdigest()[:16]
    recorder.write(record)


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


def _provenance_for_object(obj: Any) -> dict[str, Any] | None:
    try:
        if isinstance(obj, ModuleType):
            path = getattr(obj, "__file__", None)
        else:
            path = inspect.getsourcefile(obj) or inspect.getfile(obj)
    except (TypeError, OSError):
        path = getattr(inspect.getmodule(obj), "__file__", None)
    return _provenance_for_path(path) if path else None


def _object_label(obj: Any) -> str:
    target = getattr(obj, "__func__", obj)
    return ":".join(
        value
        for value in (
            getattr(target, "__module__", None),
            getattr(target, "__qualname__", None),
            type(target).__qualname__,
        )
        if value
    )


@lru_cache(maxsize=None)
def _provenance_for_path(raw_path: str | Path) -> dict[str, Any] | None:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and (repo_root := _repo_root()) is not None:
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        return None
    repo_root = _repo_root()
    try:
        display_path = str(path.relative_to(repo_root)) if repo_root else str(path)
    except ValueError:
        display_path = str(path)
    content = path.read_bytes()
    return {
        "path": display_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "git_blob": _git_blob(path, repo_root),
        "kind": "library" if path.suffix in {".so", ".dylib", ".dll"} else "source",
    }


@lru_cache(maxsize=1)
def _repo_root() -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return Path(result.stdout.strip()).resolve() if result.returncode == 0 else None


def _git_blob(path: Path, repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        path.relative_to(repo_root)
        result = subprocess.run(
            ["git", "hash-object", str(path)],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


@lru_cache(maxsize=None)
def _loaded_module_provenance(prefix: str) -> list[dict[str, Any]]:
    records = []
    package_version = None
    distributions = importlib.metadata.packages_distributions().get(prefix, [])
    for distribution in distributions or [prefix.replace("_", "-")]:
        try:
            package_version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    for name, module in sorted(sys.modules.items()):
        if name != prefix and not name.startswith(f"{prefix}."):
            continue
        path = getattr(module, "__file__", None)
        if not path or Path(path).suffix not in {".so", ".dylib", ".dll"}:
            continue
        provenance = _provenance_for_path(path)
        if provenance is not None:
            records.append(
                {"module": name, "package_version": package_version, **provenance}
            )
    if not records:
        module = sys.modules.get(prefix)
        path = getattr(module, "__file__", None) if module else None
        provenance = _provenance_for_path(path) if path else None
        records.append(
            {
                "module": prefix,
                "package_version": package_version,
                **(provenance or {"path": None, "sha256": None, "git_blob": None}),
            }
        )
    return records
