"""Self-describing manifests for serving profiler artifacts.

The profiler trace remains the timing authority.  This module only records the
configuration and provenance needed to interpret that trace later.  Manifests
are per-rank so a distributed run does not depend on shared storage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.version import __version__ as sglang_version

PROFILE_MANIFEST_SCHEMA_VERSION = 1
_SECRET_SUFFIXES = (
    "api_key",
    "password",
    "secret",
    "access_token",
    "auth_token",
    "preshared_key",
)


def profile_rank_label(ps: ParallelState) -> str:
    """Return the same stable rank label used by profiler trace filenames."""
    parts = [f"TP-{ps.tp_rank}"]
    if ps.dp_size > 1:
        parts.append(f"DP-{ps.dp_rank}")
    if ps.pp_size > 1:
        parts.append(f"PP-{ps.pp_rank}")
    if ps.moe_ep_size > 1:
        parts.append(f"EP-{ps.moe_ep_rank}")
    return "-".join(parts)


def write_profile_manifest(
    *,
    output_dir: str | Path,
    profile_id: str,
    profile_prefix: str,
    stage: str | None,
    ps: ParallelState,
    activities: list[str],
    profiler_options: dict[str, Any],
    server_args: Any,
    started_at_ns: int,
    stopped_at_ns: int,
    artifact_paths: Iterable[str | Path],
) -> Path:
    """Atomically write a per-rank manifest and return its path."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{profile_prefix}-" if profile_prefix else ""
    suffix = f"-{stage}" if stage else ""
    filename = f"{prefix}{profile_id}-{profile_rank_label(ps)}{suffix}.manifest.json"
    manifest_path = output_dir / filename

    payload = {
        "schema_version": PROFILE_MANIFEST_SCHEMA_VERSION,
        "profile_id": profile_id,
        "rank_label": profile_rank_label(ps),
        "stage": stage,
        "started_at": _format_utc(started_at_ns),
        "stopped_at": _format_utc(stopped_at_ns),
        "duration_ns": max(0, stopped_at_ns - started_at_ns),
        "activities": activities,
        "profiler_options": _jsonable(profiler_options),
        "parallel": _jsonable(dataclasses.asdict(ps)),
        "launch": _server_args(server_args),
        "software": _software(),
        "hardware": _hardware(ps.gpu_id),
        "source": _source_checkout(),
        "artifacts": _artifacts(output_dir, artifact_paths),
    }

    fd, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=f".{filename}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, manifest_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return manifest_path


def _format_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat()


def _server_args(server_args: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(server_args) and not isinstance(server_args, type):
        values = {
            field.name: getattr(server_args, field.name)
            for field in dataclasses.fields(server_args)
        }
    else:
        values = vars(server_args) if hasattr(server_args, "__dict__") else {}
    return {
        key: "<redacted>" if _is_secret_field(key) else _jsonable(value)
        for key, value in sorted(values.items())
        if not key.startswith("_")
    }


def _is_secret_field(name: str) -> bool:
    normalized = name.lower()
    return any(
        normalized == suffix or normalized.endswith(f"_{suffix}")
        for suffix in _SECRET_SUFFIXES
    )


def _software() -> dict[str, Any]:
    return {
        "sglang": sglang_version,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "hip_runtime": getattr(torch.version, "hip", None),
    }


def _hardware(gpu_id: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_id": gpu_id,
        "driver_version": _nvidia_driver_version(),
    }
    if not torch.cuda.is_available():
        return result
    try:
        properties = torch.cuda.get_device_properties(gpu_id)
        result["accelerator"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "multi_processor_count": properties.multi_processor_count,
        }
    except (AssertionError, RuntimeError):
        # A manifest must never make an otherwise successful profile fail just
        # because a platform cannot expose optional device properties.
        result["accelerator"] = None
    return result


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    versions = sorted(
        {line.strip() for line in result.stdout.splitlines() if line.strip()}
    )
    return ",".join(versions) or None


def _source_checkout() -> dict[str, Any]:
    repo_root = _git(["rev-parse", "--show-toplevel"])
    if repo_root is None:
        return {"git_commit": None, "dirty": None, "dirty_diff_sha256": None}

    commit = _git(["-C", repo_root, "rev-parse", "HEAD"])
    status = _git(["-C", repo_root, "status", "--porcelain=v1", "--untracked-files=no"])
    diff = _git_bytes(["-C", repo_root, "diff", "--no-ext-diff", "--binary", "HEAD"])
    return {
        "git_commit": commit,
        "dirty": bool(status) if status is not None else None,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def _git(args: list[str]) -> str | None:
    output = _git_bytes(args)
    if output is None:
        return None
    return output.decode("utf-8", errors="replace").strip()


def _git_bytes(args: list[str]) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _artifacts(output_dir: Path, artifact_paths: Iterable[str | Path]):
    artifacts = []
    for raw_path in artifact_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = output_dir / path
        if not path.is_file():
            continue
        artifacts.append(
            {
                "path": os.path.relpath(path, output_dir),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return sorted(artifacts, key=lambda item: item["path"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)
