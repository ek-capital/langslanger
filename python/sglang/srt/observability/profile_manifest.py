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
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sglang.version import __version__ as sglang_version

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState

PROFILE_MANIFEST_SCHEMA_VERSION = 2
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

    launch = _server_args(server_args)
    software = _software()
    hardware = _hardware(ps.gpu_id)
    source = _source_checkout()
    process = _process_identity(ps)
    profiler = _jsonable(profiler_options)
    parallel = _jsonable(dataclasses.asdict(ps))
    payload = {
        "schema_version": PROFILE_MANIFEST_SCHEMA_VERSION,
        "profile_id": profile_id,
        "rank_label": profile_rank_label(ps),
        "stage": stage,
        "started_at": _format_utc(started_at_ns),
        "stopped_at": _format_utc(stopped_at_ns),
        "duration_ns": max(0, stopped_at_ns - started_at_ns),
        "activities": activities,
        "profiler_options": profiler,
        "parallel": parallel,
        "process": process,
        "launch": launch,
        "software": software,
        "hardware": hardware,
        "runtime_environment": _runtime_environment(),
        "source": source,
        "artifacts": _artifacts(output_dir, artifact_paths),
    }
    payload["run_fingerprint"] = _run_fingerprint(
        profile_id=profile_id,
        stage=stage,
        activities=activities,
        profiler_options=profiler,
        parallel=parallel,
        process=process,
        launch=launch,
        software=software,
        hardware=hardware,
        source=source,
    )

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
        "nccl": _nccl_version(),
    }


def _hardware(gpu_id: int) -> dict[str, Any]:
    topology = _command_output(["nvidia-smi", "topo", "-m"])
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_id": gpu_id,
        "driver_version": _nvidia_driver_version(),
        "cpu_affinity": _cpu_affinity(),
        "numa_nodes": _read_text("/sys/devices/system/node/online"),
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "accelerator_inventory": _nvidia_accelerator_inventory(),
        "runtime_conditions": _nvidia_runtime_conditions(),
        "topology": {
            "nvidia_smi_topo_m": topology,
            "sha256": (
                hashlib.sha256(topology.encode("utf-8")).hexdigest()
                if topology
                else None
            ),
        },
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


def _process_identity(ps: ParallelState) -> dict[str, Any]:
    distributed = torch.distributed
    initialized = distributed.is_available() and distributed.is_initialized()
    trivial_world_size = 1 if _is_trivial_parallel_state(ps) else None
    world_size = (
        distributed.get_world_size()
        if initialized
        else _env_int("WORLD_SIZE", trivial_world_size)
    )
    global_rank = (
        distributed.get_rank()
        if initialized
        else _env_int("RANK", 0 if world_size == 1 else None)
    )
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "global_rank": global_rank,
        "local_rank": _env_int("LOCAL_RANK", ps.gpu_id),
        "node_rank": _env_int(
            "NODE_RANK", _env_int("GROUP_RANK", 0 if world_size == 1 else None)
        ),
        "world_size": world_size,
    }


def _is_trivial_parallel_state(ps: ParallelState) -> bool:
    return all(
        size == 1
        for size in (
            ps.tp_size,
            ps.pp_size,
            ps.dp_size,
            ps.attn_tp_size,
            ps.attn_cp_size,
            ps.attn_dp_size,
            ps.moe_ep_size,
            ps.moe_dp_size,
            ps.attn_dcp_size,
        )
    )


def _run_fingerprint(
    *,
    profile_id: str,
    stage: str | None,
    activities: list[str],
    profiler_options: dict[str, Any],
    parallel: dict[str, Any],
    process: dict[str, Any],
    launch: dict[str, Any],
    software: dict[str, Any],
    hardware: dict[str, Any],
    source: dict[str, Any],
) -> str:
    """Identify rank-independent deployment inputs for one profile run."""
    rank_independent_parallel = {
        key: value
        for key, value in parallel.items()
        if not key.endswith("_rank") and key != "gpu_id"
    }
    identity = {
        "profile_id": profile_id,
        "stage": stage,
        "activities": activities,
        "profiler_options": profiler_options,
        "parallel_sizes": rank_independent_parallel,
        "world_size": process.get("world_size"),
        "launch": {
            key: value
            for key, value in launch.items()
            if not key.endswith("_rank") and key not in {"base_gpu_id", "gpu_id"}
        },
        "software": software,
        "hardware_platform": hardware.get("platform"),
        "accelerator_inventory": [
            {
                key: value
                for key, value in accelerator.items()
                if key not in {"index", "uuid", "pci.bus_id"}
            }
            for accelerator in hardware.get("accelerator_inventory", [])
        ],
        "topology_sha256": (hardware.get("topology") or {}).get("sha256"),
        "source": source,
    }
    encoded = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime_environment() -> dict[str, str]:
    allowed_names = {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "ROCR_VISIBLE_DEVICES",
    }
    allowed_prefixes = ("NCCL_", "NVSHMEM_", "SGLANG_DEEPEP_")
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if (key in allowed_names or key.startswith(allowed_prefixes))
        and not _is_secret_field(key)
    }


def _nccl_version() -> str | None:
    try:
        version = torch.cuda.nccl.version()
    except (AssertionError, AttributeError, RuntimeError):
        return None
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    return str(version)


def _nvidia_accelerator_inventory() -> list[dict[str, str]]:
    fields = ("index", "uuid", "pci.bus_id", "name", "memory.total", "compute_cap")
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    return _parse_csv_rows(output, fields)


def _nvidia_runtime_conditions() -> list[dict[str, str]]:
    fields = ("index", "pstate", "power.limit", "clocks.current.sm")
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    return _parse_csv_rows(output, fields)


def _parse_csv_rows(output: str, fields: tuple[str, ...]) -> list[dict[str, str]]:
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        rows.append(dict(zip(fields, values, strict=True)))
    return rows


def _cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None


def _nvidia_driver_version() -> str | None:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    versions = sorted({line.strip() for line in output.splitlines() if line.strip()})
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
