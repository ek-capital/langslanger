"""Explicitly armed, bounded tensor capture for offline kernel replay.

Capture is disabled unless both ``SGLANG_REPLAY_CAPTURE_DIR`` and
``SGLANG_REPLAY_CAPTURE_SCOPES`` are set. ``SGLANG_REPLAY_CAPTURE_RANKS``
defaults to rank 0. Captures synchronize and perturb execution, so their
artifacts are never valid serving-performance evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file

REPLAY_CAPSULE_SCHEMA_VERSION = 1
_DEFAULT_MAX_BYTES = 1024 * 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_UNSET = object()
_CONFIG: ReplayCaptureConfig | None | object = _UNSET
_LOCK = threading.Lock()
_CASES_BY_SCOPE_RANK: dict[tuple[str, int], int] = {}
_TOTAL_CASES = 0


@dataclass(frozen=True)
class ReplayCaptureConfig:
    output_dir: Path
    scopes: frozenset[str]
    ranks: frozenset[int] = frozenset({0})
    max_cases_per_scope_rank: int = 1
    max_total_cases: int = 8
    max_bytes_per_case: int = _DEFAULT_MAX_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        object.__setattr__(self, "scopes", frozenset(self.scopes))
        object.__setattr__(self, "ranks", frozenset(self.ranks))
        if not self.scopes:
            raise ValueError("replay capture requires at least one exact scope")
        if not self.ranks:
            raise ValueError("replay capture requires at least one rank")
        for name, value in (
            ("max_cases_per_scope_rank", self.max_cases_per_scope_rank),
            ("max_total_cases", self.max_total_cases),
            ("max_bytes_per_case", self.max_bytes_per_case),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_env(cls) -> ReplayCaptureConfig | None:
        raw_dir = os.environ.get("SGLANG_REPLAY_CAPTURE_DIR")
        raw_scopes = os.environ.get("SGLANG_REPLAY_CAPTURE_SCOPES")
        if not raw_dir and not raw_scopes:
            return None
        if not raw_dir or not raw_scopes:
            raise ValueError(
                "replay capture requires both SGLANG_REPLAY_CAPTURE_DIR and "
                "SGLANG_REPLAY_CAPTURE_SCOPES"
            )
        scopes = frozenset(_comma_values(raw_scopes))
        ranks = frozenset(
            int(value)
            for value in _comma_values(
                os.environ.get("SGLANG_REPLAY_CAPTURE_RANKS", "0")
            )
        )
        return cls(
            output_dir=Path(raw_dir).expanduser(),
            scopes=scopes,
            ranks=ranks,
            max_cases_per_scope_rank=int(
                os.environ.get("SGLANG_REPLAY_CAPTURE_MAX_CASES_PER_SCOPE_RANK", "1")
            ),
            max_total_cases=int(
                os.environ.get("SGLANG_REPLAY_CAPTURE_MAX_TOTAL_CASES", "8")
            ),
            max_bytes_per_case=int(
                os.environ.get(
                    "SGLANG_REPLAY_CAPTURE_MAX_BYTES", str(_DEFAULT_MAX_BYTES)
                )
            ),
        )


class InactiveReplayCapture:
    capturing = False

    def __init__(self, reason: str) -> None:
        self.skip_reason = reason

    def finish(
        self,
        *,
        outputs: Mapping[str, torch.Tensor],
        mutable_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        return None


class ReplayCapture:
    capturing = True
    skip_reason = None

    def __init__(
        self,
        *,
        config: ReplayCaptureConfig,
        scope: str,
        rank: int,
        case_index: int,
        inputs: Mapping[str, torch.Tensor],
        read_only_state: Mapping[str, torch.Tensor],
        mutable_state: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any],
        required_topology: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.scope = scope
        self.rank = rank
        self.case_index = case_index
        self.metadata = _json_mapping(metadata)
        self.required_topology = _json_mapping(required_topology)
        self._inputs, input_metadata = _clone_tensors(inputs)
        self._read_only_state, read_only_metadata = _clone_tensors(read_only_state)
        self._mutable_state, mutable_metadata = _clone_tensors(mutable_state)
        self._tensor_metadata = {
            **{f"input.{name}": value for name, value in input_metadata.items()},
            **{
                f"read_only.{name}": value for name, value in read_only_metadata.items()
            },
            **{
                f"mutable_before.{name}": value
                for name, value in mutable_metadata.items()
            },
        }
        self._initial_bytes = _tensor_bytes(
            (
                *self._inputs.values(),
                *self._read_only_state.values(),
                *self._mutable_state.values(),
            )
        )
        self._finished = False

    def finish(
        self,
        *,
        outputs: Mapping[str, torch.Tensor],
        mutable_state: Mapping[str, torch.Tensor] | None = None,
    ) -> Path | None:
        if self._finished:
            raise RuntimeError("replay capture has already been finished")
        self._finished = True
        _validate_tensors(outputs)
        mutable_state = mutable_state or {}
        _validate_tensors(mutable_state)
        if set(mutable_state) != set(self._mutable_state):
            raise ValueError(
                "mutable state names after execution must match mutable state names "
                "declared before execution"
            )
        total_bytes = self._initial_bytes + _tensor_bytes(
            (*outputs.values(), *mutable_state.values())
        )
        if total_bytes > self.config.max_bytes_per_case:
            self.skip_reason = "byte_limit_exceeded"
            self._release()
            return None

        captured_outputs, output_metadata = _clone_tensors(outputs)
        captured_mutable_state, mutable_after_metadata = _clone_tensors(mutable_state)
        tensors = {
            **{f"input.{name}": value for name, value in self._inputs.items()},
            **{
                f"read_only.{name}": value
                for name, value in self._read_only_state.items()
            },
            **{
                f"mutable_before.{name}": value
                for name, value in self._mutable_state.items()
            },
            **{f"output.{name}": value for name, value in captured_outputs.items()},
            **{
                f"mutable_after.{name}": value
                for name, value in captured_mutable_state.items()
            },
        }
        tensor_metadata = {
            **self._tensor_metadata,
            **{f"output.{name}": value for name, value in output_metadata.items()},
            **{
                f"mutable_after.{name}": value
                for name, value in mutable_after_metadata.items()
            },
        }
        cpu_tensors = {
            name: value.to(device="cpu").contiguous() for name, value in tensors.items()
        }
        try:
            return _write_capsule(
                config=self.config,
                scope=self.scope,
                rank=self.rank,
                case_index=self.case_index,
                metadata=self.metadata,
                required_topology=self.required_topology,
                tensors=cpu_tensors,
                tensor_metadata=tensor_metadata,
                total_bytes=total_bytes,
            )
        finally:
            self._release()

    def _release(self) -> None:
        self._inputs = {}
        self._read_only_state = {}
        self._mutable_state = {}


def configure_replay_capture(config: ReplayCaptureConfig | None) -> None:
    """Set capture configuration explicitly and reset this process's case budget."""
    global _CONFIG, _TOTAL_CASES
    with _LOCK:
        _CONFIG = config
        _CASES_BY_SCOPE_RANK.clear()
        _TOTAL_CASES = 0


def begin_replay_capture(
    scope: str,
    *,
    inputs: Mapping[str, torch.Tensor],
    read_only_state: Mapping[str, torch.Tensor] | None = None,
    mutable_state: Mapping[str, torch.Tensor] | None = None,
    metadata: Mapping[str, Any] | None = None,
    required_topology: Mapping[str, Any],
) -> ReplayCapture | InactiveReplayCapture:
    """Snapshot declared inputs/state on their producer stream when armed.

    Call ``finish(outputs=...)`` immediately after the target operation. The
    input clone is enqueued before that operation, preserving pre-mutation data.
    """
    config = _get_config()
    if config is None:
        return InactiveReplayCapture("not_armed")
    if scope not in config.scopes:
        return InactiveReplayCapture("scope_not_selected")
    rank = _current_rank()
    if rank not in config.ranks:
        return InactiveReplayCapture("rank_not_selected")

    _validate_tensors(inputs)
    read_only_state = read_only_state or {}
    mutable_state = mutable_state or {}
    _validate_tensors(read_only_state)
    _validate_tensors(mutable_state)
    if not inputs and not read_only_state and not mutable_state:
        raise ValueError("replay capture requires at least one input or state tensor")
    _json_mapping(required_topology)
    initial_bytes = _tensor_bytes(
        (*inputs.values(), *read_only_state.values(), *mutable_state.values())
    )
    if initial_bytes > config.max_bytes_per_case:
        return InactiveReplayCapture("byte_limit_exceeded")
    if _is_cuda_graph_capture_active(
        (*inputs.values(), *read_only_state.values(), *mutable_state.values())
    ):
        return InactiveReplayCapture("cuda_graph_capture_active")

    case_index = _claim_case(config, scope, rank)
    if case_index is None:
        return InactiveReplayCapture("case_limit_reached")
    return ReplayCapture(
        config=config,
        scope=scope,
        rank=rank,
        case_index=case_index,
        inputs=inputs,
        read_only_state=read_only_state,
        mutable_state=mutable_state,
        metadata=metadata or {},
        required_topology=required_topology,
    )


def _get_config() -> ReplayCaptureConfig | None:
    global _CONFIG
    config = _CONFIG
    if config is not _UNSET:
        return config
    with _LOCK:
        if _CONFIG is _UNSET:
            _CONFIG = ReplayCaptureConfig.from_env()
        return _CONFIG


def _claim_case(config: ReplayCaptureConfig, scope: str, rank: int) -> int | None:
    global _TOTAL_CASES
    key = (scope, rank)
    with _LOCK:
        count = _CASES_BY_SCOPE_RANK.get(key, 0)
        if count >= config.max_cases_per_scope_rank:
            return None
        if _TOTAL_CASES >= config.max_total_cases:
            return None
        _CASES_BY_SCOPE_RANK[key] = count + 1
        _TOTAL_CASES += 1
        return count


def _validate_tensors(tensors: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(tensors, Mapping):
        raise TypeError("replay tensors must be a flat name-to-tensor mapping")
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"unsafe replay tensor name: {name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"replay value {name!r} is not a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"replay tensor {name!r} must use strided layout")


def _clone_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    _validate_tensors(tensors)
    clones = {}
    metadata = {}
    for name, tensor in tensors.items():
        metadata[name] = _tensor_metadata(tensor)
        clones[name] = tensor.detach().clone(memory_format=torch.preserve_format)
    return clones, metadata


def _tensor_bytes(tensors) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "source_device": str(tensor.device),
        "capture_stream": _current_stream_id(tensor),
        "requires_grad": tensor.requires_grad,
        "bytes": tensor.numel() * tensor.element_size(),
    }


def _is_cuda_graph_capture_active(tensors) -> bool:
    if not any(tensor.device.type == "cuda" for tensor in tensors):
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _current_stream_id(tensor: torch.Tensor) -> int | None:
    if tensor.device.type != "cuda":
        return None
    try:
        return int(torch.cuda.current_stream(tensor.device).cuda_stream)
    except RuntimeError:
        return None


def _write_capsule(
    *,
    config,
    scope,
    rank,
    case_index,
    metadata,
    required_topology,
    tensors,
    tensor_metadata,
    total_bytes,
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = re.sub(r"[^A-Za-z0-9_.-]", "_", scope).strip(".") or "scope"
    stem = f"{safe_scope}-rank{rank}-case{case_index}-{time.time_ns()}"
    final_path = config.output_dir / f"{stem}.replay"
    temporary_path = config.output_dir / f".{stem}.{uuid.uuid4().hex}.tmp"
    temporary_path.mkdir()
    try:
        tensor_path = temporary_path / "tensors.safetensors"
        save_file(
            tensors,
            str(tensor_path),
            metadata={"format": "langslanger-replay-capsule-v1"},
        )
        manifest = {
            "schema_version": REPLAY_CAPSULE_SCHEMA_VERSION,
            "scope": scope,
            "rank": rank,
            "case_index": case_index,
            "created_at_ns": time.time_ns(),
            "capture_warning": "perturbed_run_not_performance_evidence",
            "authoritative_serving_timing": False,
            "snapshot_semantics": (
                "clone_enqueued_on_calling_current_stream_before_target; "
                "host serialization after target"
            ),
            "total_tensor_bytes": total_bytes,
            "metadata": metadata,
            "required_topology": required_topology,
            "tensors": tensor_metadata,
            "source_checkout": _source_checkout(),
            "runtime": _runtime_environment(),
            "artifacts": {"tensors.safetensors": {"sha256": _sha256(tensor_path)}},
        }
        manifest_path = temporary_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        complete = {
            "manifest_sha256": _sha256(manifest_path),
            "tensors_sha256": manifest["artifacts"]["tensors.safetensors"]["sha256"],
        }
        (temporary_path / "COMPLETE").write_text(
            json.dumps(complete, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_path.replace(final_path)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    return final_path


def _source_checkout() -> dict[str, Any]:
    root = _git(["rev-parse", "--show-toplevel"])
    commit = _git(["rev-parse", "HEAD"])
    diff = _git(["diff", "--binary", "HEAD"])
    return {
        "root": root,
        "commit": commit,
        "dirty": bool(diff),
        "dirty_diff_sha256": (
            hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
        ),
    }


def _git(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _runtime_environment() -> dict[str, Any]:
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    runtime = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "distributed": {
            "initialized": distributed,
            "rank": torch.distributed.get_rank() if distributed else 0,
            "world_size": torch.distributed.get_world_size() if distributed else 1,
        },
    }
    if runtime["cuda_available"]:
        device = torch.cuda.current_device()
        runtime["cuda_device"] = {
            "index": device,
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return runtime


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("replay metadata must be a mapping")
    try:
        encoded = json.dumps(dict(value), separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError("replay metadata must already be JSON serializable") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("replay metadata must encode to an object")
    return decoded


def _comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
