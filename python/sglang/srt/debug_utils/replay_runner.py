"""Benchmark and validate a callable against a LangSlanger replay capsule."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file

ReplayCallable = Callable[
    [
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ],
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
]


def run_replay(
    capsule_path: str | Path,
    *,
    candidate: str | ReplayCallable,
    reference: str | ReplayCallable | None = None,
    device: str = "auto",
    warmup: int = 10,
    iterations: int = 100,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Check outputs, then benchmark candidate/reference with alternating order."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    capsule_path = Path(capsule_path).expanduser().resolve()
    (
        manifest,
        inputs,
        read_only_state,
        mutable_state,
        captured_outputs,
        captured_mutable_state,
    ) = load_replay_capsule(capsule_path, device=device)
    topology_validation = _validate_topology(
        manifest.get("required_topology", {}), _resolve_device(device)
    )
    resolved_device = next(
        (
            tensor.device
            for tensor in (
                *inputs.values(),
                *read_only_state.values(),
                *mutable_state.values(),
                *captured_outputs.values(),
            )
        ),
        torch.device(_resolve_device(device)),
    )
    candidate_fn, candidate_identity = _resolve_callable(candidate)
    reference_fn, reference_identity = (
        _resolve_callable(reference) if reference is not None else (None, None)
    )

    candidate_output, candidate_mutable_state = _invoke(
        candidate_fn,
        inputs,
        read_only_state,
        mutable_state,
        manifest["metadata"],
    )
    expected_output, expected_mutable_state = (
        _invoke(
            reference_fn,
            inputs,
            read_only_state,
            mutable_state,
            manifest["metadata"],
        )
        if reference_fn is not None
        else (captured_outputs, captured_mutable_state)
    )
    output_correctness = _compare_tensors(
        candidate_output, expected_output, rtol=rtol, atol=atol
    )
    mutable_state_correctness = _compare_tensors(
        candidate_mutable_state, expected_mutable_state, rtol=rtol, atol=atol
    )
    correctness = {
        "passed": output_correctness["passed"] and mutable_state_correctness["passed"],
        "outputs": output_correctness,
        "mutable_state": mutable_state_correctness,
    }

    for _ in range(warmup):
        _invoke(
            candidate_fn,
            inputs,
            read_only_state,
            mutable_state,
            manifest["metadata"],
        )
        if reference_fn is not None:
            _invoke(
                reference_fn,
                inputs,
                read_only_state,
                mutable_state,
                manifest["metadata"],
            )
    _synchronize(resolved_device)

    candidate_samples = []
    reference_samples = []
    for iteration in range(iterations):
        ordered = (
            ((candidate_fn, candidate_samples), (reference_fn, reference_samples))
            if iteration % 2 == 0
            else ((reference_fn, reference_samples), (candidate_fn, candidate_samples))
        )
        for function, samples in ordered:
            if function is None:
                continue
            samples.append(
                _time_invocation(
                    function,
                    inputs,
                    read_only_state,
                    mutable_state,
                    manifest["metadata"],
                    resolved_device,
                )
            )

    candidate_summary = _timing_summary(candidate_samples)
    reference_summary = (
        _timing_summary(reference_samples) if reference_samples else None
    )
    return {
        "schema_version": 1,
        "capsule": str(capsule_path),
        "capsule_manifest_sha256": _sha256(capsule_path / "manifest.json"),
        "scope": manifest["scope"],
        "rank": manifest["rank"],
        "device": str(resolved_device),
        "timing_authority": "offline_replay_only",
        "authoritative_serving_timing": False,
        "timing_method": {
            "timed_region": "callable_only",
            "input_and_state_clone": "outside_timed_region",
            "pair_order": "alternating" if reference_fn is not None else None,
            "warmup": warmup,
            "iterations": iterations,
        },
        "runtime": _runtime_environment(resolved_device),
        "topology_validation": topology_validation,
        "candidate": candidate_identity,
        "reference": reference_identity or "captured_outputs",
        "correctness": correctness,
        "candidate_timing": candidate_summary,
        "reference_timing": reference_summary,
        "median_speedup": (
            reference_summary["median_ms"] / candidate_summary["median_ms"]
            if reference_summary and candidate_summary["median_ms"] > 0
            else None
        ),
    }


def load_replay_capsule(capsule_path: str | Path, *, device: str = "auto"):
    capsule_path = Path(capsule_path).expanduser().resolve()
    manifest_path = capsule_path / "manifest.json"
    tensor_path = capsule_path / "tensors.safetensors"
    complete_path = capsule_path / "COMPLETE"
    if not capsule_path.is_dir() or not complete_path.is_file():
        raise ValueError(f"incomplete replay capsule: {capsule_path}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("replay capsule manifest checksum mismatch")
    if complete.get("tensors_sha256") != _sha256(tensor_path):
        raise ValueError("replay capsule tensor checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"unsupported replay capsule schema: {manifest.get('schema_version')}"
        )
    resolved_device = _resolve_device(device)
    tensors = load_file(str(tensor_path), device=resolved_device)
    groups = {
        "input": {},
        "read_only": {},
        "mutable_before": {},
        "output": {},
        "mutable_after": {},
    }
    for name, tensor in tensors.items():
        group, separator, field = name.partition(".")
        if not separator or group not in groups or not field:
            raise ValueError(f"invalid replay tensor key: {name!r}")
        groups[group][field] = tensor
    return (
        manifest,
        groups["input"],
        groups["read_only"],
        groups["mutable_before"],
        groups["output"],
        groups["mutable_after"],
    )


def _invoke(function, inputs, read_only_state, mutable_state, metadata):
    call_inputs = {name: tensor.clone() for name, tensor in inputs.items()}
    call_read_only_state = {
        name: tensor.clone() for name, tensor in read_only_state.items()
    }
    call_mutable_state = {
        name: tensor.clone() for name, tensor in mutable_state.items()
    }
    with torch.inference_mode():
        result = function(
            call_inputs, call_read_only_state, call_mutable_state, metadata
        )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "replay callable must return (outputs, mutable_state) tensor mappings"
        )
    outputs, mutated_state = result
    _validate_tensor_mapping(outputs, "outputs")
    _validate_tensor_mapping(mutated_state, "mutable_state")
    if set(mutated_state) != set(mutable_state):
        raise ValueError(
            "replay callable mutable state names must match the capsule declaration"
        )
    return outputs, mutated_state


def _time_invocation(
    function, inputs, read_only_state, mutable_state, metadata, device
) -> float:
    call_inputs = {name: tensor.clone() for name, tensor in inputs.items()}
    call_read_only_state = {
        name: tensor.clone() for name, tensor in read_only_state.items()
    }
    call_mutable_state = {
        name: tensor.clone() for name, tensor in mutable_state.items()
    }
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            function(call_inputs, call_read_only_state, call_mutable_state, metadata)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    started = time.perf_counter_ns()
    with torch.inference_mode():
        function(call_inputs, call_read_only_state, call_mutable_state, metadata)
    return (time.perf_counter_ns() - started) / 1e6


def _compare_tensors(actual, expected, *, rtol, atol):
    if set(actual) != set(expected):
        return {
            "passed": False,
            "rtol": rtol,
            "atol": atol,
            "missing": sorted(set(expected) - set(actual)),
            "unexpected": sorted(set(actual) - set(expected)),
            "tensors": [],
        }
    rows = []
    passed = True
    for name in sorted(actual):
        value = actual[name]
        reference = expected[name].to(value.device)
        same_shape = value.shape == reference.shape
        same_dtype = value.dtype == reference.dtype
        if not same_shape or not same_dtype:
            row = {
                "name": name,
                "passed": False,
                "actual_shape": list(value.shape),
                "expected_shape": list(reference.shape),
                "actual_dtype": str(value.dtype),
                "expected_dtype": str(reference.dtype),
            }
        elif value.is_floating_point() or value.is_complex():
            difference = (value - reference).abs()
            max_abs = float(difference.max().item()) if difference.numel() else 0.0
            denominator = reference.abs().clamp_min(atol)
            max_rel = (
                float((difference / denominator).max().item())
                if difference.numel()
                else 0.0
            )
            row = {
                "name": name,
                "passed": bool(torch.allclose(value, reference, rtol=rtol, atol=atol)),
                "actual_shape": list(value.shape),
                "expected_shape": list(reference.shape),
                "max_abs": max_abs,
                "max_rel": max_rel,
            }
        else:
            row = {
                "name": name,
                "passed": bool(torch.equal(value, reference)),
                "actual_shape": list(value.shape),
                "expected_shape": list(reference.shape),
            }
        rows.append(row)
        passed = passed and row["passed"]
    return {
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        "missing": [],
        "unexpected": [],
        "tensors": rows,
    }


def _validate_tensor_mapping(value, label):
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in value.items()
    ):
        raise TypeError(f"replay callable {label} must be dict[str, torch.Tensor]")


def _timing_summary(samples):
    ordered = sorted(samples)
    return {
        "iterations": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p10_ms": _percentile(ordered, 0.10),
        "p90_ms": _percentile(ordered, 0.90),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def _percentile(ordered, quantile):
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _resolve_callable(value):
    if callable(value):
        function = value
        identity = f"{function.__module__}:{function.__qualname__}"
    else:
        module_name, separator, attribute = value.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("callable must use module.path:function format")
        function = getattr(importlib.import_module(module_name), attribute)
        if not callable(function):
            raise TypeError(f"{value} is not callable")
        identity = value
    try:
        source_path = inspect.getsourcefile(function)
    except TypeError:
        source_path = None
    source = Path(source_path) if source_path else None
    return function, {
        "callable": identity,
        "source_path": source_path,
        "source_sha256": _sha256(source) if source and source.is_file() else None,
        "source_git_blob": _git_blob(source) if source and source.is_file() else None,
    }


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _validate_topology(required, device: str) -> dict[str, Any]:
    if not isinstance(required, dict) or "world_size" not in required:
        raise ValueError("required_topology must declare world_size")
    supported = {"world_size", "device_type", "minimum_compute_capability"}
    unsupported = set(required) - supported
    if unsupported:
        raise ValueError(
            "unsupported replay topology requirements: "
            + ", ".join(sorted(unsupported))
        )
    actual_world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    actual_device = torch.device(device)
    actual = {
        "world_size": actual_world_size,
        "device_type": actual_device.type,
        "compute_capability": None,
    }
    if actual_device.type == "cuda":
        actual["compute_capability"] = list(
            torch.cuda.get_device_capability(actual_device)
        )
    if int(required["world_size"]) != actual_world_size:
        raise ValueError(
            f"capsule requires world_size={required['world_size']}, "
            f"runner has {actual_world_size}"
        )
    if required.get("device_type") not in (None, actual_device.type):
        raise ValueError(
            f"capsule requires device_type={required['device_type']}, "
            f"runner has {actual_device.type}"
        )
    minimum = required.get("minimum_compute_capability")
    if minimum is not None:
        if actual["compute_capability"] is None or tuple(
            actual["compute_capability"]
        ) < tuple(minimum):
            raise ValueError(
                f"capsule requires compute capability >= {minimum}, "
                f"runner has {actual['compute_capability']}"
            )
    return {"passed": True, "required": required, "actual": actual}


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _runtime_environment(device) -> dict[str, Any]:
    runtime = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        runtime["cuda_device"] = {
            "index": device.index,
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return runtime


def _git_blob(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "hash-object", str(path)],
            cwd=path.parent,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run_replay(
        args.capsule,
        candidate=args.candidate,
        reference=args.reference,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
        print(output)
    else:
        print(encoded, end="")
    return 0 if result["correctness"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
