"""Launch a serving command under a short, API-triggered Nsight graph trace.

The server must start and stop ``CUDA_PROFILER`` through its profiling API.
Nsight stays idle during model load, captures only that window, and records
CUDA graph nodes so ``profile_graph_map`` can join kernels to declarations.
Software CUDA tracing is deliberate: Nsight's hardware tracer can make a large
LLM graph replay take minutes per node-level capture on recent GPUs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_nsys_command(output: str | Path, command: list[str]) -> list[str]:
    if not command:
        raise ValueError("a server command is required")
    output = Path(output).expanduser().resolve()
    return [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--trace=cuda-sw,nvtx,osrt",
        "--trace-fork-before-exec=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-graph-trace=node",
        "--output",
        str(output),
        "--",
        *command,
    ]


def run_nsys(output: str | Path, command: list[str]) -> int:
    if shutil.which("nsys") is None:
        raise RuntimeError("nsys is not installed or is not on PATH")
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(build_nsys_command(output, command), check=False)
    if completed.returncode:
        raise RuntimeError(f"Nsight-wrapped command exited {completed.returncode}")
    return completed.returncode


def export_nsys_sqlite(
    report_path: str | Path, output_path: str | Path | None = None
) -> Path:
    if shutil.which("nsys") is None:
        raise RuntimeError("nsys is not installed or is not on PATH")
    report_path = Path(report_path).expanduser().resolve()
    if not report_path.is_file():
        raise ValueError(f"Nsight report does not exist: {report_path}")
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else report_path.with_suffix(".sqlite")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={output_path}",
            str(report_path),
        ],
        check=True,
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Nsight output stem")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_nsys(args.output, command)


if __name__ == "__main__":
    raise SystemExit(main())
