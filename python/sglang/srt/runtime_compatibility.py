"""Runtime bands supported by the LangSlanger source overlay.

Binary extension packages must move with the PyTorch ABI they were built for.
Keeping the combinations here explicit lets a checkout run on a known SGLang
image without letting a package resolver create an arbitrary mixture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet
from packaging.version import Version


@dataclass(frozen=True)
class CudaRuntimeBand:
    name: str
    torch: str
    sglang_kernel: str
    flashinfer_python: str


OFFICIAL_SGLANG_IMAGE_BAND = CudaRuntimeBand(
    name="official-sglang-v0.5.16",
    torch=">=2.11,<2.12",
    sglang_kernel=">=0.4.5,<0.4.6",
    flashinfer_python=">=0.6.14,<0.6.15",
)

LANGSLANGER_CURRENT_BAND = CudaRuntimeBand(
    name="langslanger-current",
    torch=">=2.13,<2.14",
    sglang_kernel=">=0.4.6.post1,<0.5",
    flashinfer_python=">=0.6.15.post1,<0.7",
)


def _matches(installed: str, requirement: str) -> bool:
    return Version(installed) in SpecifierSet(requirement)


def select_cuda_runtime_band(
    torch_version: str, *, strict_current: bool = False
) -> CudaRuntimeBand:
    bands = (LANGSLANGER_CURRENT_BAND,)
    if not strict_current:
        bands = (OFFICIAL_SGLANG_IMAGE_BAND, *bands)

    for band in bands:
        if _matches(torch_version, band.torch):
            return band

    supported = ", ".join(f"{band.name} (torch{band.torch})" for band in bands)
    raise RuntimeError(
        f"Unsupported CUDA runtime: torch {torch_version}. "
        f"Supported LangSlanger runtime bands: {supported}. "
        "Use a matching prebuilt image instead of replacing binary packages in-place."
    )


def validate_cuda_runtime_band(
    torch_version: str,
    *,
    attention_backend: str,
    strict_current: bool = False,
    distribution_version: Callable[[str], str] = version,
) -> CudaRuntimeBand:
    band = select_cuda_runtime_band(torch_version, strict_current=strict_current)
    requirements = [("sglang-kernel", band.sglang_kernel)]
    if attention_backend == "flashinfer":
        requirements.append(("flashinfer_python", band.flashinfer_python))

    failures = []
    for package, requirement in requirements:
        try:
            installed = distribution_version(package)
        except PackageNotFoundError:
            failures.append(f"{package} is not installed (requires {requirement})")
            continue
        if not _matches(installed, requirement):
            failures.append(f"{package} {installed} does not satisfy {requirement}")

    if failures:
        details = "; ".join(failures)
        raise RuntimeError(
            f"Incompatible {band.name} binary runtime: {details}. "
            "Do not upgrade one compiled package inside the image; choose a coherent "
            "runtime band or rebuild the complete image in CI."
        )
    return band
