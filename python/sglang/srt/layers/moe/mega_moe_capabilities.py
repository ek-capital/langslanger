"""DeepGEMM capability checks for the MegaMoE execution boundary."""

from __future__ import annotations

_SM90_FP8_RUNTIME_SYMBOLS = (
    "fp8_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "mega_moe_pre_dispatch_sm90",
)
_SM90_FP8_LAYOUT_SYMBOLS = (
    "transform_sf_into_required_layout",
    "transform_weights_for_mega_moe_sm90",
)


def missing_sm90_fp8_mega_moe_symbols(
    deep_gemm, *, use_in_place_weights: bool
) -> tuple[str, ...]:
    required = _SM90_FP8_RUNTIME_SYMBOLS
    if not use_in_place_weights:
        required += _SM90_FP8_LAYOUT_SYMBOLS
    return tuple(symbol for symbol in required if not hasattr(deep_gemm, symbol))


def require_sm90_fp8_mega_moe_capabilities(
    deep_gemm, *, use_in_place_weights: bool
) -> None:
    missing = missing_sm90_fp8_mega_moe_symbols(
        deep_gemm, use_in_place_weights=use_in_place_weights
    )
    if missing:
        raise RuntimeError(
            "MegaMoE on SM90 requires DeepGEMM capabilities missing from this "
            f"runtime: {', '.join(missing)}. Use a compatible prebuilt runtime "
            "band or rebuild the image in CI."
        )
