"""Bake DeepSeek-V4 CUDA JIT modules for B200 into a container image.

This script only instantiates compile-time module variants. It does not launch a
kernel or require a visible GPU; ``override_jit_cuda_arch`` makes nvcc target
SM100 explicitly.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from sglang.kernels.jit.utils import override_jit_cuda_arch


def main() -> None:
    # Import inside the override so architecture-dependent memoized values are
    # first resolved for B200 rather than for the GPU-less image builder.
    with override_jit_cuda_arch(10, 0):
        from sglang.kernels.ops.attention.dsv4 import attn
        from sglang.kernels.ops.attention.dsv4 import compress
        from sglang.kernels.ops.attention.dsv4 import elementwise
        from sglang.kernels.ops.attention.dsv4 import fp8_wo_a
        from sglang.kernels.ops.attention.dsv4 import moe
        from sglang.kernels.ops.attention.dsv4 import topk

        builds: list[tuple[str, Callable[[], object]]] = [
            ("attention metadata", attn._jit_metadata_module),
            (
                "FlashMLA cache store bf16/i32/p64",
                lambda: attn._jit_fused_store_module(
                    "flashmla", torch.bfloat16, torch.int32, 64
                ),
            ),
            (
                "indexer cache store bf16/i32/p64",
                lambda: attn._jit_fused_store_module(
                    "indexer", torch.bfloat16, torch.int32, 64
                ),
            ),
            ("fused QK RoPE", elementwise._jit_fused_rope_module),
            (
                "main Q norm+RoPE bf16/d512/r64",
                lambda: elementwise._jit_main_q_norm_rope_module(
                    torch.bfloat16, 512, 64
                ),
            ),
            (
                "main K norm+RoPE FlashMLA bf16/d512/r64/p64",
                lambda: elementwise._jit_main_k_norm_rope_flashmla_module(
                    torch.bfloat16, 512, 64, 64
                ),
            ),
            (
                "indexer Q RoPE+Hadamard+FP8",
                lambda: elementwise._jit_main_q_indexer_rope_hadamard_quant_module(
                    torch.bfloat16
                ),
            ),
            (
                "indexer Q RoPE+Hadamard+FP4",
                lambda: elementwise._jit_main_q_indexer_rope_hadamard_fp4_quant_module(
                    torch.bfloat16
                ),
            ),
            ("compression plan", compress._jit_compress_plan_module),
            (
                "C4 compress bf16",
                lambda: compress._jit_compress_module(
                    512,
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                    4,
                ),
            ),
            (
                "C128 compress bf16",
                lambda: compress._jit_compress_module(
                    512,
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                    128,
                ),
            ),
            (
                "C128 online compress bf16",
                lambda: compress._jit_compress_128_online_module(512, torch.bfloat16),
            ),
            (
                "compress norm+RoPE bf16/d512/r64/p64",
                lambda: compress._jit_compress_norm_rope_module(
                    torch.bfloat16, 512, 64, 64, False
                ),
            ),
            (
                "indexer norm+RoPE FP4 bf16/d128/r64/p64",
                lambda: compress._jit_compress_norm_rope_module(
                    torch.bfloat16, 128, 64, 64, False
                ),
            ),
            ("attention top-k v1", topk._jit_topk_v1_module),
            ("attention top-k v2", topk._jit_topk_v2_module),
            ("MoE mask top-k", moe._jit_mask_topk_module),
            ("MoE hash top-k", moe._jit_hash_topk_module),
            (
                "FP8 wo_a group-major quant bf16",
                lambda: fp8_wo_a._jit_module(torch.bfloat16, True),
            ),
        ]

        for index, (name, build) in enumerate(builds, start=1):
            print(f"[{index:02d}/{len(builds):02d}] {name}", flush=True)
            build()

    print(f"Baked {len(builds)} DeepSeek-V4 SM100 JIT modules.", flush=True)


if __name__ == "__main__":
    main()
