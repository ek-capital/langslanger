# Warm DeepSeek-V4 / B200 runtime. The base digest is the linux/amd64 image
# behind lmsysorg/sglang:v0.5.16, pinned so nightly rebuilds are reproducible.
FROM lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c

ARG SOURCE_COMMIT="unknown"
ARG DEEP_GEMM_PACKAGE_VERSION="0.1.5.post2"
ARG FLASHINFER_PACKAGE_VERSION="0.6.15.post1"

LABEL org.opencontainers.image.title="LangSlanger B200 DeepSeek-V4 runtime" \
      org.opencontainers.image.description="SGLang-compatible LangSlanger runtime with SM100 DeepSeek-V4 kernels precompiled" \
      org.opencontainers.image.source="https://github.com/ek-capital/langslanger" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="docker.io/lmsysorg/sglang:v0.5.16" \
      org.opencontainers.image.base.digest="sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c"

# Match the exact FlashInfer trio and DeepGEMM release used by the profiled
# checkout. The official image already carries the 1,696-entry TRT-LLM MXFP4
# MoE cubin pool; these wheel installs add no model-weight data.
RUN python3 -m pip uninstall -y \
      flashinfer-python flashinfer-cubin flashinfer-jit-cache && \
    rm -rf /root/.cache/flashinfer /root/.cache/pip && \
    python3 -m pip install --no-deps \
      "flashinfer-python==${FLASHINFER_PACKAGE_VERSION}" && \
    python3 -m pip install --no-deps \
      "flashinfer-cubin==${FLASHINFER_PACKAGE_VERSION}" \
      --index-url https://flashinfer.ai/whl && \
    python3 -m pip install --no-deps \
      "flashinfer-jit-cache==${FLASHINFER_PACKAGE_VERSION}" \
      --index-url https://flashinfer.ai/whl/cu130 && \
    rm -rf /root/.cache/flashinfer /root/.cache/pip
RUN python3 -m pip install --no-deps --force-reinstall \
      "sgl-deep-gemm==${DEEP_GEMM_PACKAGE_VERSION}" && \
    rm -rf /root/.cache/pip

# Preserve the base image's editable installation and compiled extensions while
# replacing only the Python compatibility namespace with this exact checkout.
RUN rm -rf /sgl-workspace/sglang/python/sglang
COPY python/sglang /sgl-workspace/sglang/python/sglang

ENV TVM_FFI_CACHE_DIR="/opt/langslanger/cache/tvm-ffi" \
    LANGSLANGER_TARGET_GPU="NVIDIA B200 (SM100)" \
    LANGSLANGER_SOURCE_COMMIT="${SOURCE_COMMIT}"

COPY docker/precompile_deepseek_v4_sm100.py /opt/langslanger/precompile_deepseek_v4_sm100.py

# nvcc cross-compiles these architecture-specific modules without a GPU. Any
# source/config mismatch is a build failure rather than a cold-start surprise.
RUN mkdir -p "${TVM_FFI_CACHE_DIR}" && \
    python3 /opt/langslanger/precompile_deepseek_v4_sm100.py && \
    test "$(find "${TVM_FFI_CACHE_DIR}" -type f -name '*.so' | wc -l)" -ge 19 && \
    find "${TVM_FFI_CACHE_DIR}" -type f -name '*.so' \
      -path '*__arch_10.0a__*' | grep -q . && \
    rm -f /opt/langslanger/precompile_deepseek_v4_sm100.py

# Validate the warmed dependency surface and retain the upstream `sglang`
# command. `langslanger` is a thin alias over the same parser/runtime.
RUN python3 - <<'PY'
from importlib.metadata import version
from pathlib import Path

import sglang

expected = "0.6.15.post1"
for package in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"):
    assert version(package).split("+", 1)[0] == expected, (package, version(package))
assert version("sgl-deep-gemm").split("+", 1)[0] == "0.1.5.post2"
assert Path(sglang.__file__).resolve().is_relative_to(
    Path("/sgl-workspace/sglang/python")
), sglang.__file__
cubin_pool = Path("/opt/trtllm_gen_moe_cubin_pool")
assert len(list(cubin_pool.rglob("*.cubin"))) == 1696
PY
RUN ln -s "$(command -v sglang)" /usr/local/bin/langslanger && \
    sglang version && \
    langslanger version

WORKDIR /sgl-workspace/sglang
