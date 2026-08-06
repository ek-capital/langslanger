# syntax=docker/dockerfile:1.7

# Keep the dependency stack aligned with LangSlanger's documented upstream
# release. The digest makes a rebuild of the same LangSlanger commit reproducible.
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.16-cu130@sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0
FROM ${BASE_IMAGE}

ARG LANGSLANGER_GIT_SHA

LABEL org.opencontainers.image.title="LangSlanger CUDA 13"
LABEL org.opencontainers.image.description="Experimental SGLang-compatible research runtime"
LABEL org.opencontainers.image.source="https://github.com/ek-capital/langslanger"
LABEL org.opencontainers.image.revision="${LANGSLANGER_GIT_SHA}"

# The upstream image already carries the CUDA, PyTorch, FlashInfer, and SGLang
# dependency stack. Build only this fork's package and native extensions, once,
# in CI. Keep the checkout in the image so profiler source attribution can
# resolve paths and Git blob hashes.
COPY . /sgl-workspace/langslanger

RUN test "$(git -C /sgl-workspace/langslanger rev-parse HEAD)" = "${LANGSLANGER_GIT_SHA}" \
    && python3 -m pip install \
        --no-cache-dir \
        --no-deps \
        --force-reinstall \
        -e /sgl-workspace/langslanger/python \
    && python3 -c "from pathlib import Path; import sglang; p = Path(sglang.__file__).resolve(); expected = Path('/sgl-workspace/langslanger/python'); assert p.is_relative_to(expected), (p, expected); print(p)" \
    && python3 -c "from sglang.srt.grpc import _core as grpc_core; from sglang.srt.multimodal import _core as mm_core; print(grpc_core, mm_core)" \
    && sglang version

ENV LANGSLANGER_IMAGE_REVISION=${LANGSLANGER_GIT_SHA}
WORKDIR /sgl-workspace/langslanger
