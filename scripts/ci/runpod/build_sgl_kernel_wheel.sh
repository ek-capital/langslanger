#!/usr/bin/env bash

set -euo pipefail

command -v nvcc >/dev/null 2>&1 || {
  echo "error: the native sgl-kernel build requires nvcc" >&2
  exit 1
}

cuda_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')"
[[ "$cuda_release" == 13.* ]] || {
  echo "error: the native sgl-kernel build requires CUDA 13.x; found ${cuda_release:-unknown}" >&2
  exit 1
}

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ccache \
  libibverbs-dev \
  libnuma-dev

python3 -m pip install \
  ninja \
  numpy \
  scikit-build-core \
  setuptools==75.0.0 \
  uv \
  wheel==0.41.0
python3 -m pip install \
  torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

cuda_home="${CUDA_HOME:-/usr/local/cuda}"
export CPLUS_INCLUDE_PATH="${cuda_home}/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export C_INCLUDE_PATH="${cuda_home}/include/cccl${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export FLASHINFER_CUDA_ARCH_LIST="8.0 8.9 9.0a 10.0a 12.0a"
export CCACHE_DIR="${HOME}/.cache/sgl-kernel/ccache"
export CCACHE_BASEDIR="${GITHUB_WORKSPACE:-$(pwd)}/sgl-kernel"
export CCACHE_MAXSIZE=10G
export CCACHE_COMPILERCHECK=content
export CCACHE_COMPRESS=true
export CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
detected_jobs="$(( $(nproc) * 2 / 3 ))"
if [[ "$detected_jobs" -lt 1 ]]; then
  detected_jobs=1
elif [[ "$detected_jobs" -gt 64 ]]; then
  detected_jobs=64
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$detected_jobs}"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DSGL_KERNEL_COMPILE_THREADS=${NVCC_THREADS:-32} -DGITHUB_ARTIFACTORY=${GITHUB_ARTIFACTORY:-github.com}"

mkdir -p "$CCACHE_DIR"
cd "${GITHUB_WORKSPACE:-$(pwd)}/sgl-kernel"
python3 -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
PYTHON=python3 ./rename_wheels.sh
