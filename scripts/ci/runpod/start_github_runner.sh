#!/usr/bin/env bash

set -euo pipefail

readonly SUPPORTED_LABELS="1-gpu-5090 1-gpu-h100 2-gpu-h100 4-gpu-h100"

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

verify_hardware() {
  local expected_count expected_model
  case "$RUNNER_LABELS" in
    1-gpu-5090)
      expected_count=1
      expected_model="RTX 5090"
      ;;
    1-gpu-h100)
      expected_count=1
      expected_model="H100"
      ;;
    2-gpu-h100)
      expected_count=2
      expected_model="H100"
      ;;
    4-gpu-h100)
      expected_count=4
      expected_model="H100"
      ;;
    *)
      die "unsupported RUNNER_LABELS=$RUNNER_LABELS; expected one of: $SUPPORTED_LABELS"
      ;;
  esac

  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r')
  [[ "${#gpu_names[@]}" -eq "$expected_count" ]] ||
    die "$RUNNER_LABELS requires $expected_count GPU(s), found ${#gpu_names[@]}: ${gpu_names[*]:-none}"

  local gpu_name
  for gpu_name in "${gpu_names[@]}"; do
    [[ "$gpu_name" == *"$expected_model"* ]] ||
      die "$RUNNER_LABELS requires $expected_model GPUs, found: ${gpu_names[*]}"
  done

  echo "verified hardware for $RUNNER_LABELS: ${gpu_names[*]}"
}

resolve_runner_asset_url() {
  if [[ -n "${GITHUB_RUNNER_VERSION:-}" ]]; then
    local version="${GITHUB_RUNNER_VERSION#v}"
    echo "https://github.com/actions/runner/releases/download/v${version}/actions-runner-linux-x64-${version}.tar.gz"
    return
  fi

  curl -fsSL https://api.github.com/repos/actions/runner/releases/latest |
    sed -n 's/.*"browser_download_url": "\([^"]*actions-runner-linux-x64-[^"]*\.tar\.gz\)".*/\1/p'
}

[[ "$(uname -s)" == "Linux" ]] || die "GitHub GPU runners must run on Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "this bootstrap supports x86_64 only"

: "${GITHUB_REPOSITORY:=ek-capital/langslanger}"
: "${GITHUB_RUNNER_TOKEN:?set GITHUB_RUNNER_TOKEN to a short-lived GitHub runner registration token}"
: "${RUNNER_LABELS:?set RUNNER_LABELS to one exact LangSlanger runner label}"
: "${RUNNER_ROOT:=/workspace/github-runners}"

require_command curl
require_command nvidia-smi
require_command sed
require_command tar
verify_hardware

RUNNER_NAME="${RUNNER_NAME:-runpod-${RUNPOD_POD_ID:-$(hostname)}-$(date +%s)}"
[[ "$RUNNER_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUNNER_NAME may contain only letters, digits, dot, underscore, and hyphen"

readonly RUNNER_DIR="${RUNNER_ROOT}/${RUNNER_NAME}"
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

runner_asset_url="$(resolve_runner_asset_url)"
[[ -n "$runner_asset_url" ]] || die "could not resolve the latest GitHub Actions runner asset"
curl --fail --location --retry 3 --output actions-runner.tar.gz "$runner_asset_url"
tar xzf actions-runner.tar.gz
rm actions-runner.tar.gz

if [[ "$(id -u)" -eq 0 ]]; then
  ./bin/installdependencies.sh
  if ! command -v sudo >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y sudo
  fi
  export RUNNER_ALLOW_RUNASROOT=1
fi

./config.sh \
  --unattended \
  --ephemeral \
  --url "https://github.com/${GITHUB_REPOSITORY}" \
  --token "$GITHUB_RUNNER_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "$RUNNER_LABELS" \
  --work _work

unset GITHUB_RUNNER_TOKEN
exec ./run.sh
