# RunPod CI runners

LangSlanger's default GPU CI uses only runner labels that match the available RunPod fleet exactly:

| GitHub runner label | Required pod hardware |
| --- | --- |
| `1-gpu-5090` | 1 x NVIDIA RTX 5090 |
| `1-gpu-h100` | 1 x NVIDIA H100 |
| `2-gpu-h100` | 2 x NVIDIA H100 |
| `4-gpu-h100` | 4 x NVIDIA H100 |

B200, 8-GPU H200/H20, and ARM wheel jobs are skipped by default. They remain available through the manual `enable_upstream_full_ci` workflow input, but do not enable it until runners with those exact labels and hardware exist.

## Connect one RunPod pod

No RunPod API key is required for the initial setup. Add your SSH public key in RunPod account settings, then create a pod in the RunPod console with one of the exact GPU shapes above, a current Ubuntu image with CUDA 13.x and Python 3.10, and a persistent `/workspace` volume. Copy the SSH command from the pod's **Connect** tab. RunPod's [SSH guide](https://docs.runpod.io/pods/configuration/use-ssh) covers both its proxied connection and full SSH over a public IP.

Standard RunPod Pods do not expose nested Docker reliably. LangSlanger therefore builds the x64 `sgl-kernel` CI wheel natively on `1-gpu-h100`; the upstream Docker builder remains the default inside the reusable build workflow for non-RunPod callers. Use the same base template across the 5090 and H100 pools so that the native CI wheel sees a consistent userspace.

The repository is public, so use a fresh, isolated pod with no unrelated credentials or mounted data. Register only ephemeral runners and run GPU CI only for reviewed PRs carrying the repository's `run-ci` gate. GitHub [recommends ephemeral runners for autoscaling](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#ephemeral-runners-for-autoscaling) and separately [warns against exposing self-hosted runners to untrusted public-repository PRs](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners).

On a trusted macOS machine where `gh auth status` has repository administration access, copy a short-lived GitHub runner registration token directly to the clipboard without printing it:

```bash
gh api \
  --method POST \
  repos/ek-capital/langslanger/actions/runners/registration-token \
  --jq .token | pbcopy
```

SSH into the RunPod pod, clone LangSlanger, and set the token in that shell. Do not paste the token into chat, a committed file, a RunPod template, or a long-lived secret. The registration token expires after one hour. Then start one exact runner, for example:

```bash
cd /workspace/langslanger
export GITHUB_REPOSITORY=ek-capital/langslanger
export RUNNER_LABELS=1-gpu-h100
read -rsp 'GitHub runner token: ' GITHUB_RUNNER_TOKEN
echo
export GITHUB_RUNNER_TOKEN
bash scripts/ci/runpod/start_github_runner.sh
```

The bootstrap checks the visible GPU count and model before registering, downloads the latest official x64 GitHub Actions runner, and starts it with `--ephemeral`. It accepts only the four labels in the table, so a pod cannot accidentally advertise unsupported hardware.

Verify the runner under **Repository settings -> Actions -> Runners**. When it shows `Idle`, mark the PR ready and add the `run-ci` label. GitHub routes each job only to a runner with the matching label. After one job, the ephemeral runner exits; stop or delete the pod promptly to stop billing. A matrix workflow may require multiple sequential runner starts or multiple pods with the same label.

## Connecting the RunPod account later

Automatic provisioning and scale-to-zero are separate from runner registration. When that automation exists, create a restricted RunPod API key and store it directly as a GitHub Actions secret:

```bash
read -rsp 'RunPod API key: ' RUNPOD_API_KEY
echo
printf '%s' "$RUNPOD_API_KEY" | gh secret set RUNPOD_API_KEY --repo ek-capital/langslanger
unset RUNPOD_API_KEY
```

Do not put the RunPod API key on GPU runner pods. A future provisioner should hold it in GitHub Actions, enforce an explicit maximum hourly price and lifetime, tag every pod deterministically, and always terminate pods after completion or timeout.
