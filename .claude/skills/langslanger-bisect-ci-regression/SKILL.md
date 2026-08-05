---
name: langslanger-bisect-ci-regression
description: Diagnose LangSlanger CI regressions by extracting a stable failure signature, separating code regressions from runner or environment failures, locating the passing-to-failing commit boundary, and optionally reproducing on a GPU host. Use for failing LangSlanger tests, GitHub Actions jobs, flaky hardware runs, or suspected performance regressions. Treat upstream SGLang as an explicit read-only comparison source, never the default target.
---

# Bisect a LangSlanger CI regression

Default to the repository in the current checkout. If it cannot be resolved,
use `ek-capital/langslanger`. Do not target `sgl-project/sglang` unless the user
explicitly requests an upstream comparison. Upstream comparison is read-only:
never dispatch its workflows, change issues, push branches, or open pull
requests there.

## Inputs

Accept a failing GitHub Actions URL or test name. Optional inputs are a known
passing SHA, a known failing SHA, and an explicitly authorized remote GPU host.
Ask before starting paid compute or changing remote state.

## 1. Establish the failure signature

Resolve and record the target repository before querying runs:

```bash
gh repo view --json nameWithOwner --jq .nameWithOwner
gh run view RUN_ID --repo ek-capital/langslanger --json headSha,event,jobs
gh run view RUN_ID --repo ek-capital/langslanger --log-failed
```

Capture the exact test, assertion or error, model and launch configuration,
runner, GPU topology, driver, CUDA, container or package revisions, and head
SHA. Do not group failures together merely because their test names match.

If only a test name is available, inspect recent LangSlanger failures and find
jobs containing that test:

```bash
gh run list --repo ek-capital/langslanger --status failure --limit 30 \
  --json databaseId,workflowName,headSha,event,createdAt,conclusion
```

## 2. Separate code, hardware, environment, and noise

Build a small evidence table with run ID, SHA, runner, GPU, software stack, and
result. Classify only when the evidence supports it:

- the same SHA passing and failing on different runner classes suggests a
  hardware or environment problem;
- all comparable runners failing after one commit suggests a code regression;
- the same environment alternating between pass and fail suggests flakiness or
  a race; and
- incomparable workloads or missing environment evidence leave the cause
  unknown.

For performance regressions, compare identical workloads on the same host when
possible. Use paired runs and preserve raw samples; do not bisect on a single
noisy timing.

## 3. Find the boundary

Identify the last comparable pass and first comparable failure. Inspect the
candidate commits and affected paths:

```bash
git log --oneline LAST_PASS_SHA..FIRST_FAIL_SHA
git log --oneline LAST_PASS_SHA..FIRST_FAIL_SHA -- RELEVANT_PATHS
```

If CI history is insufficient, say so. Do not substitute upstream scheduled CI
for missing LangSlanger evidence.

Run `git bisect` only in an isolated worktree, with one deterministic command
that returns zero for pass and nonzero for fail. Record skipped commits and
clean up the worktree after resetting the bisection. Never bisect by rewriting
the user's active branch.

## 4. Reproduce only when useful

Before remote reproduction, verify the checked-out SHA, image digest, GPU and
interconnect topology, driver, CUDA, dependency versions, environment, and
launch arguments. Reproduce the original configuration first, then vary one
dimension at a time. Useful controls include TP=1 versus TP>1, graph off versus
on, and the same SHA on another runner class.

Use the operator's SSH safety and connection-reuse policy. Do not install over
an existing environment or restart a shared service without authorization.

## 5. Optional upstream comparison

When explicitly requested, use `--repo sgl-project/sglang` only for read-only
run, commit, issue, or pull-request inspection. State clearly which evidence is
from upstream and why it is comparable. Do not infer that an upstream pass
proves LangSlanger passes: patches, dependencies, runners, and workflows may
differ.

## Report

Return:

- the exact failure signature and reproducibility;
- last passing and first failing SHAs;
- a code, hardware, environment, flaky, or unknown classification;
- the smallest evidence table supporting that classification;
- the suspected commit and paths, if proven;
- reproduction commands and artifacts; and
- the next focused fix or measurement.

Prefer `unknown` over a confident diagnosis based on incomparable runs.
