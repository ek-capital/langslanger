"""Fail when LangSlanger CI can silently target unavailable or wrong hardware."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
REMOVED_UPSTREAM_AUTOMATION_PREFIXES = ("bot-", "release-")
REMOVED_UPSTREAM_AUTOMATION_NAMES = {
    "_docker-build-and-publish.yml",
    "_docker-cleanup-nightly.yml",
    "retag-docker.yml",
    "sync-lmsys-sglang-blogs.yml",
}
MANUAL_ONLY_WORKFLOWS = (
    "pr-test-musa.yml",
    "pr-test-npu.yml",
    "pr-test-xeon.yml",
    "pr-test-xpu.yml",
)
DISPATCHED_HARDWARE_WORKFLOWS = (
    "pr-test.yml",
    "pr-test-extra.yml",
    "pr-test-amd.yml",
    "pr-test-amd-extra.yml",
    "pr-test-arm64.yml",
    "pr-test-mlx.yml",
)
GATE_JOBS = {
    "pr-test.yml": ("call-gate", ("run-ci",)),
    "pr-test-extra.yml": ("call-gate", ("run-ci", "run-ci-extra")),
    "pr-test-amd.yml": ("call-gate", ("run-ci",)),
    "pr-test-amd-extra.yml": ("call-gate", ("run-ci", "run-ci-extra")),
    "pr-test-arm64.yml": ("pr-gate", ("run-ci",)),
    "pr-test-mlx.yml": ("pr-gate", ("run-ci",)),
}
DISPATCH_JOBS = {
    "base": ("pr-test.yml", ("run-ci",)),
    "extra": ("pr-test-extra.yml", ("run-ci", "run-ci-extra")),
    "amd": ("pr-test-amd.yml", ("run-ci",)),
    "amd-extra": ("pr-test-amd-extra.yml", ("run-ci", "run-ci-extra")),
    "arm64": ("pr-test-arm64.yml", ("run-ci",)),
    "mlx": ("pr-test-mlx.yml", ("run-ci",)),
}
UNSAFE_SKIPPED_GATE = (
    "needs.call-gate.result == 'success' || " "needs.call-gate.result == 'skipped'"
)


def load_workflow(name: str):
    with (WORKFLOW_DIR / name).open(encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.BaseLoader)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    workflow_names = {path.name for path in WORKFLOW_DIR.glob("*.yml")}
    forbidden_names = {
        name
        for name in workflow_names
        if name.startswith(REMOVED_UPSTREAM_AUTOMATION_PREFIXES)
    }
    forbidden_names.update(workflow_names & REMOVED_UPSTREAM_AUTOMATION_NAMES)
    require(
        not forbidden_names,
        "upstream publishing or bot workflows returned: "
        + ", ".join(sorted(forbidden_names)),
    )

    image_workflow = load_workflow("langslanger-image.yml")
    require(
        set(image_workflow["on"]) == {"workflow_dispatch"},
        "container publishing must remain manual until a builder is provisioned",
    )
    image_source = (WORKFLOW_DIR / "langslanger-image.yml").read_text(encoding="utf-8")
    for required in (
        "ghcr.io/ek-capital/langslanger",
        "packages: write",
        "BRANCH_TYPE=local",
        "github.repository == 'ek-capital/langslanger'",
    ):
        require(required in image_source, f"image workflow is missing {required!r}")
    for forbidden in ("lmsysorg/sglang", "sgl-project/whl", "pypi"):
        require(
            forbidden not in image_source,
            f"image workflow still targets upstream publishing surface {forbidden!r}",
        )

    for name in MANUAL_ONLY_WORKFLOWS:
        triggers = load_workflow(name)["on"]
        automatic = {"push", "pull_request"}.intersection(triggers)
        require(not automatic, f"{name} has automatic triggers: {sorted(automatic)}")
        require("workflow_dispatch" in triggers, f"{name} lost manual dispatch")

    arm_workflow = load_workflow("pr-test-arm64.yml")
    source_step = next(
        step
        for step in arm_workflow["jobs"]["build-test"]["steps"]
        if step.get("name") == "Resolve source revision"
    )
    build_step = next(
        step
        for step in arm_workflow["jobs"]["build-test"]["steps"]
        if step.get("name") == "Build container"
    )
    require(
        "github.repository" in source_step["env"]["CURRENT_REPO"],
        "Arm64 source is not the current repo",
    )
    require("head.sha" in source_step["env"]["PR_SHA"], "Arm64 PR ref is not a SHA")
    require("git rev-parse --verify HEAD" in source_step["run"], "Arm64 ref is mutable")
    require(
        "steps.source.outputs.repo" in build_step["env"]["SGLANG_SOURCE_REPO"],
        "Arm64 build bypasses the resolved repo",
    )
    require(
        "steps.source.outputs.ref" in build_step["env"]["SGLANG_SOURCE_REF"],
        "Arm64 build bypasses the resolved SHA",
    )

    dockerfile = (ROOT / "docker" / "arm64.Dockerfile").read_text(encoding="utf-8")
    require(
        "sgl-project/sglang.git" not in dockerfile, "Arm64 has an upstream fallback"
    )
    require("ARG SGLANG_REPO=" not in dockerfile, "Arm64 repo argument has a default")
    require("ARG VER_SGLANG=" not in dockerfile, "Arm64 ref argument has a default")

    gate_workflow = load_workflow("pr-gate.yml")
    gate_condition = gate_workflow["jobs"]["pr-gate"]["if"]
    for token in ("draft == false", "run-ci", "run-ci-extra", "event.action"):
        require(token in gate_condition, f"PR gate condition is missing {token!r}")
    gate_source = (WORKFLOW_DIR / "pr-gate.yml").read_text(encoding="utf-8")
    require("PR is draft. Blocking CI." not in gate_source, "Drafts still fail CI")

    dispatcher = load_workflow("langslanger-pr-hardware.yml")
    pr_trigger = dispatcher["on"]["pull_request"]
    for event_type in ("ready_for_review", "labeled"):
        require(event_type in pr_trigger["types"], f"dispatcher misses {event_type}")

    for job_name, (workflow_name, labels) in DISPATCH_JOBS.items():
        job = dispatcher["jobs"][job_name]
        require(
            workflow_name in job["uses"],
            f"dispatcher job {job_name} does not call {workflow_name}",
        )
        require("draft == false" in job["if"], f"{job_name} accepts draft PRs")
        for label in labels:
            require(f"'{label}'" in job["if"], f"{job_name} misses {label!r}")

    for name in DISPATCHED_HARDWARE_WORKFLOWS:
        workflow = load_workflow(name)
        require(
            "pull_request" not in workflow["on"],
            f"{name} bypasses the PR hardware dispatcher",
        )
        require("workflow_call" in workflow["on"], f"{name} is not reusable")

        gate_job, labels = GATE_JOBS[name]
        caller_condition = workflow["jobs"][gate_job].get("if", "")
        require(
            "draft == false" in caller_condition,
            f"{name} does not reject draft PRs at the caller",
        )
        for label in labels:
            require(
                f"'{label}'" in caller_condition,
                f"{name} does not require {label!r} at the caller",
            )

    for name in ("pr-test.yml", "pr-test-extra.yml", "pr-test-amd.yml"):
        source = (WORKFLOW_DIR / name).read_text(encoding="utf-8")
        require(
            UNSAFE_SKIPPED_GATE not in source,
            f"{name} dispatches after a skipped PR gate",
        )

    print("LangSlanger CI policy checks passed")


if __name__ == "__main__":
    main()
