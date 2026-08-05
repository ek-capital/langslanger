"""Fail when LangSlanger CI can silently target unavailable or wrong hardware."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
MANUAL_ONLY_WORKFLOWS = (
    "pr-test-musa.yml",
    "pr-test-npu.yml",
    "pr-test-xeon.yml",
    "pr-test-xpu.yml",
)
OPT_IN_PR_WORKFLOWS = (
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

    for name in OPT_IN_PR_WORKFLOWS:
        workflow = load_workflow(name)
        types = set(workflow["on"]["pull_request"].get("types", []))
        require("ready_for_review" in types, f"{name} misses ready_for_review")
        require("labeled" in types, f"{name} misses labeled")

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
