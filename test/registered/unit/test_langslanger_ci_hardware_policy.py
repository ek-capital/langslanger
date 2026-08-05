"""Structural tests for LangSlanger's affordable GPU CI policy."""

import unittest
from pathlib import Path

import yaml

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[3]
FULL_CI_EXPRESSION = "inputs.enable_upstream_full_ci == true"
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

WORKFLOWS = {
    "pr-test.yml": {
        "triggers": ("workflow_dispatch", "workflow_call"),
        "unsupported_jobs": (
            "sgl-kernel-build-wheels-arm",
            "base-b-test-4-gpu-b200",
            "base-c-test-8-gpu-h200",
            "base-c-test-8-gpu-h20",
            "base-c-test-deepep-4-gpu-b200",
            "base-c-test-deepep-8-gpu-h200",
            "base-c-test-4-gpu-b200",
        ),
    },
    "pr-test-extra.yml": {
        "triggers": ("workflow_dispatch", "workflow_call"),
        "unsupported_jobs": (
            "extra-b-test-4-gpu-b200",
            "extra-b-test-8-gpu-h200",
            "extra-b-test-deepep-4-gpu-b200",
            "extra-b-test-deepep-8-gpu-h200",
        ),
    },
    "pr-test-jit-kernel.yml": {
        "triggers": ("workflow_call",),
        "unsupported_jobs": (
            "jit-kernel-multigpu-unit-test",
            "jit-kernel-b200-test",
        ),
    },
    "pr-test-sgl-kernel.yml": {
        "triggers": ("workflow_call",),
        "unsupported_jobs": ("sgl-kernel-b200-test",),
    },
    "pr-test-multimodal-gen.yml": {
        "triggers": ("workflow_call",),
        "unsupported_jobs": ("multimodal-gen-test-1-b200",),
    },
}


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.BaseLoader)


class TestLangSlangerCIHardwarePolicy(unittest.TestCase):
    def test_unavailable_hardware_workflows_are_manual_only(self):
        for workflow_name in MANUAL_ONLY_WORKFLOWS:
            with self.subTest(workflow=workflow_name):
                triggers = load_yaml(ROOT / ".github" / "workflows" / workflow_name)[
                    "on"
                ]
                self.assertNotIn("push", triggers)
                self.assertNotIn("pull_request", triggers)
                self.assertIn("workflow_dispatch", triggers)

    def test_arm_build_pins_the_current_repository_revision(self):
        workflow = load_yaml(ROOT / ".github" / "workflows" / "pr-test-arm64.yml")
        source_step = next(
            step
            for step in workflow["jobs"]["build-test"]["steps"]
            if step.get("name") == "Resolve source revision"
        )
        build_step = next(
            step
            for step in workflow["jobs"]["build-test"]["steps"]
            if step.get("name") == "Build container"
        )
        self.assertIn("github.repository", source_step["env"]["CURRENT_REPO"])
        self.assertIn("head.sha", source_step["env"]["PR_SHA"])
        self.assertIn("git rev-parse --verify HEAD", source_step["run"])
        self.assertIn(
            "steps.source.outputs.repo", build_step["env"]["SGLANG_SOURCE_REPO"]
        )
        self.assertIn(
            "steps.source.outputs.ref", build_step["env"]["SGLANG_SOURCE_REF"]
        )

        dockerfile = (ROOT / "docker" / "arm64.Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("sgl-project/sglang.git", dockerfile)
        self.assertNotIn("ARG SGLANG_REPO=", dockerfile)
        self.assertNotIn("ARG VER_SGLANG=", dockerfile)

    def test_pr_opt_in_gates_skip_without_dispatching_hardware(self):
        gate_path = ROOT / ".github" / "workflows" / "pr-gate.yml"
        gate = load_yaml(gate_path)["jobs"]["pr-gate"]
        for token in ("draft == false", "run-ci", "run-ci-extra", "event.action"):
            self.assertIn(token, gate["if"])
        self.assertNotIn("PR is draft. Blocking CI.", gate_path.read_text())

        unsafe = (
            "needs.call-gate.result == 'success' || "
            "needs.call-gate.result == 'skipped'"
        )
        for workflow_name in ("pr-test.yml", "pr-test-extra.yml", "pr-test-amd.yml"):
            with self.subTest(workflow=workflow_name):
                source = (ROOT / ".github" / "workflows" / workflow_name).read_text()
                self.assertNotIn(unsafe, source)

    def test_opt_in_workflows_recheck_ready_and_labeled_prs(self):
        for workflow_name in OPT_IN_PR_WORKFLOWS:
            with self.subTest(workflow=workflow_name):
                pull_request = load_yaml(
                    ROOT / ".github" / "workflows" / workflow_name
                )["on"]["pull_request"]
                self.assertIn("ready_for_review", pull_request["types"])
                self.assertIn("labeled", pull_request["types"])

    def test_opt_in_is_enforced_by_reusable_workflow_callers(self):
        for workflow_name, (gate_job, labels) in GATE_JOBS.items():
            with self.subTest(workflow=workflow_name):
                workflow = load_yaml(ROOT / ".github" / "workflows" / workflow_name)
                condition = workflow["jobs"][gate_job].get("if", "")
                self.assertIn("draft == false", condition)
                for label in labels:
                    self.assertIn(f"'{label}'", condition)

    def test_full_ci_is_explicitly_opt_in_everywhere(self):
        for workflow_name, policy in WORKFLOWS.items():
            with self.subTest(workflow=workflow_name):
                workflow = load_yaml(ROOT / ".github" / "workflows" / workflow_name)
                for trigger in policy["triggers"]:
                    input_config = workflow["on"][trigger]["inputs"][
                        "enable_upstream_full_ci"
                    ]
                    self.assertEqual(input_config["default"], "false")

                for job_name in policy["unsupported_jobs"]:
                    condition = workflow["jobs"][job_name].get("if", "")
                    self.assertIn(FULL_CI_EXPRESSION, condition)

    def test_supported_runner_labels_match_the_runpod_fleet(self):
        runner_configs = load_yaml(ROOT / "scripts" / "ci" / "runner_configs.yml")[
            "runner_configs"
        ]
        expected = {
            "1-gpu-small": "1-gpu-5090",
            "1-gpu-large": "1-gpu-h100",
            "2-gpu-large": "2-gpu-h100",
            "4-gpu-h100": "4-gpu-h100",
            "deepep-4-gpu-h100": "4-gpu-h100",
        }
        for runner_config, runner_label in expected.items():
            with self.subTest(runner_config=runner_config):
                self.assertEqual(runner_configs[runner_config]["runs_on"], runner_label)

    def test_main_workflow_forwards_policy_to_reusable_workflows(self):
        workflow = load_yaml(ROOT / ".github" / "workflows" / "pr-test.yml")
        jobs = workflow["jobs"]
        for job_name in (
            "call-pr-test-extra",
            "call-sgl-kernel-tests",
            "call-jit-kernel-tests",
            "call-multimodal-gen-tests",
        ):
            with self.subTest(job=job_name):
                forwarded = jobs[job_name]["with"]["enable_upstream_full_ci"]
                self.assertIn("inputs.enable_upstream_full_ci", forwarded)

        for workflow_name in ("pr-test.yml", "pr-test-extra.yml"):
            with self.subTest(native_build_workflow=workflow_name):
                build_job = load_yaml(ROOT / ".github" / "workflows" / workflow_name)[
                    "jobs"
                ]["sgl-kernel-build-wheels"]
                self.assertEqual(build_job["with"]["runs_on"], "1-gpu-h100")
                self.assertEqual(build_job["with"]["build_mode"], "native")
        self.assertIn("langslanger-hardware-coverage", jobs)


if __name__ == "__main__":
    unittest.main()
