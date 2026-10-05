"""Independent check that runner actuator output is sourced from the kernel."""

import numpy as np

from contracts import ActionProposal, Uncertainty
from robot.runner import CycleRunner, Modules
from safety import SafetyConfig, SafetyKernel
from state.telemetry import TelemetryLog


class FixedPolicy:
    version = "independent-fixed"

    def propose(self, state, prediction, rng):
        return ActionProposal("p", self.version, (20.0,), 1.0, Uncertainty())


class RecordingKernel(SafetyKernel):
    def __init__(self, *args, **kwargs):
        self.results = []
        super().__init__(*args, **kwargs)

    def check(self, *args, **kwargs):
        result = super().check(*args, **kwargs)
        self.results.append(result)
        return result


class RecordingEnvironment:
    version = "independent-env"

    def __init__(self):
        self.commands = []

    def observe(self, rng):
        from contracts import Observation

        return Observation("sensor", ("pos_x", "vel_x"), (0.0, 0.0), Uncertainty())

    def actuate(self, command, rng):
        from contracts import Outcome

        self.commands.append(command)
        return Outcome("act", True, 0.0, (), (), Uncertainty())


def test_rejected_proposal_actuates_only_the_kernel_safe_command(tmp_path):
    config = SafetyConfig(
        axis_names=("x",), action_low=(-1.0,), action_high=(1.0,),
        workspace_low=(-1.0,), workspace_high=(1.0,),
        max_command_age_s=1.0, watchdog_timeout_s=2.0,
    )
    now = 10.0
    env = RecordingEnvironment()
    with TelemetryLog(tmp_path / "runner.jsonl", clock=lambda: now) as log:
        kernel = RecordingKernel(config, log, "operator", clock=lambda: now)
        runner = CycleRunner(
            kernel, log, np.random.default_rng(0),
            modules=Modules(environment=env, system1=FixedPolicy()),
            clock=lambda: now,
        )
        (result,) = runner.run(1)

    assert len(kernel.results) == 1
    checked = kernel.results[0]
    assert checked.decision.verdict == "reject"
    assert env.commands == [checked.actuator_command] == [(0.0,)]
    assert result.actuated_command is checked.actuator_command
