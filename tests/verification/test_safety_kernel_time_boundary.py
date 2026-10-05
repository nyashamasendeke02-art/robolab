"""Independent checks for finite safety-kernel time inputs."""

import math

from contracts import ActionProposal, Envelope, Uncertainty
from safety import SafetyConfig, SafetyKernel
from state.telemetry import TelemetryLog


def test_nonfinite_kernel_time_never_approves_actuator_command(tmp_path):
    config = SafetyConfig(
        axis_names=("x",),
        action_low=(-1.0,),
        action_high=(1.0,),
        workspace_low=(-1.0,),
        workspace_high=(1.0,),
        max_command_age_s=1.0,
        watchdog_timeout_s=2.0,
    )
    command = Envelope(
        message_id="m",
        timestamp=10.0,
        source="test",
        destination="safety",
        cycle_id=1,
        correlation_id="c",
        payload=ActionProposal(
            proposal_id="p",
            policy_version="v1",
            action=(0.5,),
            confidence=1.0,
            uncertainty=Uncertainty(),
        ),
    )
    with TelemetryLog(tmp_path / "time.jsonl") as log:
        kernel = SafetyKernel(config, log, "operator", clock=lambda: 10.0)
        result = kernel.check(command, position=(0.0,), velocity=(0.0,), now=math.nan)

    assert result.decision.verdict != "approve"
    assert result.actuator_command == (0.0,)
