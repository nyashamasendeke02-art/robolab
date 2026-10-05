"""Independent safety invariants for the Safety Kernel."""

from contracts import ActionProposal, Envelope, Uncertainty
from safety import SafetyConfig, SafetyKernel
from state.telemetry import TelemetryLog


def _setup(tmp_path):
    config = SafetyConfig(
        axis_names=("x",), action_low=(-2.0,), action_high=(2.0,),
        workspace_low=(-1.0,), workspace_high=(1.0,),
        max_command_age_s=0.1, watchdog_timeout_s=0.2,
    )
    log = TelemetryLog(tmp_path / "safety.jsonl")
    return SafetyKernel(config, log, "operator"), log


def _command(timestamp=10.0, value=0.0):
    return Envelope(
        message_id="m", timestamp=timestamp, source="test", destination="safety",
        cycle_id=1, correlation_id="c",
        payload=ActionProposal(
            proposal_id="p", policy_version="v1", action=(value,),
            confidence=1.0, uncertainty=Uncertainty(),
        ),
    )


def test_estop_cannot_be_cleared_by_attribute_assignment(tmp_path):
    kernel, log = _setup(tmp_path)
    kernel.emergency_stop("test", now=10.0)
    # The latch contract says only the explicit operator reset API clears it.
    kernel._estop_reason = None
    result = kernel.check(_command(), position=(0.0,), velocity=(0.0,), now=10.0)
    log.close()
    assert result.decision.verdict == "emergency_stop"
    assert result.actuator_command == (0.0,)


def test_nominal_command_is_bounded_and_logged(tmp_path):
    kernel, log = _setup(tmp_path)
    result = kernel.check(_command(value=1.0), position=(0.0,), velocity=(0.0,), now=10.0)
    assert result.decision.verdict == "approve"
    assert result.actuator_command == (1.0,)
    assert len(log.records()) == 1
    log.close()
