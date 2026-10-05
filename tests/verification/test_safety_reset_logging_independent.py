"""Independent fail-safe checks for operator reset logging."""

from safety import SafetyConfig, SafetyKernel
from state.telemetry import TelemetryLog


def test_reset_does_not_clear_estop_when_reset_event_cannot_be_logged(tmp_path):
    config = SafetyConfig(
        axis_names=("x",),
        action_low=(-1.0,),
        action_high=(1.0,),
        workspace_low=(-1.0,),
        workspace_high=(1.0,),
        max_command_age_s=1.0,
        watchdog_timeout_s=2.0,
    )
    # Too small even for one telemetry record, so every append fails.
    with TelemetryLog(tmp_path / "capped.jsonl", max_bytes=1) as log:
        kernel = SafetyKernel(config, log, "operator", clock=lambda: 10.0)
        kernel.emergency_stop("test", now=10.0)
        assert kernel.estopped

        kernel.reset_emergency_stop("operator", now=10.1)

        assert kernel.estopped, "an unlogged reset must not release the safety latch"


def test_operator_reset_clears_latch_when_event_is_logged(tmp_path):
    config = SafetyConfig(
        axis_names=("x",), action_low=(-1.0,), action_high=(1.0,),
        workspace_low=(-1.0,), workspace_high=(1.0,),
        max_command_age_s=1.0, watchdog_timeout_s=2.0,
    )
    with TelemetryLog(tmp_path / "normal.jsonl") as log:
        kernel = SafetyKernel(config, log, "operator", clock=lambda: 10.0)
        kernel.emergency_stop("test", now=10.0)
        kernel.reset_emergency_stop("operator", now=10.1)
        assert not kernel.estopped
        assert log.records()[-1].decision == "estop_reset"
