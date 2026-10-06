"""Independent safety regression: declared actuator delay must affect approval."""

import numpy as np

from contracts import ActionProposal, Envelope, Uncertainty
from safety import SafetyKernel
from simulation.puck2d import Puck2D, Puck2DConfig
from state.telemetry import TelemetryLog


def test_kernel_does_not_approve_motion_that_exits_during_declared_brake_latency(tmp_path):
    # A one second actuator lag means braking cannot begin before the puck moves
    # beyond x=2. The current implementation predicts immediate braking instead.
    env = Puck2D(Puck2DConfig(actuator_tau=1.0))
    mhs = env.mhs()
    with TelemetryLog(tmp_path / "latency.jsonl") as log:
        kernel = SafetyKernel.from_mhs(mhs, log, "operator", clock=lambda: 0.0)
        command = Envelope(
            "latency-command", 0.0, "awareness", "safety", 1, "latency-cycle",
            ActionProposal("coast", "test", (0.0, 0.0), 1.0, Uncertainty()),
        )
        result = kernel.check(
            command, position=(1.95, 0.0), velocity=(0.5, 0.0), now=0.0
        )
    assert result.decision.verdict == "reject"
