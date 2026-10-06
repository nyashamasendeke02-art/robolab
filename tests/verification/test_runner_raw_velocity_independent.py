"""Independent check that runner braking uses raw observed velocity."""

import numpy as np

from contracts import StateUpdate, Uncertainty
from robot.runner import CycleRunner, Modules, NullSystem1
from safety import SafetyKernel
from simulation.puck2d import Puck2D, Puck2DConfig
from state.telemetry import TelemetryLog


class LyingEstimator:
    version = "lying-estimator"

    def estimate(self, observation, rng):
        return StateUpdate(
            "physical", ("pos_x", "pos_y", "vel_x", "vel_y"),
            (0.0, 0.0, 0.0, 0.0), Uncertainty(), (),
        )


def test_abstention_brakes_from_observation_despite_false_estimate(tmp_path):
    env = Puck2D(Puck2DConfig(start_pos=(0.0, 0.0), start_vel=(1.0, -0.5), goal=(9.0, 9.0)))
    mhs = env.mhs()
    with TelemetryLog(tmp_path / "raw-velocity.jsonl") as log:
        kernel = SafetyKernel.from_mhs(mhs, log, "operator", clock=lambda: env.time)
        runner = CycleRunner(
            kernel, log, np.random.default_rng(0),
            Modules(environment=env, state_estimator=LyingEstimator(), system1=NullSystem1()),
            clock=lambda: env.time, mhs=mhs,
        )
        result = runner.step()

    assert result.kernel_source == "no_command"
    # Both components oppose the measured motion; the actuator limit means the
    # first cycle need not stop completely.
    assert result.actuated_command == (-10.0, 10.0)
    assert env.vel[0] < 1.0 and env.vel[1] > -0.5
