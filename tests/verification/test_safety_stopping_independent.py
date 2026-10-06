"""Independent stopping-distance checks for the Safety Kernel v1.1 contract."""

import numpy as np

from contracts import ActionProposal, Envelope, Uncertainty
from safety import SafetyConfig, SafetyKernel
from state.telemetry import TelemetryLog


def _command(action, timestamp=10.0):
    return Envelope(
        message_id="independent-check",
        timestamp=timestamp,
        source="awareness",
        destination="safety",
        cycle_id=1,
        correlation_id="independent",
        payload=ActionProposal(
            proposal_id="independent",
            policy_version="test",
            action=tuple(float(x) for x in action),
            confidence=1.0,
            uncertainty=Uncertainty(),
        ),
    )


def test_stopping_check_respects_asymmetric_axis_braking(tmp_path):
    """A small opposing limit must govern stopping distance, regardless of push limit."""
    cfg = SafetyConfig(
        axis_names=("x",), action_low=(-1.0,), action_high=(100.0,),
        workspace_low=(-100.0,), workspace_high=(10.0,),
        max_command_age_s=1.0, watchdog_timeout_s=2.0,
        control_dt_s=0.1, mass_kg=1.0,
    )
    with TelemetryLog(tmp_path / "independent.jsonl") as log:
        kernel = SafetyKernel(cfg, log, "key", clock=lambda: 10.0)
        # At 4 m/s, the actual available braking acceleration is only 1 m/s^2;
        # stopping distance is 8 m, while this state leaves under 8 m after one step.
        result = kernel.check(
            _command((0.0,)), position=(2.5,), velocity=(4.0,), now=10.0
        )
    assert result.decision.verdict == "reject"
    assert "stopping[x]" in result.decision.violated_constraints
    assert result.actuator_command == (-1.0,)


def test_seeded_random_actions_keep_puck_inside_workspace(tmp_path):
    """Exercise independent seeds and check the actuator-facing output each step."""
    from simulation.puck2d import Puck2D, Puck2DConfig

    dt = 0.02
    env_cfg = Puck2DConfig(
        dt=dt, start_pos=(0.0, 0.0), start_vel=(0.0, 0.0),
        goal=(5.0, 5.0), max_steps=200,
    )
    cfg = SafetyConfig(
        axis_names=("x", "y"), action_low=(-10.0, -10.0), action_high=(10.0, 10.0),
        workspace_low=(-1.0, -1.0), workspace_high=(1.0, 1.0),
        max_command_age_s=0.05, watchdog_timeout_s=0.2, control_dt_s=dt,
    )
    for seed in range(4):
        rng = np.random.default_rng(seed)
        env = Puck2D(env_cfg, seed=seed)
        with TelemetryLog(tmp_path / f"seed-{seed}.jsonl") as log:
            kernel = SafetyKernel(cfg, log, "key", clock=lambda: 10.0)
            for step in range(env_cfg.max_steps):
                result = kernel.check(
                    _command(rng.uniform(-50.0, 50.0, 2)),
                    position=tuple(env.pos), velocity=tuple(env.vel), now=10.0,
                )
                assert np.all(np.abs(result.actuator_command) <= 10.0)
                env.step(result.actuator_command)
                assert np.all(np.abs(env.pos) <= 1.0), (seed, step, env.pos)
