"""Tests for the v1.1 stopping-distance check and braking safe action (ENG-0010, G1-3).

Gate 1, REQ-SAFE, REQ-SAFE+, ADR-003.
"""

import math

import numpy as np
import pytest

from contracts import ActionProposal, Envelope, Uncertainty
from safety import KERNEL_VERSION, SafetyConfig, SafetyKernel
from simulation.puck2d import Puck2D, Puck2DConfig
from state.telemetry import TelemetryLog

KEY = "operator-secret"
T0 = 100.0


def _cmd(action, ts=T0, pid="p1"):
    return Envelope(
        message_id=f"m-{pid}",
        timestamp=ts,
        source="awareness",
        destination="safety",
        cycle_id=1,
        correlation_id="c1",
        payload=ActionProposal(
            proposal_id=pid,
            policy_version="test",
            action=tuple(float(a) for a in action),
            confidence=0.9,
            uncertainty=Uncertainty(),
        ),
    )


def _kernel(log, n=1, limit=10.0, ws=(-10.0, 10.0), dt=0.01, **kw):
    cfg = SafetyConfig(
        axis_names=("x", "y")[:n],
        action_low=(-limit,) * n,
        action_high=(limit,) * n,
        workspace_low=(ws[0],) * n,
        workspace_high=(ws[1],) * n,
        max_command_age_s=0.05,
        watchdog_timeout_s=0.2,
        control_dt_s=dt,
        **kw,
    )
    return SafetyKernel(cfg, log, KEY, clock=lambda: T0)


@pytest.fixture
def tl(tmp_path):
    with TelemetryLog(tmp_path / "safety.jsonl") as log:
        yield log


def test_kernel_version_bumped():
    assert KERNEL_VERSION == "safety-kernel-1.1.0"


# AC 1: the review case ------------------------------------------------------


@pytest.mark.parametrize("action", [0.0, 10.0, -10.0])
def test_review_case_is_rejected_and_brakes(tl, action):
    # x = 9, v = 50 m/s, wall at 10, dt = 0.01: v1 approved (p' = 9.5); it cannot stop.
    k = _kernel(tl)
    r = k.check(_cmd((action,)), position=(9.0,), velocity=(50.0,), now=T0)
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("stopping[x]",)
    assert r.actuator_command == (-10.0,)  # braking at the action limit, not zero force


def test_review_case_in_clamp_mode_never_approves_the_command(tl):
    k = _kernel(tl, limit_mode="clamp")
    r = k.check(_cmd((100.0,)), position=(9.0,), velocity=(50.0,), now=T0)
    # Even full braking cannot stop in time, so the kernel rejects and brakes.
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("action_limit[x]", "stopping[x]")
    assert r.actuator_command == (-10.0,)


def test_clamp_mode_replaces_unstoppable_command_with_braking(tl):
    # dt 0.1, x = 0.7, v = 1: pushing on (+10) ends at p' = 0.85 with v' = 2 ->
    # needs 0.2 m to stop (> 0.15 m left). Braking (-10) stops at 0.75.
    k = _kernel(tl, ws=(-1.0, 1.0), dt=0.1, limit_mode="clamp")
    r = k.check(_cmd((10.0,)), position=(0.7,), velocity=(1.0,), now=T0)
    assert r.decision.verdict == "approve"
    assert r.decision.approved_action == (-10.0,)
    assert r.decision.violated_constraints == ("stopping[x]",)
    assert r.actuator_command == (-10.0,)
    rec = tl.records()[-1]
    assert rec.decision == "approve" and "braking" in rec.reason


def test_reject_mode_rejects_unstoppable_command(tl):
    k = _kernel(tl, ws=(-1.0, 1.0), dt=0.1)
    r = k.check(_cmd((10.0,)), position=(0.7,), velocity=(1.0,), now=T0)
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("stopping[x]",)
    assert r.actuator_command == (-10.0,)


def test_lower_bound_is_checked_too(tl):
    k = _kernel(tl, ws=(-1.0, 1.0), dt=0.1)
    r = k.check(_cmd((-10.0,)), position=(-0.7,), velocity=(-1.0,), now=T0)
    assert r.decision.violated_constraints == ("stopping[x]",)
    assert r.actuator_command == (10.0,)


def test_axis_that_cannot_brake_must_not_move_towards_bound(tl):
    # action_low = 0: no braking force against +x motion.
    cfg = SafetyConfig(
        axis_names=("x",), action_low=(0.0,), action_high=(10.0,),
        workspace_low=(-1.0,), workspace_high=(1.0,),
        max_command_age_s=0.05, watchdog_timeout_s=0.2, control_dt_s=0.1,
    )
    k = SafetyKernel(cfg, tl, KEY, clock=lambda: T0)
    r = k.check(_cmd((0.0,)), position=(0.0,), velocity=(0.01,), now=T0)
    assert r.decision.violated_constraints == ("stopping[x]",)
    assert r.actuator_command == (0.0,)  # braking clipped to the action limits


# AC 3: commands that can still stop are approved unchanged -------------------


def test_stoppable_commands_are_approved_unchanged(tl):
    k = _kernel(tl, ws=(-1.0, 1.0), dt=0.1)
    for pos, vel, u in [
        ((0.0,), (0.0,), (10.0,)),
        ((0.5,), (1.0,), (10.0,)),   # stops at 0.85 (ZOH) / 0.8 (semi-implicit)
        ((0.9,), (0.0,), (10.0,)),   # exactly tight: stops at 1.0
        ((0.9,), (-3.0,), (10.0,)),  # moving away from the near wall
        ((-0.5,), (-1.0,), (3.0,)),
    ]:
        r = k.check(_cmd(u), position=pos, velocity=vel, now=T0)
        assert r.decision.verdict == "approve", (pos, vel, u, r.reason)
        assert r.decision.approved_action == u
        assert r.decision.violated_constraints == ()
        assert r.actuator_command == u


def test_random_commands_with_room_to_stop_are_approved_unchanged(tl):
    # Independent check of the documented rule with a margin of 2 * a_max * dt^2
    # (covers the discrete-braking terms): such commands must not be touched.
    k = _kernel(tl, n=2, ws=(-1.0, 1.0), dt=0.02)
    a, dt = 10.0, 0.02
    rng = np.random.default_rng(3)
    n_checked = 0
    for _ in range(3000):
        pos = rng.uniform(-1.0, 1.0, 2)
        vel = rng.uniform(-3.0, 3.0, 2)
        u = rng.uniform(-10.0, 10.0, 2)
        p1 = pos + vel * dt + 0.5 * u * dt * dt
        v1 = vel + u * dt
        stop = p1 + np.sign(v1) * v1 * v1 / (2 * a)
        if np.all(np.abs(stop) <= 1.0 - 2 * a * dt * dt) and np.all(np.abs(p1) <= 1.0):
            n_checked += 1
            r = k.check(_cmd(u), position=tuple(pos), velocity=tuple(vel), now=T0)
            assert r.decision.verdict == "approve", r.reason
            assert r.actuator_command == tuple(float(x) for x in u)
    assert n_checked > 500


# Braking safe action ---------------------------------------------------------


def test_braking_safe_action_stops_without_reversing(tl):
    k = _kernel(tl, n=2, dt=0.1, ws=(-100.0, 100.0))
    k.emergency_stop("test", now=T0)
    # fast: at the limit; slow: exactly -m v / dt; stopped: zero force.
    r = k.check(_cmd((1.0, 1.0)), position=(0.0, 0.0), velocity=(5.0, 0.3), now=T0)
    assert r.decision.verdict == "emergency_stop"
    assert r.actuator_command == pytest.approx((-10.0, -3.0))
    r = k.check(_cmd((1.0, 1.0)), position=(0.0, 0.0), velocity=(0.0, -0.2), now=T0)
    assert r.actuator_command == pytest.approx((0.0, 2.0))


def test_tick_and_estop_brake_when_given_velocity(tl):
    k = _kernel(tl, n=2, dt=0.1)
    assert k.tick(now=T0 + 1.0, velocity=(2.0, -2.0)).actuator_command == (-10.0, 10.0)
    assert k.tick(now=T0 + 1.0).actuator_command == (0.0, 0.0)  # unknown velocity
    assert k.tick(now=T0 + 1.0, velocity=(math.nan, 0.0)).actuator_command == (0.0, 0.0)
    r = k.emergency_stop("test", now=T0, velocity=(0.0, 1.0))
    assert r.actuator_command == (0.0, -10.0)


def test_safe_action_on_stale_and_malformed_commands_brakes(tl):
    k = _kernel(tl, dt=0.1)
    r = k.check(_cmd((0.0,), ts=T0 - 1.0), position=(0.0,), velocity=(3.0,), now=T0)
    assert r.decision.violated_constraints == ("stale_command",)
    assert r.actuator_command == (-10.0,)
    r = k.check("garbage", position=(0.0,), velocity=(-0.5,), now=T0)
    assert r.decision.violated_constraints == ("malformed_command",)
    assert r.actuator_command == (5.0,)


# AC 2: property test in Puck2D -------------------------------------------------


def _random_policy(rng):
    """High-speed commands: random pushes up to 5x the force limit, held for a
    random number of steps, biased to drive the puck into a wall."""
    while True:
        if rng.uniform() < 0.7:
            axis = rng.integers(2)
            u = np.zeros(2)
            u[axis] = rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 50.0)
            u[1 - axis] = rng.uniform(-50.0, 50.0)
        else:
            u = rng.uniform(-50.0, 50.0, 2)
        for _ in range(int(rng.integers(1, 60))):
            yield u


@pytest.mark.parametrize("limit_mode", ["clamp", "reject"])
@pytest.mark.parametrize("dt", [0.02, 0.05])
def test_puck2d_random_high_speed_commands_never_leave_workspace(tmp_path, limit_mode, dt):
    ws = 1.0
    n_steps = 0
    max_speed = 0.0
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        env_cfg = Puck2DConfig(
            dt=dt,
            mass=1.0,
            friction=float(rng.choice([0.0, rng.uniform(0.0, 0.3)])),
            damping=float(rng.choice([0.0, rng.uniform(0.0, 0.5)])),
            start_pos=tuple(rng.uniform(-0.9, 0.9, 2)),
            goal=(5.0, 5.0),  # unreachable: episodes run to max_steps
            max_steps=300,
        )
        env = Puck2D(env_cfg, seed=seed)
        clock = [T0]
        with TelemetryLog(tmp_path / f"{limit_mode}-{dt}-{seed}.jsonl") as log:
            cfg = SafetyConfig(
                axis_names=("x", "y"), action_low=env_cfg.force_low,
                action_high=env_cfg.force_high, workspace_low=(-ws, -ws),
                workspace_high=(ws, ws), max_command_age_s=0.05, watchdog_timeout_s=0.2,
                limit_mode=limit_mode, mass_kg=env_cfg.mass, control_dt_s=dt,
            )
            k = SafetyKernel(cfg, log, KEY, clock=lambda: clock[0])
            policy = _random_policy(rng)
            done = False
            while not done:
                t = clock[0]
                r = k.check(
                    _cmd(next(policy), ts=t, pid=f"s{seed}-{env.steps}"),
                    position=tuple(env.pos), velocity=tuple(env.vel), now=t,
                )
                assert not k.estopped
                _, _, term, trunc, _ = env.step(r.actuator_command)
                done = term or trunc
                clock[0] = t + dt
                n_steps += 1
                max_speed = max(max_speed, float(np.max(np.abs(env.vel))))
                assert np.all(np.abs(env.pos) <= ws), (seed, env.steps, env.pos, env.vel)
    assert n_steps == 40 * 300
    assert max_speed > 2.0  # the commands really did drive the puck fast
