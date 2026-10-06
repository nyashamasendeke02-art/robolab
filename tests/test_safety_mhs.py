"""Safety independence and braking through the runner (G1-5, REQ-SAFE+, ADR-003; ENG-0012).

AC4: a state estimator reporting a false in-bounds state cannot get a
     workspace-exiting command approved (fault injection).
AC5: with the body moving, abstain / module-failure / watchdog / e-stop cycles
     actuate a braking safe action and Puck2D stops inside the workspace.
Plus the v1.2 mass interval (APR-0003: ``mass_kg`` is an upper bound).
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts import ActionProposal, Envelope, StateUpdate, Uncertainty
from robot.runner import CycleRunner, Modules, NullSystem1
from safety import SafetyConfig, SafetyKernel
from safety.kernel import _stopping_distances
from simulation.harness import HarnessConfig
from simulation.puck2d import Puck2D, Puck2DConfig
from state.telemetry import TelemetryLog

KEY = "operator"
WS = 2.0


class PushPolicy:
    """Always pushes +x at the force limit (towards the wall)."""

    version = "push-0"

    def __init__(self):
        self.n = 0

    def propose(self, state, prediction, rng):
        self.n += 1
        return ActionProposal(f"push-{self.n}", self.version, (10.0, 0.0), 1.0, Uncertainty())


class LyingEstimator:
    """Faulty / learned estimator: always reports the body at rest at the origin."""

    version = "liar-0"

    def estimate(self, observation, rng):
        return StateUpdate("physical", ("pos_x", "pos_y", "vel_x", "vel_y"),
                           (0.0, 0.0, 0.0, 0.0), Uncertainty(), ())


class RaisingS1:
    def propose(self, state, prediction, rng):
        raise RuntimeError("policy crashed")


def _runner(env, log, modules, mhs=None, limit_mode="reject"):
    mhs = mhs or HarnessConfig().mhs(env)
    clock = lambda: env.time  # noqa: E731  simulation time, advances with each actuation
    kernel = SafetyKernel.from_mhs(mhs, log, KEY, clock=clock, limit_mode=limit_mode)
    return CycleRunner(kernel, log, np.random.default_rng(0),
                       Modules(environment=env, **modules), clock=clock, mhs=mhs), kernel


# -- AC4: fault injection ----------------------------------------------------------------------


@pytest.mark.parametrize("limit_mode", ["reject", "clamp"])
def test_lying_estimator_cannot_get_a_workspace_exit_approved(tmp_path, limit_mode):
    env = Puck2D(Puck2DConfig(start_pos=(1.95, 0.0), start_vel=(1.0, 0.0), goal=(9.0, 9.0),
                              max_steps=1000))
    with TelemetryLog(tmp_path / "f.jsonl") as log:
        runner, kernel = _runner(env, log, {"state_estimator": LyingEstimator(),
                                            "system1": PushPolicy()}, limit_mode=limit_mode)
        # Non-vacuous: given the lie, the kernel would approve the push.
        with TelemetryLog(tmp_path / "probe.jsonl") as plog:
            probe = SafetyKernel(kernel.config, plog, KEY, clock=lambda: 0.0)
            cmd = Envelope("m", 0.0, "awareness", "safety", 1, "c",
                           ActionProposal("p", "t", (10.0, 0.0), 1.0, Uncertainty()))
            assert probe.check(cmd, position=(0.0, 0.0), velocity=(0.0, 0.0), now=0.0) \
                .decision.approved_action == (10.0, 0.0)
        results = []
        for _ in range(300):
            results.append(runner.step())
            assert abs(env.pos[0]) <= WS and abs(env.pos[1]) <= WS, env.pos
    first = results[0].kernel_result.decision
    # The push from x=1.95 at 1 m/s would leave the workspace: never approved as is.
    assert first.approved_action != (10.0, 0.0)
    assert set(first.violated_constraints) & {"stopping[x]", "workspace[x]"}
    assert all(r.kernel_source == "check" for r in results)
    # Pushes were approved again once the true state allowed them, and the body
    # still never left the workspace (asserted every cycle above).
    assert any(r.kernel_result.decision.approved_action == (10.0, 0.0) for r in results[1:])


def test_lying_velocity_cannot_hide_a_fast_body(tmp_path):
    # Fast body heading out, estimator says "at rest in the middle"; the kernel brakes.
    # y = 1.15 at 4 m/s: braking now stops at ~1.91; one more period without
    # braking would not leave room, so the push (no y braking) must be refused.
    env = Puck2D(Puck2DConfig(start_pos=(0.0, 1.15), start_vel=(0.0, 4.0), goal=(9.0, 9.0)))
    with TelemetryLog(tmp_path / "f.jsonl") as log:
        runner, _ = _runner(env, log, {"state_estimator": LyingEstimator(), "system1": PushPolicy()})
        res = runner.run(100)
    assert res[0].actuated_command[1] == -10.0  # braking against the observed +y motion
    assert env.pos[1] <= WS and env.vel[1] == pytest.approx(0.0, abs=1e-9)


# -- AC5: braking on every cycle without an approved command -------------------------------------


def _estop_runner(env, log, mhs):
    runner, kernel = _runner(env, log, {"system1": PushPolicy()}, mhs=mhs)
    kernel.emergency_stop("test", now=0.0)
    return runner


SCENARIOS = {
    "abstain": lambda env, log, mhs: _runner(env, log, {"system1": NullSystem1()}, mhs=mhs)[0],
    "module_failure": lambda env, log, mhs: _runner(env, log, {"system1": RaisingS1()}, mhs=mhs)[0],
    "estop": _estop_runner,
}
MASS_SCHEDULES = {
    "exact": ((), None),
    "lighter": (({"step": 0, "mass": 0.5},), None),  # true mass at the lower bound
    "heavier": (({"step": 0, "mass": 2.0},), None),  # true mass at the upper bound
    "fleet": (({"step": 0, "mass": 1.3},), (0.5, 2.0)),
}


@pytest.mark.parametrize("schedule", list(MASS_SCHEDULES))
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_moving_body_brakes_and_stops_inside_workspace(tmp_path, scenario, schedule):
    changes, bounds = MASS_SCHEDULES[schedule]
    cfg = Puck2DConfig(start_pos=(1.5, -1.5), start_vel=(2.0, -1.5), goal=(9.0, 9.0),
                       mass_changes=changes, max_steps=1000)
    env = Puck2D(cfg)
    mhs = HarnessConfig().mhs(env, mass_bounds=bounds)
    # Without braking (zero force, the pre-G1-5 behaviour) the puck coasts out.
    assert abs(cfg.start_pos[0] + cfg.start_vel[0] * 100 * cfg.dt) > WS
    with TelemetryLog(tmp_path / "b.jsonl") as log:
        runner = SCENARIOS[scenario](env, log, mhs)
        results = []
        for _ in range(100):
            v_before = env.vel.copy()
            res = runner.step()
            results.append(res)
            u = np.array(res.actuated_command)
            assert np.all(u * v_before <= 0.0), (u, v_before)  # opposes the motion
            # Never reversed (beyond floating-point rounding of an exact stop).
            assert np.all(env.vel * np.sign(v_before) >= -1e-12), (env.vel, v_before)
            assert np.all(np.abs(env.pos) <= WS)
    assert np.all(np.abs(env.vel) <= 1e-9), env.vel
    assert np.any(np.abs(results[0].actuated_command) == 10.0)  # braking, not zero force
    sources = {r.kernel_source for r in results}
    verdicts = {r.kernel_result.decision.verdict for r in results}
    if scenario == "estop":
        assert verdicts == {"emergency_stop"}
    else:
        assert verdicts == {"reject"}
        assert sources == {"no_command", "tick"}  # abstain, then the watchdog fires
        assert any("watchdog_timeout" in r.kernel_result.decision.violated_constraints
                   for r in results)


def test_actuator_failure_estop_brakes(tmp_path):
    class FlakyPuck(Puck2D):
        fail = True

        def actuate(self, command, rng):
            if self.fail:
                self.fail = False
                raise IOError("bus error")
            return super().actuate(command, rng)

    env = FlakyPuck(Puck2DConfig(start_pos=(0.0, 0.0), start_vel=(1.0, 0.0)))
    with TelemetryLog(tmp_path / "a.jsonl") as log:
        runner, kernel = _runner(env, log, {"system1": PushPolicy()})
        runner.run(1)
        assert kernel.estopped
        res = runner.run(30)
    assert res[0].actuated_command == (-10.0, 0.0)
    assert env.vel[0] == pytest.approx(0.0, abs=1e-9)


# -- v1.2 mass interval ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_stopping_distance_bound_covers_any_mass_in_the_interval(seed):
    rng = np.random.default_rng(seed)
    for _ in range(400):
        dt = float(rng.choice([0.01, 0.02, 0.1]))
        limit = float(rng.uniform(1.0, 20.0))
        m_hi = float(rng.uniform(0.2, 5.0))
        m_lo = m_hi * float(rng.uniform(0.1, 1.0))
        m = float(rng.uniform(m_lo, m_hi))
        v0 = float(rng.uniform(0.0, 5.0))
        d_zoh, d_si = _stopping_distances(v0, limit / m_hi, dt, 1.0 - m_lo / m_hi)
        v, x_zoh, x_si = v0, 0.0, 0.0
        for _ in range(100000):
            if v <= 1e-15:
                break
            u = max(-limit, -m_lo * v / dt)  # the kernel's braking safe action
            v_new = v + u / m * dt
            assert v_new >= -1e-12  # never reversed
            x_zoh += 0.5 * (v + v_new) * dt
            x_si += v_new * dt
            v = v_new
        assert x_zoh <= d_zoh * (1 + 1e-9) + 1e-12
        assert x_si <= d_si * (1 + 1e-9) + 1e-12


def _push_policy(rng):
    while True:
        u = rng.uniform(-50.0, 50.0, 2)
        for _ in range(int(rng.integers(1, 40))):
            yield u


@pytest.mark.parametrize("limit_mode", ["clamp", "reject"])
def test_puck2d_with_mass_changes_never_leaves_workspace(tmp_path, limit_mode):
    for seed in range(12):
        rng = np.random.default_rng(500 + seed)
        changes = [{"step": int(rng.integers(0, 200)), "mass": float(rng.uniform(0.5, 2.0))}
                   for _ in range(2)]
        if changes[0]["step"] == changes[1]["step"]:
            changes = changes[:1]
        env = Puck2D(Puck2DConfig(mass=float(rng.uniform(0.5, 2.0)), mass_changes=changes,
                                  start_pos=tuple(rng.uniform(-0.9, 0.9, 2)), goal=(9.0, 9.0),
                                  max_steps=300), seed=seed)
        mhs = HarnessConfig(workspace_low=(-1.0, -1.0), workspace_high=(1.0, 1.0)).mhs(
            env, mass_bounds=(0.5, 2.0))
        with TelemetryLog(tmp_path / f"m{seed}-{limit_mode}.jsonl") as log:
            k = SafetyKernel.from_mhs(mhs, log, KEY, clock=lambda: env.time, limit_mode=limit_mode)
            policy = _push_policy(rng)
            done = False
            while not done:
                t = env.time
                cmd = Envelope(f"m{env.steps}", t, "awareness", "safety", env.steps, "c",
                               ActionProposal("p", "t", tuple(float(x) for x in next(policy)),
                                              1.0, Uncertainty()))
                r = k.check(cmd, position=tuple(env.pos), velocity=tuple(env.vel), now=t)
                _, _, term, trunc, _ = env.step(r.actuator_command)
                done = term or trunc
                assert np.all(np.abs(env.pos) <= 1.0), (seed, env.steps, env.pos, env.vel)


def test_speed_limit_and_brake_decel_cap(tmp_path):
    base = dict(axis_names=("x",), action_low=(-10.0,), action_high=(10.0,),
                workspace_low=(-10.0,), workspace_high=(10.0,), max_command_age_s=1.0,
                watchdog_timeout_s=1.0, control_dt_s=0.1)
    cmd = Envelope("m", 0.0, "a", "safety", 1, "c",
                   ActionProposal("p", "t", (10.0,), 1.0, Uncertainty()))
    with TelemetryLog(tmp_path / "s.jsonl") as log:
        fast = SafetyKernel(SafetyConfig(**base, max_speed=(1.5,)), log, KEY, clock=lambda: 0.0)
        r = fast.check(cmd, position=(0.0,), velocity=(1.0,), now=0.0)  # v' = 2 > 1.5
        assert r.decision.verdict == "reject" and r.decision.violated_constraints == ("speed[x]",)
        assert fast.check(cmd, position=(0.0,), velocity=(0.0,), now=0.0).decision.verdict == "approve"
        # brake_decel caps the braking deceleration: 1 m/s^2 needs 2 m from 2 m/s.
        capped = SafetyKernel(SafetyConfig(**{**base, "workspace_high": (1.5,)}, brake_decel=(1.0,)),
                              log, KEY, clock=lambda: 0.0)
        free = SafetyKernel(SafetyConfig(**{**base, "workspace_high": (1.5,)}), log, KEY, clock=lambda: 0.0)
        assert capped.check(cmd, position=(0.0,), velocity=(1.0,), now=0.0).decision.verdict == "reject"
        assert free.check(cmd, position=(0.0,), velocity=(1.0,), now=0.0).decision.verdict == "approve"
        with pytest.raises(ValueError):
            SafetyConfig(**base, mass_kg=1.0, mass_lower_kg=2.0)


def test_no_command_does_not_reset_the_watchdog(tmp_path):
    cfg = SafetyConfig(("x",), (-1.0,), (1.0,), (-1.0,), (1.0,), 0.05, 0.1)
    with TelemetryLog(tmp_path / "w.jsonl") as log:
        k = SafetyKernel(cfg, log, KEY, clock=lambda: 0.0)
        for t in (0.05, 0.09):
            r = k.no_command(now=t, velocity=(0.5,))
            assert r.decision.violated_constraints == ("no_command",)
            assert r.actuator_command == (-1.0,)
        assert k.tick(now=0.11).decision.violated_constraints == ("watchdog_timeout",)
        assert [rec.decision for rec in log.records()] == ["reject", "reject", "reject"]
