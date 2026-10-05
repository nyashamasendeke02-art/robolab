"""Tests for src/simulation/puck2d.py (Gate 1, REQ-SIM; ENG-0006)."""

import math
import time

import numpy as np
import pytest

from robot.runner import CycleRunner, Environment, Modules
from safety import SafetyConfig, SafetyKernel
from simulation.puck2d import (
    EpisodeOver,
    FrictionPatch,
    Impulse,
    MassChange,
    Obstacle,
    Puck2D,
    Puck2DConfig,
)
from state.telemetry import TelemetryLog

DT = 0.02


def _env(seed=0, **kw):
    kw.setdefault("goal", (100.0, 100.0))  # out of reach unless a test sets it
    return Puck2D(Puck2DConfig(**kw), seed=seed)


def _rollout(env, actions):
    out = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        out.append((obs.tobytes(), r, term, trunc, info["pos"], info["vel"]))
        if term or trunc:
            break
    return out


# -- 1. determinism -------------------------------------------------------------


def test_same_seed_same_actions_bit_identical():
    kw = dict(
        damping=0.3, friction=0.05, actuator_tau=0.05, start_pos_std=0.1,
        pos_noise_std=0.01, vel_noise_std=0.02, max_steps=300,
        impulses=[{"step": 10, "impulse": (0.5, -0.2)}],
        mass_changes=[{"step": 50, "mass": 2.0}],
        friction_patches=[{"friction": 0.3, "low": (0.0, -1.0), "high": (0.5, 1.0)}],
    )
    actions = np.random.default_rng(123).uniform(-15, 15, (300, 2))
    a = _rollout(_env(seed=7, **kw), actions)
    b = _rollout(_env(seed=7, **kw), actions)
    assert a == b
    env = _env(seed=7, **kw)
    _rollout(env, actions[:40])
    env.reset(7)  # reset reproduces the trajectory on the same instance
    assert _rollout(env, actions) == a
    c = _rollout(_env(seed=8, **kw), actions)
    assert c != a  # the seed matters (noise)


def test_no_global_rng_used():
    env1, env2 = _env(seed=3, pos_noise_std=0.1), _env(seed=3, pos_noise_std=0.1)
    o1 = env1.step((1.0, 0.0))[0]
    np.random.seed(999)
    np.random.normal(size=100)
    o2 = env2.step((1.0, 0.0))[0]
    assert o1.tobytes() == o2.tobytes()


# -- 2. physics sanity ----------------------------------------------------------


def test_momentum_conserved_without_force_or_friction():
    env = _env(start_vel=(0.7, -0.3), mass=2.5)
    p0 = np.array(env.pos)
    for k in range(1, 201):
        _, _, _, _, info = env.step((0.0, 0.0))
        assert info["vel"] == [0.7, -0.3]
        assert env.mass * env.vel[0] == 2.5 * 0.7
    np.testing.assert_allclose(env.pos, p0 + np.array([0.7, -0.3]) * 200 * DT, rtol=1e-12)


@pytest.mark.parametrize("friction", [0.0, 0.05])
def test_damping_speed_decays_monotonically(friction):
    env = _env(start_vel=(2.0, 1.0), damping=0.8, friction=friction)
    speeds = [math.hypot(*env.vel)]
    for _ in range(300):
        env.step((0.0, 0.0))
        speeds.append(math.hypot(*env.vel))
    assert all(b < a or a == b == 0.0 for a, b in zip(speeds, speeds[1:]))
    assert speeds[-1] < speeds[0] * 0.05
    if friction == 0.0:  # pure viscous: exact geometric decay v *= (1 - c dt / m)
        np.testing.assert_allclose(speeds[1] / speeds[0], 1 - 0.8 * DT / 1.0, rtol=1e-12)


@pytest.mark.parametrize("mass", [0.5, 1.0, 3.0])
def test_acceleration_at_force_limit_is_F_over_m(mass):
    env = _env(mass=mass, force_low=(-4.0, -6.0), force_high=(4.0, 6.0))
    _, _, _, _, info = env.step((100.0, -100.0))  # clipped to the limits
    assert info["action_clipped"] is True
    assert info["applied_force"] == [4.0, -6.0]
    np.testing.assert_allclose(np.array(info["vel"]) / DT, [4.0 / mass, -6.0 / mass], rtol=1e-12)
    for _ in range(10):
        v0 = env.vel.copy()
        env.step((4.0, -6.0))
        np.testing.assert_allclose((env.vel - v0) / DT, [4.0 / mass, -6.0 / mass], rtol=1e-9)


def test_semi_implicit_euler_position_uses_new_velocity():
    env = _env()
    env.step((1.0, 0.0))
    assert env.vel[0] == 1.0 * DT
    assert env.pos[0] == env.vel[0] * DT  # explicit Euler would leave the position at 0


def test_coulomb_stiction_and_sliding():
    mu, g = 0.2, 9.81
    env = _env(friction=mu)
    env.step((0.9 * mu * g, 0.0))  # below mu m g: stays at rest
    assert env.vel.tolist() == [0.0, 0.0]
    env.step((2.0 * mu * g, 0.0))  # above: a = (F - mu m g) / m
    np.testing.assert_allclose(env.vel[0], mu * g * DT, rtol=1e-12)
    env2 = _env(friction=mu, start_vel=(0.1, 0.0))
    for _ in range(10):
        env2.step((0.0, 0.0))
        assert env2.vel[0] >= 0.0  # friction never reverses the velocity
    assert env2.vel[0] == 0.0


def test_actuator_lag_first_order():
    tau = 0.1
    env = _env(actuator_tau=tau)
    alpha = 1 - math.exp(-DT / tau)
    f = 0.0
    for _ in range(20):
        _, _, _, _, info = env.step((5.0, 0.0))
        f += alpha * (5.0 - f)
        np.testing.assert_allclose(info["applied_force"][0], f, rtol=1e-12)
    assert 0 < env.force[0] < 5.0
    np.testing.assert_allclose(env.force[0], 5.0 * (1 - math.exp(-20 * DT / tau)), rtol=1e-9)


def test_sensor_noise_statistics_and_ground_truth_in_info():
    env = _env(pos_noise_std=0.1, vel_noise_std=0.5, max_steps=5000)
    errs = []
    for _ in range(4000):
        obs, _, _, _, info = env.step((0.0, 0.0))
        errs.append(obs - np.array(info["pos"] + info["vel"]))
    std = np.std(errs, axis=0)
    np.testing.assert_allclose(std, [0.1, 0.1, 0.5, 0.5], rtol=0.06)
    clean = _env()
    obs, _, _, _, info = clean.step((1.0, 2.0))
    assert obs.tolist() == info["pos"] + info["vel"]


# -- 3. disturbances --------------------------------------------------------------


def test_impulse_applied_at_configured_step():
    env = _env(mass=2.0, impulses=[Impulse(5, (1.0, -0.5))])
    for k in range(10):
        _, _, _, _, info = env.step((0.0, 0.0))
        kinds = [d["type"] for d in info["active_disturbances"]]
        if k < 5:
            assert info["vel"] == [0.0, 0.0] and kinds == []
        else:
            assert info["vel"] == [0.5, -0.25]
            assert kinds == (["impulse"] if k == 5 else [])


def test_mass_change_at_configured_step():
    env = _env(mass=1.0, mass_changes=[MassChange(3, 4.0)])
    for k in range(6):
        v0 = env.vel.copy()
        _, _, _, _, info = env.step((2.0, 0.0))
        acc = (env.vel[0] - v0[0]) / DT
        expected_mass = 1.0 if k < 3 else 4.0
        assert info["mass"] == expected_mass
        np.testing.assert_allclose(acc, 2.0 / expected_mass, rtol=1e-9)
        assert ("mass_change" in [d["type"] for d in info["active_disturbances"]]) == (k == 3)


@pytest.mark.parametrize(
    "patch",
    [
        FrictionPatch(0.5, low=(0.1, -1.0), high=(0.3, 1.0)),
        FrictionPatch(0.5, center=(0.2, 0.0), radius=0.1),
    ],
)
def test_friction_patch_applies_only_inside_region(patch):
    env = _env(start_vel=(1.0, 0.0), friction=0.0, friction_patches=[patch])
    for _ in range(40):
        x0, v0 = float(env.pos[0]), float(env.vel[0])
        _, _, _, _, info = env.step((0.0, 0.0))
        inside = 0.1 <= x0 <= 0.3
        assert info["friction"] == (0.5 if inside else 0.0)
        assert bool(info["active_disturbances"]) == inside
        if inside:
            np.testing.assert_allclose(v0 - env.vel[0], min(v0, 0.5 * 9.81 * DT), rtol=1e-9)
        else:
            assert env.vel[0] == v0
        if env.vel[0] == 0.0:
            break
    assert env.pos[0] > 0.1  # it did reach and slow down inside the patch


# -- 4. termination ---------------------------------------------------------------


def test_goal_termination_and_reward():
    env = _env(goal=(0.3, 0.0), goal_radius=0.05, start_vel=(1.0, 0.0), goal_bonus=10.0)
    for k in range(100):
        _, r, term, trunc, info = env.step((0.0, 0.0))
        dist = abs(info["pos"][0] - 0.3)
        assert term == (dist <= 0.05)
        if term:
            assert info["goal_reached"] and info["collision"] is None and not trunc
            np.testing.assert_allclose(r, 10.0 - dist)
            break
        np.testing.assert_allclose(r, -dist)
    assert term and env.pos[0] >= 0.25
    with pytest.raises(EpisodeOver):
        env.step((0.0, 0.0))


def test_collision_terminates_as_failure():
    env = _env(
        goal=(1.0, 0.0), start_vel=(1.0, 0.0), puck_radius=0.02,
        obstacles=[Obstacle((5.0, 5.0), 0.1), Obstacle((0.5, 0.0), 0.1)],
    )
    for _ in range(100):
        _, r, term, trunc, info = env.step((0.0, 0.0))
        if term:
            break
    assert term and not trunc
    assert info["collision"] == 1 and not info["goal_reached"]
    assert env.pos[0] >= 0.5 - 0.12 - 1e-12  # touched exactly when within r + puck_radius
    assert env.pos[0] - DT * 1.0 < 0.5 - 0.12
    assert r < -9.0


def test_fast_puck_cannot_tunnel_through_obstacle():
    env = _env(start_vel=(50.0, 0.0), obstacles=[Obstacle((0.5, 0.0), 0.05)])
    _, _, term, _, info = env.step((0.0, 0.0))  # moves 1 m in one step, over the obstacle
    assert env.pos[0] > 0.55 and term and info["collision"] == 0


def test_collision_takes_precedence_over_goal():
    env = _env(goal=(0.1, 0.0), goal_radius=0.5, start_vel=(1.0, 0.0),
               obstacles=[Obstacle((0.03, 0.0), 0.01)])
    _, _, term, _, info = env.step((0.0, 0.0))
    assert term and info["collision"] == 0 and not info["goal_reached"]


def test_truncation_at_step_limit():
    env = _env(max_steps=7)
    for k in range(7):
        _, _, term, trunc, _ = env.step((0.0, 0.0))
        assert not term and trunc == (k == 6)
    with pytest.raises(EpisodeOver):
        env.step((0.0, 0.0))
    env.reset(0)
    env.step((0.0, 0.0))


# -- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(mass=0.0), dict(mass=-1.0), dict(damping=-0.1), dict(dt=0.0),
        dict(force_low=(1.0, 0.0), force_high=(0.0, 1.0)), dict(max_steps=0),
        dict(damping=100.0),  # c dt >= m
        dict(mass_changes=[MassChange(1, 0.001)], damping=1.0),
        dict(goal=(math.nan, 0.0)), dict(pos_noise_std=-1.0),
    ],
)
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        Puck2DConfig(**kw)


def test_invalid_actions_and_seeds_rejected():
    env = _env()
    for bad in [(math.nan, 0.0), (1.0,), (1.0, 2.0, 3.0), (math.inf, 0.0)]:
        with pytest.raises(ValueError):
            env.step(bad)
    for bad in [-1, 1.5, None, True]:
        with pytest.raises(ValueError):
            env.reset(bad)
    with pytest.raises(ValueError):
        Puck2D({"no_such_key": 1})


# -- 5. performance ---------------------------------------------------------------


def test_step_costs_less_than_1ms():
    env = _env(
        damping=0.2, friction=0.05, actuator_tau=0.05, pos_noise_std=0.01, vel_noise_std=0.01,
        max_steps=10**9, obstacles=[Obstacle((50.0, 50.0), 1.0)] * 5,
        friction_patches=[FrictionPatch(0.1, center=(9.0, 9.0), radius=1.0)] * 5,
    )
    n = 2000
    best = math.inf
    for _ in range(3):  # best of 3 to absorb scheduler noise
        t0 = time.perf_counter()
        for _ in range(n):
            env.step((0.1, -0.1))
        best = min(best, (time.perf_counter() - t0) / n)
    assert best < 1e-3, f"{best * 1e3:.3f} ms per step"


# -- runner integration -------------------------------------------------------------


class PDSystem1:
    version = "pd-s1-test"

    def __init__(self, goal):
        self.goal = goal
        self.n = 0

    def propose(self, state, prediction, rng):
        from contracts import ActionProposal, Uncertainty

        self.n += 1
        v = dict(zip(state.variables, state.values))
        action = tuple(
            float(8.0 * (g - v[f"pos_{a}"]) - 4.0 * v[f"vel_{a}"]) for a, g in zip("xy", self.goal)
        )
        return ActionProposal(f"pd-{self.n}", self.version, action, 1.0, Uncertainty())


def _runner(tmp_path, env, name):
    clock = lambda: 100.0 + env.time  # noqa: E731  sim time drives the kernel's time base
    tl = TelemetryLog(tmp_path / f"{name}.jsonl", clock=clock)
    cfg = SafetyConfig(
        axis_names=("x", "y"), action_low=(-5.0, -5.0), action_high=(5.0, 5.0),
        workspace_low=(-2.0, -2.0), workspace_high=(2.0, 2.0), max_command_age_s=0.05,
        watchdog_timeout_s=0.1, limit_mode="clamp", mass_kg=1.0, control_dt_s=DT,
    )
    kernel = SafetyKernel(cfg, tl, "operator-secret", clock=clock)
    mods = Modules(environment=env, system1=PDSystem1(env.config.goal))
    return CycleRunner(kernel, tl, np.random.default_rng(0), mods, clock=clock, run_id=name)


def test_plugs_into_cycle_runner_and_reaches_goal(tmp_path):
    def run(name):
        env = Puck2D(Puck2DConfig(goal=(0.5, 0.3), goal_radius=0.05, pos_noise_std=0.002,
                                  vel_noise_std=0.002, damping=0.1), seed=4)
        r = _runner(tmp_path, env, name)
        results = []
        while not env.terminated and len(results) < 400:
            results.append(r.step())
        return env, results

    env, results = run("a")
    assert isinstance(env, Environment)
    assert env.goal_reached and env.collision is None
    assert all(res.kernel_result is not None for res in results)
    assert all(res.outcome is not None for res in results)
    assert results[-1].outcome.success is True
    # every command the environment received is the kernel's approved command
    assert all(res.actuated_command == res.kernel_result.actuator_command for res in results)
    env_b, results_b = run("b")
    assert env_b.pos.tobytes() == env.pos.tobytes() and len(results_b) == len(results)


def test_from_config_builds_puck2d():
    mods = Modules.from_config({
        "environment": {"factory": "simulation.puck2d:Puck2D",
                        "kwargs": {"config": {"goal": (0.2, 0.0)}, "seed": 1}},
    })
    assert isinstance(mods.environment, Puck2D)
    assert mods.environment.config.goal == (0.2, 0.0)
    obs = mods.environment.observe(np.random.default_rng(0))
    obs.validate()
    assert obs.channels == ("pos_x", "pos_y", "vel_x", "vel_y")
