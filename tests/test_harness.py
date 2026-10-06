"""Tests for src/simulation/harness.py and policies.py (Gate 1, REQ-SIM, REQ-LOG, REQ-REPRO; ENG-0009)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from contracts import StateUpdate, Uncertainty
from robot.runner import CycleRunner
from simulation.harness import (
    EvalSet,
    EvalSetSpec,
    HarnessConfig,
    SimClock,
    TaskInstance,
    build_eval_set,
    classify,
    decision_trace,
    run_episode,
    run_episodes,
    summarize,
)
from simulation.policies import PDConfig, PDController, RandomPolicy
from simulation.puck2d import Impulse, Obstacle
from state.telemetry import TelemetryLog, read_records

REPO = Path(__file__).resolve().parents[1]
PLAIN = EvalSetSpec(
    n_tasks=8, max_obstacles=0, max_impulses=0, mass_change_prob=0.0, friction_patch_prob=0.0
)
RICH = EvalSetSpec(
    n_tasks=6, max_obstacles=3, max_impulses=2, mass_change_prob=1.0, friction_patch_prob=1.0
)


def _run(tmp_path, name, eval_set, policy, seed=7, **kw):
    path = tmp_path / f"{name}.jsonl"
    with TelemetryLog(path) as log:
        metrics = run_episodes(eval_set, policy, log, seed=seed, **kw)
    return metrics, read_records(path)


# -- evaluation sets ---------------------------------------------------------


def test_eval_set_json_round_trip_unchanged(tmp_path):
    for spec in (PLAIN, RICH, EvalSetSpec()):
        es = build_eval_set(11, spec, env_params={"friction": 0.05, "max_steps": 300})
        assert EvalSet.from_json(es.to_json()) == es
        path = tmp_path / "set.json"
        es.save(path)
        loaded = EvalSet.load(path)
        assert loaded == es
        assert loaded.to_json() == es.to_json()
        assert json.loads(path.read_text(encoding="utf-8")) == es.to_dict()


def test_rich_eval_set_contains_every_disturbance_kind():
    es = build_eval_set(5, RICH)
    assert any(t.obstacles for t in es.tasks)
    assert any(t.impulses for t in es.tasks)
    assert all(t.mass_changes and t.friction_patches for t in es.tasks)
    # Round trip keeps the typed disturbance objects.
    back = EvalSet.from_json(es.to_json())
    assert back.tasks[0].friction_patches == es.tasks[0].friction_patches


def test_eval_set_builder_is_deterministic_in_seed():
    assert build_eval_set(3, RICH) == build_eval_set(3, RICH)
    assert build_eval_set(3, RICH).to_json() == build_eval_set(3, RICH).to_json()
    assert build_eval_set(3, RICH).tasks != build_eval_set(4, RICH).tasks


def test_eval_set_tasks_respect_spec():
    es = build_eval_set(9, RICH)
    for t in es.tasks:
        assert math.dist(t.start, t.goal) >= RICH.min_start_goal_distance
        for ob in t.obstacles:
            assert math.dist(ob.center, t.start) > ob.radius + RICH.obstacle_clearance
            assert math.dist(ob.center, t.goal) > ob.radius + RICH.obstacle_clearance
        for imp in t.impulses:
            assert 0 <= imp.step < RICH.max_disturbance_step


def test_eval_set_rejects_bad_input():
    es = build_eval_set(1, PLAIN)
    raw = es.to_dict()
    with pytest.raises(ValueError):
        EvalSet.from_dict({**raw, "extra": 1})
    with pytest.raises(ValueError):
        EvalSet.from_dict({**raw, "env_params": {"goal": [0.0, 0.0]}})
    with pytest.raises(ValueError):
        EvalSet.from_dict({**raw, "version": "other"})
    with pytest.raises(ValueError):
        EvalSet.from_dict({**raw, "tasks": raw["tasks"] + raw["tasks"][:1]})


# -- determinism ---------------------------------------------------------------


@pytest.mark.parametrize("policy_factory", [PDController, RandomPolicy])
def test_same_seed_same_metrics_and_decision_trace(tmp_path, policy_factory):
    es = build_eval_set(
        2, RICH, env_params={"pos_noise_std": 0.01, "vel_noise_std": 0.01, "max_steps": 200}
    )
    m1, r1 = _run(tmp_path, "a", es, policy_factory(), seed=5)
    m2, r2 = _run(tmp_path, "b", es, policy_factory(), seed=5)
    assert m1 == m2
    assert decision_trace(r1) == decision_trace(r2)
    assert len(r1) > 0


def test_different_runner_seed_changes_random_policy(tmp_path):
    es = build_eval_set(2, PLAIN)
    m1, r1 = _run(tmp_path, "a", es, RandomPolicy(), seed=5, n_episodes=2)
    m2, r2 = _run(tmp_path, "b", es, RandomPolicy(), seed=6, n_episodes=2)
    assert decision_trace(r1) != decision_trace(r2)


def test_telemetry_uses_simulation_clock(tmp_path):
    es = build_eval_set(2, PLAIN)
    metrics, records = _run(tmp_path, "a", es, PDController(), n_episodes=1)
    dt = 0.02
    runner = [r for r in records if r.component == "runner"]
    assert len(runner) == metrics[0].cycles
    # Every runner record is stamped with its cycle's start time (cycle_id - 1) * dt.
    for r in runner:
        assert r.timestamp == (r.cycle_id - 1) * dt
    assert max(r.timestamp for r in records) == pytest.approx(metrics[0].steps * dt)
    assert records[0].decision == "episode_start" and records[-1].decision == "episode_end"


# -- every action through the kernel ------------------------------------------------


def test_every_actuated_command_is_a_kernel_output(tmp_path, monkeypatch):
    seen = []
    orig = CycleRunner.step

    def step(self):
        res = orig(self)
        seen.append(res)
        return res

    monkeypatch.setattr(CycleRunner, "step", step)
    es = build_eval_set(4, PLAIN)
    metrics, _ = _run(tmp_path, "a", es, RandomPolicy(), n_episodes=2)
    assert len(seen) == sum(m.cycles for m in metrics)
    for res in seen:
        assert res.kernel_result is not None
        assert res.actuated_command == res.kernel_result.actuator_command
    # Steps == actuated cycles: the environment only moves on kernel outputs.
    assert sum(m.steps for m in metrics) == sum(r.actuated_command is not None for r in seen)


def test_interventions_counted_from_kernel_verdicts(tmp_path):
    # Tight workspace: commands toward a goal outside it are rejected. (The
    # zero-force safe action does not brake a frictionless puck, so it may still
    # coast out of the workspace; success is therefore not asserted here.)
    task = TaskInstance("t0", 0, (0.0, 0.0), (1.0, 0.0))
    es = EvalSet("tight", 0, PLAIN, {"max_steps": 100}, (task,))
    cfg = HarnessConfig(workspace_low=(-0.5, -0.5), workspace_high=(0.5, 0.5))
    with TelemetryLog(tmp_path / "t.jsonl") as log:
        m = run_episode(es, task, PDController(), log, seed=0, config=cfg)
    assert m.kernel_rejections > 0
    assert m.safety_interventions == (
        m.kernel_clamps + m.kernel_rejections + m.watchdog_timeouts + m.emergency_stops
    )
    safety = [r for r in read_records(tmp_path / "t.jsonl") if r.component == "safety"]
    assert sum(r.decision == "reject" for r in safety) >= m.kernel_rejections


def test_clamps_counted_when_policy_exceeds_limits(tmp_path):
    task = TaskInstance("t0", 0, (0.0, 0.0), (1.0, 0.0))
    es = EvalSet("clamp", 0, PLAIN, {"max_steps": 400}, (task,))
    with TelemetryLog(tmp_path / "t.jsonl") as log:
        m = run_episode(es, task, PDController({"force_limit": 50.0}), log, seed=0)
    assert m.kernel_clamps > 0
    assert m.success  # clamped to the actuator limits it still reaches the goal


def test_idle_policy_still_advances_time_and_trips_watchdog(tmp_path):
    class Idle:
        version = "idle-0"

        def propose(self, state, prediction, rng):
            return None

    task = TaskInstance("t0", 0, (0.0, 0.0), (1.0, 0.0))
    es = EvalSet("idle", 0, PLAIN, {"max_steps": 50}, (task,))
    with TelemetryLog(tmp_path / "t.jsonl") as log:
        m = run_episode(es, task, Idle(), log, seed=0)
    assert m.cycles == 50
    assert m.watchdog_timeouts > 0
    assert m.idle_cycles + m.steps == m.cycles
    assert m.truncated and not m.success


# -- policies / instrument sanity ------------------------------------------------


def test_pd_reaches_goal_without_disturbances_or_obstacles(tmp_path):
    es = build_eval_set(21, PLAIN)
    metrics, _ = _run(tmp_path, "pd", es, PDController(PDConfig.load(REPO / "configs" / "pd_controller.json")))
    for m in metrics:
        assert m.success, m
        assert m.collisions == 0
        assert m.time_to_goal == pytest.approx(m.steps * 0.02)
        assert m.final_distance <= 0.05
        assert m.path_length > 0
    assert summarize(metrics)["success_rate"] == 1.0


def test_pd_default_task(tmp_path):
    task = TaskInstance("t0", 0, (0.0, 0.0), (1.0, 0.0))
    es = EvalSet("one", 0, PLAIN, {}, (task,))
    with TelemetryLog(tmp_path / "t.jsonl") as log:
        m = run_episode(es, task, PDController(), log, seed=0)
    assert m.success and m.safety_interventions == 0
    assert m.path_length == pytest.approx(1.0, abs=0.06)


def test_random_policy_mostly_fails(tmp_path):
    es = build_eval_set(21, PLAIN)
    rnd, _ = _run(tmp_path, "rnd", es, RandomPolicy())
    pd, _ = _run(tmp_path, "pd", es, PDController())
    s_rnd, s_pd = summarize(rnd)["success_rate"], summarize(pd)["success_rate"]
    assert s_rnd < 0.5
    assert s_pd - s_rnd >= 0.5  # the instrument separates the two baselines


def test_collision_counted(tmp_path):
    task = TaskInstance("t0", 0, (0.0, 0.0), (1.0, 0.0), obstacles=(Obstacle((0.5, 0.0), 0.1),))
    es = EvalSet("obs", 0, PLAIN, {}, (task,))
    with TelemetryLog(tmp_path / "t.jsonl") as log:
        m = run_episode(es, task, PDController(), log, seed=0)
    assert m.collisions == 1 and not m.success and m.time_to_goal is None
    assert not m.truncated


def test_disturbances_reach_the_environment(tmp_path):
    base = TaskInstance("t0", 0, (0.0, 0.0), (1.0, 0.0))
    kicked = TaskInstance("t0", 0, (0.0, 0.0), (1.0, 0.0), impulses=(Impulse(5, (0.0, 2.0)),))
    with TelemetryLog(tmp_path / "t.jsonl") as log:
        m0 = run_episode(EvalSet("a", 0, PLAIN, {}, (base,)), base, PDController(), log, seed=0)
        m1 = run_episode(EvalSet("b", 0, PLAIN, {}, (kicked,)), kicked, PDController(), log, seed=0)
    assert m1.path_length > m0.path_length


def test_pd_controller_law_and_abstention():
    pd = PDController({"kp": 2.0, "kd": 1.0, "force_limit": 10.0}, goal=(1.0, 1.0))
    st = StateUpdate("physical", ("pos_x", "pos_y", "vel_x", "vel_y"),
                     (0.0, 0.5, 0.5, 0.0), Uncertainty(), ())
    p = pd.propose(st, None, np.random.default_rng(0))
    assert p.action == pytest.approx((2.0 * 1.0 - 0.5, 2.0 * 0.5))
    assert p.proposal_id == "pd-1"
    missing = StateUpdate("physical", ("pos_x",), (0.0,), Uncertainty(), ())
    assert pd.propose(missing, None, np.random.default_rng(0)) is None
    pd.reset(TaskInstance("t", 0, (0.0, 0.0), (3.0, -3.0)))
    assert tuple(pd.goal) == (3.0, -3.0)
    with pytest.raises(ValueError):
        PDConfig.from_mapping({"kp": 1.0, "ki": 0.1})


def test_random_policy_uniform_within_limits():
    pol = RandomPolicy((-1.0, -2.0), (1.0, 2.0))
    rng = np.random.default_rng(0)
    acts = np.array([pol.propose(None, None, rng).action for _ in range(2000)])
    assert np.all(acts[:, 0] >= -1.0) and np.all(acts[:, 0] <= 1.0)
    assert np.all(acts[:, 1] >= -2.0) and np.all(acts[:, 1] <= 2.0)
    assert abs(acts[:, 0].mean()) < 0.1 and abs(acts[:, 1].mean()) < 0.2


def test_sim_clock_and_classify():
    c = SimClock(0.02)
    c.cycle = 3
    assert c() == 3 * 0.02
    assert classify(None) is None
