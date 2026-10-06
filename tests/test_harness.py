"""Tests for the episode harness, policies and evaluation sets (Gate 1, REQ-SIM,
REQ-LOG, REQ-REPRO; ENG-0008)."""

import json
from pathlib import Path

import numpy as np
import pytest

from contracts import StateUpdate, Uncertainty
from robot.runner import Modules
from simulation.evalset import EvaluationSet, TaskInstance, build_evaluation_set
from simulation.harness import (
    HarnessConfig,
    SimClock,
    decision_sequence,
    run_episode,
    run_episodes,
    summarize,
)
from simulation.policies import PDController, RandomPolicy
from simulation.puck2d import Obstacle
from state.telemetry import read_records

ROOT = Path(__file__).resolve().parents[1]


def _state(px, py, vx, vy):
    return StateUpdate("physical", ("pos_x", "pos_y", "vel_x", "vel_y"),
                       tuple(float(x) for x in (px, py, vx, vy)), Uncertainty(), ())


def _plain_task(start=(-0.5, 0.2), goal=(0.6, -0.3), seed=3, tid="t0"):
    return TaskInstance(tid, seed, start, goal)


# -- policies ---------------------------------------------------------------------


def test_pd_law_and_saturation():
    pd = PDController(kp=2.0, kd=1.0, max_force=10.0, goal=(1.0, 0.0))
    p = pd.propose(_state(0.0, 0.5, 0.2, 0.0), None, np.random.default_rng(0))
    assert p.action == pytest.approx((2.0 * 1.0 - 0.2, 2.0 * -0.5))
    pd = PDController(kp=100.0, kd=0.0, max_force=3.0, goal=(1.0, -1.0))
    assert pd.propose(_state(0, 0, 0, 0), None, None).action == (3.0, -3.0)


def test_pd_abstains_without_state_or_goal():
    assert PDController().propose(_state(0, 0, 0, 0), None, None) is None  # no goal
    bad = StateUpdate("physical", ("pos_x",), (0.0,), Uncertainty(), ())
    assert PDController(goal=(1, 0)).propose(bad, None, None) is None


def test_random_policy_uniform_in_bounds_and_seeded():
    pol = RandomPolicy(low=(-2.0, -1.0), high=(2.0, 1.0))
    a = [pol.propose(None, None, np.random.default_rng(7)).action for _ in range(2)]
    assert a[0] == a[1]
    rng = np.random.default_rng(1)
    xs = np.array([pol.propose(None, None, rng).action for _ in range(2000)])
    assert np.all(xs[:, 0] >= -2) and np.all(xs[:, 0] <= 2)
    assert np.all(np.abs(xs[:, 1]) <= 1)
    assert abs(xs[:, 0].mean()) < 0.15 and xs[:, 0].std() > 1.0


def test_policies_load_from_config():
    cfg = json.loads((ROOT / "configs" / "policies.json").read_text())
    pd = Modules.from_config({"system1": cfg["pd"]}).system1
    assert isinstance(pd, PDController) and (pd.kp, pd.kd) == (20.0, 8.0)
    assert isinstance(Modules.from_config({"system1": cfg["random"]}).system1, RandomPolicy)


# -- evaluation sets ---------------------------------------------------------------


def _rich_set(seed=11):
    return build_evaluation_set(
        seed, 8, n_obstacles=(1, 3), n_impulses=(0, 2), n_mass_changes=(0, 2)
    )


def test_evaluation_set_json_round_trip(tmp_path):
    es = _rich_set()
    assert any(t.obstacles for t in es) and any(t.impulses for t in es)
    assert any(t.mass_changes for t in es)
    assert EvaluationSet.from_json(es.to_json()) == es
    path = es.save(tmp_path / "eval.json")
    loaded = EvaluationSet.load(path)
    assert loaded == es
    assert loaded.to_json() == es.to_json() == path.read_text(encoding="utf-8")
    assert loaded.fingerprint() == es.fingerprint()
    with pytest.raises(FileExistsError):
        es.save(path)  # frozen: never overwritten


def test_evaluation_set_deterministic_from_seed():
    assert _rich_set(11) == _rich_set(11)
    assert _rich_set(11).fingerprint() != _rich_set(12).fingerprint()


def test_evaluation_set_tasks_are_valid():
    for t in _rich_set():
        start, goal = np.array(t.start), np.array(t.goal)
        assert np.all(np.abs(start) <= 1.0) and np.all(np.abs(goal) <= 1.0)
        assert np.hypot(*(goal - start)) >= 0.5
        for ob in t.obstacles:
            c = np.array(ob.center)
            assert np.hypot(*(c - start)) > ob.radius + 0.1
            assert np.hypot(*(c - goal)) > ob.radius + 0.05 + 0.1
        assert len({m.step for m in t.mass_changes}) == len(t.mass_changes)


def test_evaluation_set_is_immutable_after_construction():
    es = _rich_set()
    fingerprint, text = es.fingerprint(), es.to_json()
    es.params["n_tasks"] = 99
    es.params.clear()
    es.to_dict()["params"]["n_tasks"] = 99
    source = {"n_tasks": 1, "arena_low": [-1.0, -1.0]}
    built = EvaluationSet(name="x", seed=0, params=source, tasks=())
    before = built.fingerprint()
    source["n_tasks"] = 2
    source["arena_low"].append(0.0)
    assert built.fingerprint() == before
    assert es.fingerprint() == fingerprint and es.to_json() == text
    with pytest.raises(AttributeError):
        es.params = {}  # type: ignore[misc]


def test_evaluation_set_rejects_unknown_keys():
    raw = json.loads(_rich_set().to_json())
    raw["tasks"][0]["extra"] = 1
    with pytest.raises(ValueError):
        EvaluationSet.from_json(json.dumps(raw))


# -- harness ----------------------------------------------------------------------


def test_pd_reaches_goal_without_disturbance_or_obstacles(tmp_path):
    es = build_evaluation_set(5, 6)  # no obstacles, no disturbances by default
    assert all(not (t.obstacles or t.impulses or t.mass_changes) for t in es)
    ms = run_episodes(PDController(kp=20.0, kd=8.0), es.tasks, tmp_path)
    for m in ms:
        assert m.success and m.terminated and not m.truncated
        assert m.collisions == 0 and m.safety_interventions == 0
        assert m.time_to_goal == pytest.approx(m.steps * 0.02)
        assert m.final_distance <= 0.05
    task = es.tasks[0]
    assert ms[0].path_length >= np.hypot(*(np.subtract(task.goal, task.start))) - 0.05


def test_random_policy_mostly_fails(tmp_path):
    es = build_evaluation_set(5, 10)
    ms = run_episodes(RandomPolicy(), es.tasks, tmp_path)
    s = summarize(ms)
    assert s["success_rate"] <= 0.2
    assert all(m.truncated or m.collisions for m in ms if not m.success)
    # Leaving the workspace is refused by the kernel and counted.
    assert s["total_safety_interventions"] > 0


def _run_twice(tmp_path, policy_factory, es):
    out = []
    for name in ("a", "b"):
        ms = run_episodes(policy_factory(), es.tasks, tmp_path / name)
        seqs = [decision_sequence(read_records(m.telemetry_path)) for m in ms]
        out.append(([{k: v for k, v in m.to_dict().items() if k != "telemetry_path"}
                     for m in ms], seqs))
    return out


@pytest.mark.parametrize("factory", [RandomPolicy, PDController])
def test_determinism_metrics_and_telemetry(tmp_path, factory):
    es = build_evaluation_set(21, 3, n_obstacles=(0, 2), n_impulses=(0, 2),
                              n_mass_changes=(0, 1))
    (m1, s1), (m2, s2) = _run_twice(tmp_path, factory, es)
    assert m1 == m2
    assert s1 == s2
    assert all(len(s) > 0 for s in s1)
    # Timestamps are the simulation clock (cycle index x dt), not wall time.
    stamps = sorted({r[2] for r in s1[0]})
    assert stamps[0] == 0.0
    assert stamps[-1] == pytest.approx((m1[0]["cycles"] - 1) * 0.02)


def test_every_actuated_command_passes_the_kernel(tmp_path):
    m = run_episode(RandomPolicy(), _plain_task(), tmp_path / "t.jsonl")
    recs = read_records(m.telemetry_path)
    runner = [r for r in recs if r.component == "runner"]
    safety = [r for r in recs if r.component == "safety"]
    assert len(runner) == m.cycles and m.steps == m.cycles
    assert all(r.data["kernel_verdict"] is not None for r in runner)
    # Each actuated command equals the actuator_command of a kernel decision of that cycle.
    for r in runner:
        same = [s for s in safety if s.timestamp == r.timestamp]
        assert r.data["actuated_command"] in [s.data["actuator_command"] for s in same]
    n_bad = sum(r.data["kernel_verdict"] != "approve" for r in runner)
    assert n_bad == m.kernel_rejects + m.kernel_estops


def test_obstacle_collision_counted(tmp_path):
    task = TaskInstance("t", 0, (-0.5, 0.0), (0.5, 0.0), obstacles=(Obstacle((0.0, 0.0), 0.1),))
    m = run_episode(PDController(), task, tmp_path / "t.jsonl")
    assert m.collisions == 1 and not m.success and m.terminated


def test_telemetry_never_overwritten(tmp_path):
    p = tmp_path / "t.jsonl"
    run_episode(PDController(), _plain_task(), p)
    with pytest.raises(FileExistsError):
        run_episode(PDController(), _plain_task(), p)
    run_episodes(PDController(), [_plain_task()], tmp_path / "d")
    with pytest.raises(FileExistsError):
        run_episodes(PDController(), [_plain_task(tid="t1")], tmp_path / "d")


def test_harness_config_safety_limits_follow_env():
    cfg = HarnessConfig().safety_config()
    assert cfg.action_low == (-10.0, -10.0) and cfg.action_high == (10.0, 10.0)
    assert cfg.control_dt_s == 0.02 and cfg.mass_kg == 1.0
    assert cfg.limit_mode == "reject"


def test_sim_clock():
    c = SimClock()
    assert c() == 0.0
    c.t = 0.4
    assert c() == 0.4
