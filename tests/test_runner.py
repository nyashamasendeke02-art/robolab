"""Tests for src/robot/runner.py (Gate 0, ADR-001, ADR-002, REQ-STATE, REQ-S1; ENG-0005)."""

import numpy as np
import pytest

from contracts import (
    ActionProposal,
    AwarenessDecision,
    Observation,
    Outcome,
    PlanProposal,
    PredictionResult,
    Uncertainty,
)
from robot.runner import (
    C_ACTUATE,
    C_AWARENESS,
    C_CYCLE,
    C_ESTIMATE,
    C_OBSERVE,
    C_PREDICT,
    C_S1,
    C_S2,
    CycleRunner,
    Modules,
    NullSystem1,
    RunnerError,
)
from safety import SafetyConfig, SafetyKernel
from state.telemetry import TelemetryLog

KEY = "operator-secret"
T0 = 100.0
DT = 0.01
AXES = ("x", "y")


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def _config(**kw):
    base = dict(
        axis_names=AXES,
        action_low=(-10.0, -10.0),
        action_high=(10.0, 10.0),
        workspace_low=(-1.0, -1.0),
        workspace_high=(1.0, 1.0),
        max_command_age_s=0.05,
        watchdog_timeout_s=0.05,
        limit_mode="reject",
        mass_kg=1.0,
        control_dt_s=DT,
    )
    base.update(kw)
    return SafetyConfig(**base)


class SpyKernel(SafetyKernel):
    """Records every KernelResult it hands out (and every checked command)."""

    def __init__(self, *args, **kwargs):
        # Set before SafetyKernel seals itself; later only mutated, never rebound.
        self.results = []
        self.checked = []
        self.checked_velocity = []
        self.tick_calls = []
        self.tick_velocity = []
        self.no_command_calls = []
        super().__init__(*args, **kwargs)

    def check(self, command, **kwargs):
        r = super().check(command, **kwargs)
        self.checked.append(command)
        self.checked_velocity.append(tuple(kwargs["velocity"]))
        self.results.append(r)
        return r

    def tick(self, now=None, *, velocity=None):
        r = super().tick(now, velocity=velocity)
        self.tick_calls.append(now)
        self.tick_velocity.append(velocity)
        if r is not None:
            self.results.append(r)
        return r

    def no_command(self, now=None, *, velocity=None, **kw):
        r = super().no_command(now, velocity=velocity, **kw)
        self.no_command_calls.append(velocity)
        self.results.append(r)
        return r


# -- stub modules ---------------------------------------------------------------


class PointMassEnv:
    """Deterministic 2-axis point mass with rng measurement noise; advances the clock."""

    version = "point-mass-stub-0.1"

    def __init__(self, clock, noise=0.001):
        self.clock = clock
        self.noise = noise
        self.pos = np.array([0.2, -0.3])
        self.vel = np.zeros(2)
        self.actuated = []

    def observe(self, rng):
        z = np.concatenate([self.pos, self.vel]) + rng.normal(0.0, self.noise, 4)
        names = tuple(f"pos_{a}" for a in AXES) + tuple(f"vel_{a}" for a in AXES)
        return Observation("stub", names, tuple(float(v) for v in z), Uncertainty(measurement=self.noise))

    def actuate(self, command, rng):
        self.actuated.append(command)
        self.vel = self.vel + np.asarray(command) * DT
        self.pos = self.pos + self.vel * DT
        self.clock.t += DT
        return Outcome(
            "stub-act", True, float(-np.sum(self.pos**2)), AXES,
            tuple(float(p) for p in self.pos), Uncertainty(),
        )


class RandomS1:
    """PD controller plus large rng noise: some proposals exceed the kernel limits."""

    version = "random-s1-0.1"

    def __init__(self, scale=12.0):
        self.scale = scale
        self.n = 0
        self.proposals = []

    def propose(self, state, prediction, rng):
        self.n += 1
        v = dict(zip(state.variables, state.values))
        action = tuple(
            float(-20.0 * v[f"pos_{a}"] - 5.0 * v[f"vel_{a}"] + rng.normal(0.0, self.scale))
            for a in AXES
        )
        p = ActionProposal(f"s1-{self.n}", self.version, action, float(rng.uniform()), Uncertainty())
        self.proposals.append(p)
        return p


class FixedS1:
    version = "fixed-s1-0.1"

    def __init__(self, action=(50.0, 0.0)):
        self.action = tuple(float(a) for a in action)

    def propose(self, state, prediction, rng):
        return ActionProposal("fixed", self.version, self.action, 1.0, Uncertainty())


class RaisingS1:
    def propose(self, state, prediction, rng):
        raise RuntimeError("policy crashed")


class EchoWorldModel:
    version = "echo-wm-0.1"

    def __init__(self):
        self.requests = []

    def predict(self, request, rng):
        self.requests.append(request)
        return PredictionResult(self.version, request.variables, request.state, 1, Uncertainty(model=0.1))


class S2FirstAwareness:
    """Requests System 2 when no plan is present, then accepts the S1 proposal."""

    version = "s2-first-0.1"

    def __init__(self):
        self.plans = []

    def arbitrate(self, state, prediction, proposal, plan, rng):
        if plan is None:
            return AwarenessDecision("request_s2", "deliberate", (), None)
        self.plans.append(plan)
        return AwarenessDecision("accept", "after plan", (proposal.proposal_id,), proposal.proposal_id)


class PlanSelectingAwareness:
    """Tries to 'accept' the S2 plan instead of the S1 action."""

    def arbitrate(self, state, prediction, proposal, plan, rng):
        if plan is None:
            return AwarenessDecision("request_s2", "deliberate", (), None)
        return AwarenessDecision("accept", "pick plan", (plan.proposal_id,), plan.proposal_id)


class CountingS2:
    version = "counting-s2-0.1"

    def __init__(self):
        self.calls = 0

    def plan(self, state, prediction, proposal, rng):
        self.calls += 1
        return PlanProposal(f"plan-{self.calls}", self.version, "reach", ("reach",), 1.0, Uncertainty())


# -- helpers ------------------------------------------------------------------


def _setup(tmp_path, name="t", seed=0, modules=None, env=True, run_id="run", **cfg):
    clock = Clock()
    tl = TelemetryLog(tmp_path / f"{name}.jsonl", clock=clock)
    kernel = SpyKernel(_config(**cfg), tl, KEY, clock=clock)
    mods = modules or {}
    if env:
        mods.setdefault("environment", PointMassEnv(clock))
    runner = CycleRunner(
        kernel, tl, np.random.default_rng(seed), modules=Modules(**mods), clock=clock, run_id=run_id
    )
    return runner, kernel, tl, clock


def _decision_sequence(tl):
    """Everything in the log except wall-time latency and the kernel's uuid event ids."""
    return [
        (r.component, r.cycle_id, r.timestamp, r.decision, r.reason, r.level, r.model_version, r.data)
        for r in tl.records()
    ]


# -- acceptance criterion 1: determinism -------------------------------------------


def test_same_seed_gives_identical_telemetry_decision_sequences(tmp_path):
    seqs = []
    for name in ("a", "b"):
        runner, kernel, tl, _ = _setup(tmp_path, name=name, seed=7, modules={"system1": RandomS1()})
        runner.run(40)
        seqs.append(_decision_sequence(tl))
        tl.close()
    a, b = seqs
    assert len(a) > 40 * 5
    assert a == b
    verdicts = {r[3] for r in a if r[0] == "safety"}
    assert {"approve", "reject"} <= verdicts  # the sequence exercises both paths


def test_different_seed_changes_the_sequence(tmp_path):
    seqs = []
    for seed in (1, 2):
        runner, _, tl, _ = _setup(tmp_path, name=f"s{seed}", seed=seed, modules={"system1": RandomS1()})
        runner.run(20)
        seqs.append([(r[0], r[3], r[7]) for r in _decision_sequence(tl) if r[0] == C_S1])
        tl.close()
    assert seqs[0] != seqs[1]


# -- acceptance criterion 2: no actuation bypasses the kernel -------------------------


def test_every_actuated_command_came_from_a_kernel_result(tmp_path):
    s1 = RandomS1(scale=30.0)
    runner, kernel, tl, clock = _setup(tmp_path, modules={"system1": s1})
    env = runner.modules.environment
    results = runner.run(60)

    assert len(env.actuated) == 60
    # Same objects, same order as the actuator commands the kernel handed out.
    kernel_cmds = [r.actuator_command for r in kernel.results]
    it = iter(kernel_cmds)
    for cmd in env.actuated:
        assert any(cmd is k for k in it)
    for res, cmd in zip(results, env.actuated):
        assert res.actuated_command is cmd is res.kernel_result.actuator_command
        assert any(res.kernel_result is r for r in kernel.results)
    # Every S1 proposal was checked by the kernel, and every actuated command is in limits.
    assert [c.payload for c in kernel.checked] == s1.proposals
    for cmd in env.actuated:
        assert all(-10.0 <= u <= 10.0 for u in cmd)
    # Out-of-limit proposals existed and were replaced by the safe action, which
    # since kernel v1.1 brakes the measured velocity: clip(-m v / dt, limits).
    rejected = [
        (r, v) for r, v in zip(results, kernel.checked_velocity)
        if r.kernel_result.decision.verdict == "reject"
    ]
    assert rejected
    for r, vel in rejected:
        assert r.actuated_command == tuple(min(max(-v / DT, -10.0), 10.0) for v in vel)
    tl.close()


def test_out_of_limit_action_never_reaches_environment(tmp_path):
    # Noise-free body at rest: the braking safe action is zero force.
    runner, kernel, tl, _ = _setup(tmp_path, modules={"system1": FixedS1((50.0, 0.0))})
    runner.modules.environment.noise = 0.0
    runner.run(5)
    env = runner.modules.environment
    assert env.actuated == [(0.0, 0.0)] * 5
    assert all(r.decision.verdict == "reject" for r in kernel.results)
    tl.close()


def test_clamp_mode_actuates_the_clamped_action(tmp_path):
    runner, kernel, tl, _ = _setup(
        tmp_path, modules={"system1": FixedS1((50.0, 0.0))}, limit_mode="clamp"
    )
    runner.run(1)
    assert runner.modules.environment.actuated == [(10.0, 0.0)]
    tl.close()


def test_missing_observed_kinematics_is_rejected_by_kernel(tmp_path):
    # G1-5: the kernel reads position/velocity from the raw observation. An
    # observation without the velocity channels makes the command invalid_state.
    class NoVelocityEnv(PointMassEnv):
        def observe(self, rng):
            return Observation("stub", ("pos_x", "pos_y"), tuple(float(p) for p in self.pos),
                               Uncertainty())

    runner, kernel, tl, _ = _setup(
        tmp_path, modules={"system1": FixedS1((1.0, 1.0)), "environment": NoVelocityEnv(Clock())},
    )
    (res,) = runner.run(1)
    assert res.kernel_result.decision.violated_constraints == ("invalid_state",)
    assert res.actuated_command == (0.0, 0.0)  # unknown velocity: configured safe action
    tl.close()


def test_kernel_ignores_the_state_estimate(tmp_path):
    # An estimator that drops the kinematics no longer blinds the kernel (it used
    # to cause invalid_state); the kernel checks the observed state.
    class BadEstimator:
        def estimate(self, observation, rng):
            from contracts import StateUpdate

            return StateUpdate("physical", ("pos_x",), (0.0,), Uncertainty(), ())

    runner, kernel, tl, _ = _setup(
        tmp_path, modules={"system1": FixedS1((1.0, 1.0)), "state_estimator": BadEstimator()}
    )
    (res,) = runner.run(1)
    assert res.kernel_result.decision.verdict == "approve"
    assert res.actuated_command == (1.0, 1.0)
    tl.close()


def test_kernel_tick_every_cycle_and_safe_action_without_commands(tmp_path):
    runner, kernel, tl, clock = _setup(tmp_path)  # NullSystem1: never proposes
    env = runner.modules.environment
    env.noise = 0.0
    env.vel = np.array([0.02, -0.01])  # moving slowly
    res = runner.run(2)
    # No command: the kernel's no_command decision brakes the observed velocity.
    assert [r.kernel_source for r in res] == ["no_command", "no_command"]
    assert all(r.kernel_result.decision.violated_constraints == ("no_command",) for r in res)
    assert env.actuated[0] == pytest.approx((-2.0, 1.0))  # -m v / dt
    assert np.all(env.vel == 0.0)  # stopped exactly in one step
    assert [r.actuated_command for r in res] == env.actuated
    clock.t += 1.0  # longer than watchdog_timeout_s
    (r3,) = runner.run(1)
    assert r3.kernel_source == "tick"
    assert r3.kernel_result.decision.violated_constraints == ("watchdog_timeout",)
    assert env.actuated[-1] == (0.0, 0.0)  # at rest: zero force
    assert r3.actuated_command is r3.kernel_result.actuator_command
    assert len(kernel.tick_calls) == 3  # one watchdog tick per cycle
    assert kernel.tick_velocity[0] == pytest.approx((0.02, -0.01))  # observed velocity
    tl.close()


def test_emergency_stop_blocks_commands(tmp_path):
    runner, kernel, tl, _ = _setup(tmp_path, modules={"system1": FixedS1((1.0, 0.0))})
    runner.modules.environment.noise = 0.0  # body at rest: braking safe action is zero
    kernel.emergency_stop("test")
    res = runner.run(3)
    assert all(r.kernel_result.decision.verdict == "emergency_stop" for r in res)
    assert runner.modules.environment.actuated == [(0.0, 0.0)] * 3
    tl.close()


def test_faulty_module_is_logged_and_no_command_is_sent(tmp_path):
    runner, kernel, tl, _ = _setup(tmp_path, modules={"system1": RaisingS1()})
    (res,) = runner.run(1)
    assert res.awareness_decision == "abstain"
    assert kernel.checked == []
    # ... but the kernel's braking safe action is still actuated.
    assert res.kernel_source == "no_command"
    assert runner.modules.environment.actuated == [res.kernel_result.actuator_command]
    errs = [r for r in tl.records() if r.component == C_S1]
    assert errs[0].decision == "error" and errs[0].level == "error"
    assert "policy crashed" in errs[0].reason
    tl.close()


def test_runner_telemetry_failure_latches_estop(tmp_path):
    runner, kernel, tl, _ = _setup(tmp_path, modules={"system1": FixedS1((1.0, 0.0))})
    tl.close()
    with pytest.raises(RunnerError):
        runner.step()
    assert kernel.estopped


# -- System 2 path (ADR-002) ---------------------------------------------------------


def test_system2_only_on_request_and_plans_are_never_actuated(tmp_path):
    s2 = CountingS2()
    runner, kernel, tl, _ = _setup(tmp_path, modules={"system1": FixedS1((1.0, 0.0)), "system2": s2})
    runner.run(3)
    assert s2.calls == 0  # pass-through awareness never asks for S2

    s2b, aw = CountingS2(), S2FirstAwareness()
    runner, kernel, tl2, _ = _setup(
        tmp_path, name="s2", modules={"system1": FixedS1((1.0, 0.0)), "system2": s2b, "awareness": aw}
    )
    res = runner.run(3)
    assert s2b.calls == 3 and len(aw.plans) == 3
    assert all(type(c.payload) is ActionProposal for c in kernel.checked)
    assert [r.actuated_command for r in res] == [(1.0, 0.0)] * 3
    comps = [r.component for r in tl2.records() if r.cycle_id == 1]
    assert comps.index(C_S2) < comps.index("safety")
    tl.close()
    tl2.close()


def test_accepting_a_plan_sends_only_the_safe_action_to_actuators(tmp_path):
    runner, kernel, tl, _ = _setup(
        tmp_path,
        modules={"system1": FixedS1((1.0, 0.0)), "system2": CountingS2(),
                 "awareness": PlanSelectingAwareness()},
    )
    runner.modules.environment.noise = 0.0  # at rest: the braking safe action is zero
    (res,) = runner.run(1)
    assert kernel.checked == []
    assert res.kernel_source == "no_command"
    assert runner.modules.environment.actuated == [(0.0, 0.0)]
    tl.close()


# -- acceptance criterion 3: modules swap without runner changes -----------------------


def test_modules_swap_by_configuration(tmp_path):
    clock = Clock()
    mods = Modules.from_config(
        {
            "environment": {"factory": f"{__name__}:PointMassEnv", "kwargs": {"clock": clock}},
            "system1": f"{__name__}:RandomS1",
            "world_model": f"{__name__}:EchoWorldModel",
            "awareness": "robot.runner:PassThroughAwareness",
        }
    )
    assert isinstance(mods.system1, RandomS1) and isinstance(mods.world_model, EchoWorldModel)
    tl = TelemetryLog(tmp_path / "cfg.jsonl", clock=clock)
    kernel = SpyKernel(_config(), tl, KEY, clock=clock)
    runner = CycleRunner(kernel, tl, np.random.default_rng(3), modules=mods, clock=clock)
    runner.run(5)
    assert len(mods.environment.actuated) == 5
    assert len(mods.world_model.requests) == 5
    # The world model is told the last actuated command (zero-order hold).
    assert mods.world_model.requests[1].action == mods.environment.actuated[0]
    wm = [r for r in tl.records() if r.component == C_PREDICT]
    assert all(r.decision == "predicted" and r.model_version == "echo-wm-0.1" for r in wm)

    # Swap S1 on the same runner class: only the configuration changes.
    runner2, _, tl2, _ = _setup(tmp_path, name="swap", modules={"system1": NullSystem1()})
    runner2.run(2)
    assert {r.decision for r in tl2.records() if r.component == C_S1} == {"no_proposal"}
    tl.close()
    tl2.close()


def test_modules_reject_non_conforming_objects():
    with pytest.raises(TypeError):
        Modules(system1=object())
    with pytest.raises(ValueError):
        Modules.from_config({"planner": "robot.runner:NullSystem2"})


def test_runner_requires_real_kernel_and_generator(tmp_path):
    clock = Clock()
    tl = TelemetryLog(tmp_path / "x.jsonl", clock=clock)
    kernel = SafetyKernel(_config(), tl, KEY, clock=clock)
    with pytest.raises(TypeError):
        CycleRunner(object(), tl, np.random.default_rng(0))
    with pytest.raises(TypeError):
        CycleRunner(kernel, tl, np.random.RandomState(0))
    tl.close()


# -- acceptance criterion 4: per-module latency in the telemetry summary -----------------


def test_per_module_latency_in_telemetry_summary(tmp_path):
    runner, _, tl, _ = _setup(
        tmp_path,
        modules={"system1": RandomS1(), "world_model": EchoWorldModel(),
                 "system2": CountingS2(), "awareness": S2FirstAwareness()},
    )
    results = runner.run(10)
    summary = runner.summary()
    for comp in (C_OBSERVE, C_ESTIMATE, C_PREDICT, C_S1, C_S2, C_AWARENESS, "safety", C_ACTUATE, C_CYCLE):
        assert comp in summary, comp
        assert summary[comp]["count"] >= 10
        assert summary[comp]["p50_ms"] >= 0.0 and summary[comp]["p95_ms"] >= summary[comp]["p50_ms"]
    assert summary[C_AWARENESS]["count"] == 20  # arbitrate, S2, re-arbitrate
    for res in results:
        assert {C_OBSERVE, C_ESTIMATE, C_PREDICT, C_S1, C_S2, C_AWARENESS, C_ACTUATE, C_CYCLE} <= set(res.latency_ms)
    tl.close()
