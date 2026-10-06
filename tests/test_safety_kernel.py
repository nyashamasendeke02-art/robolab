"""Tests for src/safety/kernel.py (Gate 0, REQ-SAFE, ADR-003, ENG-0004)."""

import dataclasses
import json

import pytest

from contracts import (
    ActionProposal,
    Envelope,
    Observation,
    SafetyDecision,
    Uncertainty,
)
from state.telemetry import TelemetryLog
from safety import SafetyConfig, SafetyKernel

KEY = "operator-secret"
T0 = 100.0
POS = (0.0, 0.0)
VEL = (0.0, 0.0)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def _config(**kw):
    base = dict(
        axis_names=("x", "y"),
        action_low=(-10.0, -10.0),
        action_high=(10.0, 10.0),
        workspace_low=(-1.0, -1.0),
        workspace_high=(1.0, 1.0),
        max_command_age_s=0.05,
        watchdog_timeout_s=0.2,
        limit_mode="reject",
        mass_kg=1.0,
        control_dt_s=0.1,
    )
    base.update(kw)
    return SafetyConfig(**base)


@pytest.fixture
def tl(tmp_path):
    with TelemetryLog(tmp_path / "safety.jsonl") as log:
        yield log


def _kernel(tl, clock=None, **kw):
    return SafetyKernel(_config(**kw), tl, KEY, clock=clock or Clock())


def _cmd(action, ts=T0, pid="p1", cycle=1):
    return Envelope(
        message_id=f"m-{pid}",
        timestamp=ts,
        source="awareness",
        destination="safety",
        cycle_id=cycle,
        correlation_id="c1",
        payload=ActionProposal(
            proposal_id=pid,
            policy_version="s1-0.1",
            action=tuple(float(a) for a in action),
            confidence=0.9,
            uncertainty=Uncertainty(),
        ),
    )


def _check(k, cmd, now=T0, pos=POS, vel=VEL):
    return k.check(cmd, position=pos, velocity=vel, now=now)


# 0. Nominal path -----------------------------------------------------------


def test_in_limit_command_is_approved_unchanged(tl):
    k = _kernel(tl)
    r = _check(k, _cmd((1.0, -2.0)))
    assert isinstance(r.decision, SafetyDecision)
    assert r.decision.verdict == "approve"
    assert r.decision.approved_action == (1.0, -2.0)
    assert r.decision.violated_constraints == ()
    assert r.actuator_command == (1.0, -2.0)
    assert r.reason


# 1. Action limits: clamp or reject per config ------------------------------


def test_out_of_limit_rejected_in_reject_mode(tl):
    k = _kernel(tl, limit_mode="reject")
    r = _check(k, _cmd((25.0, 0.0)))
    assert r.decision.verdict == "reject"
    assert r.decision.approved_action == ()
    assert r.decision.violated_constraints == ("action_limit[x]",)
    assert r.actuator_command == (0.0, 0.0)


def test_out_of_limit_clamped_in_clamp_mode(tl):
    k = _kernel(tl, limit_mode="clamp")
    r = _check(k, _cmd((25.0, -30.0)))
    assert r.decision.verdict == "approve"
    assert r.decision.approved_action == (10.0, -10.0)
    assert r.decision.violated_constraints == ("action_limit[x]", "action_limit[y]")
    assert r.actuator_command == (10.0, -10.0)


def test_limit_boundary_is_inclusive(tl):
    k = _kernel(tl)
    assert _check(k, _cmd((10.0, -10.0))).decision.verdict == "approve"


# 2. Workspace --------------------------------------------------------------


def test_predicted_workspace_exit_rejected(tl):
    k = _kernel(tl)
    # p' = 0.95 + 0.5 * 0.1 + 0.5 * 10 * 0.01 = 1.05 > 1.0
    r = _check(k, _cmd((10.0, 0.0)), pos=(0.95, 0.0), vel=(0.5, 0.0))
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("workspace[x]",)
    # v1.1 safe action brakes: clip(-m v / dt) = clip(-1 * 0.5 / 0.1) = -5.
    assert r.actuator_command == (-5.0, 0.0)


def test_workspace_uses_clamped_action(tl):
    # Clamped force keeps the point inside; the raw force would not.
    k = _kernel(tl, limit_mode="clamp")
    # raw: 0.9 + 0.5*100*0.01 = 1.4 (out); clamped: 0.9 + 0.5*10*0.01 = 0.95 (in)
    r = _check(k, _cmd((100.0, 0.0)), pos=(0.9, 0.0))
    assert r.decision.verdict == "approve"
    assert r.decision.approved_action == (10.0, 0.0)


def test_workspace_exit_in_clamp_mode_lists_all_violations(tl):
    k = _kernel(tl, limit_mode="clamp")
    r = _check(k, _cmd((100.0, 0.0)), pos=(0.99, 0.0), vel=(1.0, 0.0))
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("action_limit[x]", "workspace[x]")


def test_invalid_state_rejected(tl):
    k = _kernel(tl)
    for pos in [(0.0,), (float("nan"), 0.0), None]:
        r = _check(k, _cmd((0.0, 0.0)), pos=pos)
        assert r.decision.verdict == "reject"
        assert r.decision.violated_constraints == ("invalid_state",)


# 3. Stale commands ---------------------------------------------------------


def test_stale_command_rejected(tl):
    k = _kernel(tl)
    r = _check(k, _cmd((0.0, 0.0), ts=T0), now=T0 + 0.051)
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("stale_command",)
    assert r.actuator_command == (0.0, 0.0)
    # within the age limit is accepted
    assert _check(k, _cmd((0.0, 0.0), ts=T0), now=T0 + 0.049).decision.verdict == "approve"


def test_future_command_rejected(tl):
    k = _kernel(tl)
    r = _check(k, _cmd((0.0, 0.0), ts=T0 + 1.0), now=T0)
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("future_command",)


# 4. Malformed commands -----------------------------------------------------


def _malformed_inputs():
    good = json.loads(_cmd((1.0, 1.0)).to_json())
    bad_nan = '{"bad": NaN}'
    wrong_len = _cmd((1.0, 1.0, 1.0))
    missing = dict(good)
    del missing["timestamp"]
    bad_type = json.loads(json.dumps(good))
    bad_type["payload"]["action"] = ["a", "b"]
    obs = Envelope(
        message_id="m", timestamp=T0, source="s", destination="safety", cycle_id=1,
        correlation_id="c",
        payload=Observation(sensor_id="s", channels=(), values=(), uncertainty=Uncertainty()),
    )
    forged = _cmd((1.0, 1.0))
    object.__setattr__(forged.payload, "action", (float("inf"), 0.0))
    return [None, 42, "not json", bad_nan, wrong_len, missing, bad_type, obs, forged,
            json.dumps(missing)]


@pytest.mark.parametrize("cmd", _malformed_inputs())
def test_malformed_command_rejected(tl, cmd):
    k = _kernel(tl)
    r = _check(k, cmd)
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("malformed_command",)
    assert r.actuator_command == (0.0, 0.0)


def test_json_and_dict_commands_accepted(tl):
    k = _kernel(tl)
    env = _cmd((1.0, 2.0))
    assert _check(k, env.to_json()).decision.approved_action == (1.0, 2.0)
    assert _check(k, env.to_dict()).decision.approved_action == (1.0, 2.0)


# 5. Watchdog ---------------------------------------------------------------


def test_watchdog_outputs_safe_action_after_silence(tl):
    clock = Clock()
    k = _kernel(tl, clock=clock)
    assert k.tick(now=T0 + 0.19) is None  # within the timeout: no override
    r = k.tick(now=T0 + 0.21)
    assert r.decision.verdict == "reject"
    assert r.decision.violated_constraints == ("watchdog_timeout",)
    assert r.actuator_command == (0.0, 0.0)


def test_valid_command_feeds_watchdog_rejected_does_not(tl):
    k = _kernel(tl)
    assert _check(k, _cmd((1.0, 0.0), ts=T0 + 0.15), now=T0 + 0.15).decision.verdict == "approve"
    assert k.tick(now=T0 + 0.3) is None
    # Rejected (out-of-limit) commands do not count as valid.
    assert _check(k, _cmd((99.0, 0.0), ts=T0 + 0.3), now=T0 + 0.3).decision.verdict == "reject"
    r = k.tick(now=T0 + 0.36)
    assert r is not None and r.actuator_command == (0.0, 0.0)


def test_custom_safe_action(tl):
    k = _kernel(tl, safe_action=(0.0, -1.0))
    assert k.tick(now=T0 + 1.0).actuator_command == (0.0, -1.0)


# 6. Emergency stop latch ---------------------------------------------------


def test_estop_forces_safe_action_until_operator_reset(tl):
    k = _kernel(tl)
    r = k.emergency_stop("button", now=T0)
    assert r.decision.verdict == "emergency_stop" and r.actuator_command == (0.0, 0.0)
    for i in range(5):
        t = T0 + 0.01 * i
        r = _check(k, _cmd((1.0, 1.0), ts=t), now=t)
        assert r.decision.verdict == "emergency_stop"
        assert r.decision.approved_action == ()
        assert r.actuator_command == (0.0, 0.0)
    assert k.tick(now=T0 + 0.05).decision.verdict == "emergency_stop"
    # Malformed commands during e-stop also yield the e-stop safe action.
    assert _check(k, "garbage").decision.verdict == "emergency_stop"

    with pytest.raises(PermissionError):
        k.reset_emergency_stop("wrong", now=T0 + 0.1)
    assert k.estopped
    assert _check(k, _cmd((1.0, 1.0), ts=T0 + 0.1), now=T0 + 0.1).decision.verdict == "emergency_stop"

    k.reset_emergency_stop(KEY, now=T0 + 0.2)
    assert not k.estopped
    # A command issued before the reset is not replayed.
    old = _check(k, _cmd((1.0, 1.0), ts=T0 + 0.19), now=T0 + 0.2)
    assert old.decision.violated_constraints == ("pre_reset_command",)
    r = _check(k, _cmd((1.0, 1.0), ts=T0 + 0.21), now=T0 + 0.21)
    assert r.decision.verdict == "approve" and r.actuator_command == (1.0, 1.0)


def test_estop_latch_ignores_repeated_trigger_reason(tl):
    k = _kernel(tl)
    k.emergency_stop("first", now=T0)
    r = k.emergency_stop("second", now=T0)
    assert "first" in r.reason


def test_telemetry_failure_latches_estop(tmp_path):
    with TelemetryLog(tmp_path / "small.jsonl", max_bytes=10) as log:
        k = SafetyKernel(_config(), log, KEY, clock=Clock())
        r = _check(k, _cmd((1.0, 1.0)))
    assert r.decision.verdict == "emergency_stop"
    assert r.actuator_command == (0.0, 0.0)
    assert k.estopped and k.telemetry_failures >= 1


def test_unlogged_reset_keeps_estop_latched(tmp_path):
    with TelemetryLog(tmp_path / "capped.jsonl", max_bytes=1) as log:
        k = SafetyKernel(_config(), log, KEY, clock=Clock())
        k.emergency_stop("test", now=T0)
        assert k.reset_emergency_stop(KEY, now=T0 + 0.1) is False
        assert k.estopped
        r = _check(k, _cmd((1.0, 1.0), ts=T0 + 0.1), now=T0 + 0.1)
    assert r.decision.verdict == "emergency_stop"
    assert r.actuator_command == (0.0, 0.0)


def test_logged_reset_returns_true(tl):
    k = _kernel(tl)
    k.emergency_stop("test", now=T0)
    assert k.reset_emergency_stop(KEY, now=T0 + 0.1) is True
    assert not k.estopped


# 7. Immutable limits -------------------------------------------------------


def test_config_is_frozen_and_detached(tl):
    low = [-10.0, -10.0]
    cfg = _config(action_low=low)
    low[0] = -1000.0
    assert cfg.action_low == (-10.0, -10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.action_high = (1e9, 1e9)


def test_kernel_limits_cannot_be_rebound(tl):
    k = _kernel(tl)
    with pytest.raises(AttributeError):
        k.config = _config(action_high=(1e9, 1e9))
    with pytest.raises(AttributeError):
        k._config = _config(action_high=(1e9, 1e9))
    with pytest.raises(AttributeError):
        k._operator_key = "x"
    with pytest.raises(AttributeError):
        del k._config
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.config.action_high = (1e9, 1e9)
    assert _check(k, _cmd((11.0, 0.0))).decision.verdict == "reject"


def test_estop_latch_cannot_be_cleared_by_attribute_write(tl):
    k = _kernel(tl)
    k.emergency_stop("test", now=T0)
    k._estop_reason = None
    assert k.estopped
    r = _check(k, _cmd((1.0, 0.0)))
    assert r.decision.verdict == "emergency_stop"
    assert r.actuator_command == (0.0, 0.0)
    k.reset_emergency_stop(KEY, now=T0 + 1.0)
    assert not k.estopped


@pytest.mark.parametrize(
    "name, value",
    [
        ("_estop_reason", None),
        ("_last_valid_time", 1e12),
        ("_last_reset_time", -1e12),
        ("estopped", False),
        ("_new_attr", 1),
    ],
)
def test_outside_state_write_latches_estop_and_is_logged(tl, name, value):
    k = _kernel(tl)
    setattr(k, name, value)
    assert k.estopped
    assert _check(k, _cmd((1.0, 0.0))).decision.verdict == "emergency_stop"
    assert any(r.decision == "tamper" and name in r.reason for r in tl.records())
    assert not hasattr(k, "_new_attr")


def test_no_public_mutator_api(tl):
    k = _kernel(tl)
    public = {n for n in dir(k) if not n.startswith("_")}
    # v1.2 (ENG-0012) adds no mutator: from_mhs is a constructor, observed_kinematics
    # is a pure read and no_command only emits a (logged) safe-action decision.
    assert public == {
        "check", "tick", "emergency_stop", "reset_emergency_stop",
        "config", "estopped", "telemetry_failures",
        "from_mhs", "observed_kinematics", "no_command",
    }


@pytest.mark.parametrize(
    "kw",
    [
        dict(action_low=(1.0, 0.0), action_high=(0.0, 0.0)),
        dict(workspace_low=(1.0, -1.0), workspace_high=(1.0, 1.0)),
        dict(action_low=(-1.0,)),
        dict(max_command_age_s=0.0),
        dict(watchdog_timeout_s=float("inf")),
        dict(limit_mode="clip"),
        dict(safe_action=(20.0, 0.0)),
        dict(axis_names=("x", "x")),
        dict(mass_kg=-1.0),
    ],
)
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        _config(**kw)


# Logging and determinism ---------------------------------------------------


def test_every_decision_is_logged_with_reason(tl):
    k = _kernel(tl, limit_mode="clamp")
    results = [
        _check(k, _cmd((1.0, 0.0))),
        _check(k, _cmd((50.0, 0.0))),
        _check(k, "garbage"),
        k.tick(now=T0 + 1.0),
        k.emergency_stop("test", now=T0 + 1.0),
    ]
    k.reset_emergency_stop(KEY, now=T0 + 2.0)
    recs = tl.records()
    assert [r.component for r in recs] == ["safety"] * 6
    assert [r.decision for r in recs] == [
        "approve", "approve", "reject", "reject", "emergency_stop", "estop_reset",
    ]
    for res, rec in zip(results, recs):
        assert rec.reason == res.reason and rec.reason
        assert rec.data["violated_constraints"] == list(res.decision.violated_constraints)
        assert rec.data["actuator_command"] == list(res.actuator_command)
    assert recs[1].cycle_id == 1


def test_decisions_are_deterministic(tmp_path):
    def run(name):
        with TelemetryLog(tmp_path / name) as log:
            k = SafetyKernel(_config(limit_mode="clamp"), log, KEY, clock=Clock())
            out = []
            for i in range(20):
                t = T0 + 0.01 * i
                out.append(_check(k, _cmd((3.0 * i - 20.0, 1.0), ts=t), now=t,
                                  pos=(0.05 * i - 0.5, 0.0), vel=(0.1, 0.0)))
            return [(r.decision, r.reason, r.actuator_command) for r in out]

    assert run("a.jsonl") == run("b.jsonl")
