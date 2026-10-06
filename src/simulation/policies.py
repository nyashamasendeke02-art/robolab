"""Scripted reference policies for Puck2D (Gate 1, REQ-SIM; ENG-0008).

Both policies implement :class:`robot.runner.System1`: they read the physical
state (``pos_x, pos_y, vel_x, vel_y``) from the :class:`StateUpdate` and return
an :class:`ActionProposal` holding a 2D force. They never actuate anything: the
runner sends the accepted proposal to the Safety Kernel.

* :class:`RandomPolicy` - force drawn uniformly in ``[low, high]`` per axis from
  the ``rng`` the runner passes (the instrument's "should mostly fail" sanity
  baseline).
* :class:`PDController` - ``u = kp * (goal - p) - kd * v``, saturated per axis
  to ``[-max_force, max_force]`` so a large error does not produce a command
  outside the kernel's action limits. Gains come from configuration (e.g.
  ``configs/policies.json`` via :meth:`robot.runner.Modules.from_config`).

The harness calls :meth:`reset` with the task at the start of every episode;
it sets the goal (PD) and restarts the proposal counter so proposal ids are a
function of the episode only.

``confidence`` is a fixed, uncalibrated constant (1.0 for the hand-designed PD
law, 0.0 for the random policy); nothing downstream may treat it as calibrated.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from contracts import ActionProposal, StateUpdate, Uncertainty

RANDOM_POLICY_VERSION = "random-policy-0.1.0"
PD_CONTROLLER_VERSION = "pd-controller-0.1.0"
STATE_VARS = ("pos_x", "pos_y", "vel_x", "vel_y")


def _physical(state: StateUpdate) -> Optional[tuple[float, float, float, float]]:
    """``(px, py, vx, vy)`` from a physical-layer state, else None."""
    if state.layer != "physical":
        return None
    values = dict(zip(state.variables, state.values))
    if not all(v in values for v in STATE_VARS):
        return None
    out = tuple(float(values[v]) for v in STATE_VARS)
    if not all(math.isfinite(x) for x in out):
        return None
    return out  # type: ignore[return-value]


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < 0 or (positive and v == 0):
        raise ValueError(f"{name} must be finite and {'>' if positive else '>='} 0, got {value!r}")
    return v


class _Policy:
    version = "policy"
    name = "policy"
    confidence = 0.0

    def __init__(self) -> None:
        self._n = 0

    def reset(self, task: Any = None) -> None:
        self._n = 0

    def _proposal(self, action: tuple[float, float]) -> ActionProposal:
        self._n += 1
        return ActionProposal(
            proposal_id=f"{self.name}-{self._n}",
            policy_version=self.version,
            action=(float(action[0]), float(action[1])),
            confidence=self.confidence,
            uncertainty=Uncertainty(),
        )


class RandomPolicy(_Policy):
    """Uniform random force in ``[low, high]`` per axis."""

    version = RANDOM_POLICY_VERSION
    name = "random"
    confidence = 0.0

    def __init__(
        self,
        low: tuple[float, float] = (-10.0, -10.0),
        high: tuple[float, float] = (10.0, 10.0),
    ) -> None:
        super().__init__()
        lo = tuple(float(x) for x in low)
        hi = tuple(float(x) for x in high)
        if len(lo) != 2 or len(hi) != 2 or not all(math.isfinite(x) for x in lo + hi):
            raise ValueError(f"low/high must be two finite numbers each, got {low}, {high}")
        if not all(a <= b for a, b in zip(lo, hi)):
            raise ValueError(f"low must be <= high, got {lo}, {hi}")
        self.low, self.high = lo, hi

    def propose(self, state, prediction, rng: np.random.Generator) -> Optional[ActionProposal]:
        u = rng.uniform(self.low, self.high)
        return self._proposal((u[0], u[1]))


class PDController(_Policy):
    """``u = kp (goal - p) - kd v``, saturated to ``max_force`` per axis."""

    version = PD_CONTROLLER_VERSION
    name = "pd"
    confidence = 1.0

    def __init__(
        self,
        kp: float = 20.0,
        kd: float = 8.0,
        max_force: float = 10.0,
        goal: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        self.kp = _finite(kp, "kp")
        self.kd = _finite(kd, "kd")
        self.max_force = _finite(max_force, "max_force", positive=True)
        self.goal: Optional[tuple[float, float]] = None if goal is None else self._goal(goal)

    @staticmethod
    def _goal(goal: Any) -> tuple[float, float]:
        g = tuple(float(x) for x in goal)
        if len(g) != 2 or not all(math.isfinite(x) for x in g):
            raise ValueError(f"goal must be two finite numbers, got {goal!r}")
        return g  # type: ignore[return-value]

    def reset(self, task: Any = None) -> None:
        super().reset(task)
        if task is not None:
            self.goal = self._goal(task.goal)

    def propose(self, state, prediction, rng: np.random.Generator) -> Optional[ActionProposal]:
        s = _physical(state)
        if s is None or self.goal is None:
            return None  # no usable state or no goal: abstain
        px, py, vx, vy = s
        f = self.max_force
        ux = min(f, max(-f, self.kp * (self.goal[0] - px) - self.kd * vx))
        uy = min(f, max(-f, self.kp * (self.goal[1] - py) - self.kd * vy))
        return self._proposal((ux, uy))
