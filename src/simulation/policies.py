"""Scripted reference policies for Puck2D (Gate 1, REQ-SIM, REQ-S1; ENG-0009).

Both policies implement :class:`robot.runner.System1` (they plug into the cycle
runner's ``system1`` slot) plus ``reset(task)``, which the episode harness calls
before every episode:

* :class:`RandomPolicy` - a uniform random 2D force in ``[force_low, force_high]``
  per axis, drawn from the runner's Generator (the instrument's lower bound);
* :class:`PDController` - ``u = kp (goal - p) - kd v`` per axis, saturated to
  ``+-force_limit``; gains come from :class:`PDConfig` (e.g.
  ``configs/pd_controller.json``).

Both read ``pos_<axis>`` / ``vel_<axis>`` from a physical-layer
:class:`StateUpdate` and abstain (``None``) when they are missing or non-finite.
``confidence`` is a fixed placeholder (random 0.0, PD 1.0); it is not calibrated.
Proposal ids are ``<name>-<n>`` with ``n`` counted from ``reset``, so they are
deterministic.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from contracts import ActionProposal, StateUpdate, Uncertainty

RANDOM_POLICY_VERSION = "random-policy-0.1.0"
PD_CONTROLLER_VERSION = "pd-controller-0.1.0"
AXES = ("x", "y")


def _kinematics(state: StateUpdate) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if state.layer != "physical":
        return None
    values = dict(zip(state.variables, state.values))
    try:
        pos = np.array([values[f"pos_{a}"] for a in AXES], dtype=float)
        vel = np.array([values[f"vel_{a}"] for a in AXES], dtype=float)
    except KeyError:
        return None
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
        return None
    return pos, vel


class RandomPolicy:
    """Uniform random force per axis; ignores the state."""

    version = RANDOM_POLICY_VERSION

    def __init__(
        self,
        force_low: Sequence[float] = (-10.0, -10.0),
        force_high: Sequence[float] = (10.0, 10.0),
    ) -> None:
        self.force_low = tuple(float(v) for v in force_low)
        self.force_high = tuple(float(v) for v in force_high)
        if len(self.force_low) != 2 or len(self.force_high) != 2 or not all(
            math.isfinite(lo) and math.isfinite(hi) and lo <= hi
            for lo, hi in zip(self.force_low, self.force_high)
        ):
            raise ValueError(f"bad force limits {self.force_low}, {self.force_high}")
        self._n = 0

    def reset(self, task: Any = None) -> None:
        self._n = 0

    def propose(self, state, prediction, rng: np.random.Generator) -> Optional[ActionProposal]:
        u = rng.uniform(self.force_low, self.force_high)
        self._n += 1
        return ActionProposal(
            proposal_id=f"random-{self._n}",
            policy_version=self.version,
            action=tuple(float(x) for x in u),
            confidence=0.0,
            uncertainty=Uncertainty(),
        )


@dataclass(frozen=True)
class PDConfig:
    """PD gains (per axis, same on both axes) and output saturation (N)."""

    kp: float = 16.0          # N/m
    kd: float = 8.0           # N s/m; 2*sqrt(kp*m) = critical damping for m = 1 kg
    force_limit: float = 10.0  # N, |u_i| <= force_limit

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"PDConfig.{f.name} must be a finite number >= 0, got {v!r}")
            object.__setattr__(self, f.name, float(v))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PDConfig":
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown PD config key(s) {unknown}")
        return cls(**raw)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "PDConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_mapping(json.load(fh))


class PDController:
    """Proportional-derivative force toward the task goal."""

    version = PD_CONTROLLER_VERSION

    def __init__(
        self,
        config: Union[PDConfig, Mapping[str, Any], None] = None,
        goal: Sequence[float] = (1.0, 0.0),
    ) -> None:
        if config is None:
            config = PDConfig()
        elif isinstance(config, Mapping):
            config = PDConfig.from_mapping(config)
        elif not isinstance(config, PDConfig):
            raise TypeError(f"config must be PDConfig or a mapping, got {type(config).__name__}")
        self.config = config
        self._set_goal(goal)
        self._n = 0

    def _set_goal(self, goal: Sequence[float]) -> None:
        g = np.array([float(v) for v in goal])
        if g.shape != (2,) or not np.all(np.isfinite(g)):
            raise ValueError(f"goal must be two finite numbers, got {goal!r}")
        self.goal = g

    def reset(self, task: Any = None) -> None:
        """Start an episode; takes the goal from ``task.goal`` when given."""
        if task is not None:
            self._set_goal(task.goal)
        self._n = 0

    def propose(self, state, prediction, rng: np.random.Generator) -> Optional[ActionProposal]:
        kin = _kinematics(state)
        if kin is None:
            return None
        pos, vel = kin
        c = self.config
        u = c.kp * (self.goal - pos) - c.kd * vel
        u = np.clip(u, -c.force_limit, c.force_limit)
        self._n += 1
        return ActionProposal(
            proposal_id=f"pd-{self._n}",
            policy_version=self.version,
            action=tuple(float(x) for x in u),
            confidence=1.0,
            uncertainty=Uncertainty(),
        )
