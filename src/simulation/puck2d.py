"""Puck2D: deterministic 2D point-mass environment, EnvA (Gate 1, REQ-SIM; ENG-0006).

State is position ``p`` and velocity ``v`` (2-vectors, metres and m/s). One
:meth:`Puck2D.step` advances ``dt`` seconds (default 0.02 s) with semi-implicit
Euler integration::

    u      = clip(action, force_low, force_high)          # force limits
    f     += alpha * (u - f)  (alpha = 1 - exp(-dt/tau))  # first-order actuator lag;
                                                           # f = u when tau == 0
    v'     = v + J/m + (f - c v) / m * dt                 # J: scheduled impulse
    v_new  = v' - min(mu g dt, |v'|) * v'/|v'|            # Coulomb friction
    p_new  = p + v_new * dt

Coulomb friction is applied at the velocity level: it removes at most
``mu*g*dt`` of speed per step and never reverses the velocity, so a puck at rest
stays at rest while the applied force is below ``mu*m*g`` (stiction) and the
integration does not chatter around zero velocity. ``mu`` is the base
coefficient, or the coefficient of the last friction patch containing ``p`` at
the start of the step.

Task: reach the goal disc (``|p - goal| <= goal_radius`` at the end of a step:
terminated, success) within ``max_steps`` steps (else truncated). Touching an
obstacle (the path segment ``p -> p_new`` comes within ``obstacle radius +
puck_radius`` of its centre; swept, so a fast puck cannot tunnel through)
terminates the episode as a failure; a collision takes precedence over reaching
the goal in the same step. Stepping a finished episode raises
:class:`EpisodeOver`.

Disturbances (scheduled by config; step ``k`` is the transition from step
count ``k`` to ``k + 1``, i.e. the ``k``-th call of :meth:`step` counting from
0): :class:`Impulse` adds ``J/m`` to the velocity at step ``k``;
:class:`MassChange` sets the mass from step ``k`` on; :class:`FrictionPatch`
replaces ``mu`` inside a disc or axis-aligned box.

Observation: ``(pos_x, pos_y, vel_x, vel_y)`` plus independent Gaussian noise
(``pos_noise_std``, ``vel_noise_std``), drawn once per step / reset. Ground
truth and the active disturbances are only in ``info`` (and its copy
:meth:`Puck2D.ground_truth`), which brain modules never receive (G1-4).

Determinism: all randomness (initial-state noise, sensor noise) comes from the
instance's ``numpy.random.Generator``, created by :meth:`reset` from its seed;
the instance holds no global state. Same seed + same actions give bit-identical
trajectories.

Runner integration (:class:`robot.runner.Environment`): :meth:`observe` returns
the latest observation as a contract :class:`Observation` (it draws nothing, so
the runner and :meth:`step` consume the same random stream) and :meth:`actuate`
performs one :meth:`step` with the kernel-approved command and returns an
:class:`Outcome`. The ``rng`` the runner passes is not used: the environment's
own seeded Generator keeps a simulated trajectory independent of how many
draws other modules make. :attr:`Puck2D.time` (``steps * dt``) can serve as
the runner clock offset.

Model Hardware Standard (G1-5, REQ-MHS; ENG-0012): :meth:`Puck2D.mhs` publishes
the body's :class:`contracts.MHS` - a ``point_mass`` with force actuators
``force_x`` / ``force_y`` (action layout in that order), scalar sensors
``pos_x``, ``pos_y``, ``vel_x``, ``vel_y`` (observation layout =
:data:`CHANNELS`), control period ``dt`` and a safety envelope whose mass
bounds contain every mass the body will have. ``Body.mass_kg`` is declared
unknown: the true (possibly changing) mass is ground truth (G1-4).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Union

import numpy as np

from contracts import (
    MHS,
    Actuator,
    Body,
    Control,
    Footprint,
    NoiseModel,
    Observation,
    Outcome,
    SafetyEnvelope,
    Sensor,
    Uncertainty,
)

PUCK2D_VERSION = "puck2d-0.1.0"
CHANNELS = ("pos_x", "pos_y", "vel_x", "vel_y")
AXES = ("x", "y")


class EpisodeOver(RuntimeError):
    """step() was called after the episode terminated or was truncated."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _vec2(value: Any, name: str) -> tuple[float, float]:
    try:
        out = tuple(float(v) for v in value)
    except TypeError:
        raise ValueError(f"{name} must be a pair of numbers, got {value!r}") from None
    if len(out) != 2 or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be two finite numbers, got {value!r}")
    return out  # type: ignore[return-value]


def _num(value: Any, name: str, *, low: float = -math.inf, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < low or (strict and v == low):
        bound = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {bound} {low}, got {value!r}")
    return v


def _step_index(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


@dataclass(frozen=True)
class Obstacle:
    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec2(self.center, "Obstacle.center"))
        object.__setattr__(self, "radius", _num(self.radius, "Obstacle.radius", low=0.0, strict=True))


@dataclass(frozen=True)
class Impulse:
    """Adds ``impulse / mass`` (N s / kg) to the velocity at step ``step``."""

    step: int
    impulse: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _step_index(self.step, "Impulse.step"))
        object.__setattr__(self, "impulse", _vec2(self.impulse, "Impulse.impulse"))


@dataclass(frozen=True)
class MassChange:
    """Sets the mass to ``mass`` from step ``step`` on."""

    step: int
    mass: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _step_index(self.step, "MassChange.step"))
        object.__setattr__(self, "mass", _num(self.mass, "MassChange.mass", low=0.0, strict=True))


@dataclass(frozen=True)
class FrictionPatch:
    """Region with Coulomb coefficient ``friction``: a disc (``center``, ``radius``)
    or an axis-aligned box (``low``, ``high``). Boundaries are inside."""

    friction: float
    center: Optional[tuple[float, float]] = None
    radius: Optional[float] = None
    low: Optional[tuple[float, float]] = None
    high: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "friction", _num(self.friction, "FrictionPatch.friction", low=0.0))
        disc = self.center is not None or self.radius is not None
        box = self.low is not None or self.high is not None
        if disc == box:
            raise ValueError("FrictionPatch needs either center+radius or low+high")
        if disc:
            if self.center is None or self.radius is None:
                raise ValueError("FrictionPatch disc needs both center and radius")
            object.__setattr__(self, "center", _vec2(self.center, "FrictionPatch.center"))
            object.__setattr__(
                self, "radius", _num(self.radius, "FrictionPatch.radius", low=0.0, strict=True)
            )
        else:
            if self.low is None or self.high is None:
                raise ValueError("FrictionPatch box needs both low and high")
            lo = _vec2(self.low, "FrictionPatch.low")
            hi = _vec2(self.high, "FrictionPatch.high")
            if not all(a < b for a, b in zip(lo, hi)):
                raise ValueError(f"FrictionPatch box needs low < high, got {lo}, {hi}")
            object.__setattr__(self, "low", lo)
            object.__setattr__(self, "high", hi)

    def contains(self, x: float, y: float) -> bool:
        if self.center is not None:
            return math.hypot(x - self.center[0], y - self.center[1]) <= self.radius
        return self.low[0] <= x <= self.high[0] and self.low[1] <= y <= self.high[1]


def _items(value: Any, cls: type, name: str) -> tuple:
    out = []
    for i, item in enumerate(value):
        if isinstance(item, cls):
            out.append(item)
        elif isinstance(item, Mapping):
            out.append(cls(**item))
        else:
            raise ValueError(f"{name}[{i}] must be {cls.__name__} or a mapping, got {item!r}")
    return tuple(out)


@dataclass(frozen=True)
class Puck2DConfig:
    """All Puck2D parameters (SI units). Validated on construction."""

    mass: float = 1.0                  # kg
    damping: float = 0.0               # viscous coefficient c, N s/m
    friction: float = 0.0              # Coulomb coefficient mu (dimensionless)
    gravity: float = 9.81              # m/s^2, normal force = m * g
    dt: float = 0.02                   # s
    force_low: tuple[float, float] = (-10.0, -10.0)   # N
    force_high: tuple[float, float] = (10.0, 10.0)    # N
    actuator_tau: float = 0.0          # s, first-order lag time constant; 0 = none
    start_pos: tuple[float, float] = (0.0, 0.0)
    start_vel: tuple[float, float] = (0.0, 0.0)
    start_pos_std: float = 0.0         # Gaussian noise on the initial position
    goal: tuple[float, float] = (1.0, 0.0)
    goal_radius: float = 0.05
    puck_radius: float = 0.0           # used for obstacle collision only
    obstacles: tuple[Obstacle, ...] = ()
    max_steps: int = 500
    pos_noise_std: float = 0.0
    vel_noise_std: float = 0.0
    impulses: tuple[Impulse, ...] = ()
    mass_changes: tuple[MassChange, ...] = ()
    friction_patches: tuple[FrictionPatch, ...] = ()
    goal_bonus: float = 10.0           # reward on reaching the goal
    collision_penalty: float = 10.0    # subtracted from the reward on collision

    def __post_init__(self) -> None:
        s = lambda n, v: object.__setattr__(self, n, v)  # noqa: E731
        s("mass", _num(self.mass, "mass", low=0.0, strict=True))
        for n in ("damping", "friction", "actuator_tau", "start_pos_std", "goal_radius",
                  "puck_radius", "pos_noise_std", "vel_noise_std", "goal_bonus",
                  "collision_penalty", "gravity"):
            s(n, _num(getattr(self, n), n, low=0.0))
        s("dt", _num(self.dt, "dt", low=0.0, strict=True))
        for n in ("force_low", "force_high", "start_pos", "start_vel", "goal"):
            s(n, _vec2(getattr(self, n), n))
        if not all(lo <= hi for lo, hi in zip(self.force_low, self.force_high)):
            raise ValueError(f"force_low must be <= force_high, got {self.force_low}, {self.force_high}")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError(f"max_steps must be an int >= 1, got {self.max_steps!r}")
        s("obstacles", _items(self.obstacles, Obstacle, "obstacles"))
        s("impulses", _items(self.impulses, Impulse, "impulses"))
        s("mass_changes", _items(self.mass_changes, MassChange, "mass_changes"))
        s("friction_patches", _items(self.friction_patches, FrictionPatch, "friction_patches"))
        steps = [m.step for m in self.mass_changes]
        if len(steps) != len(set(steps)):
            raise ValueError(f"at most one mass change per step, got steps {steps}")
        # Explicit damping update v *= (1 - c dt / m) must stay a decay for every mass used.
        for m in (self.mass, *(mc.mass for mc in self.mass_changes)):
            if self.damping * self.dt >= m:
                raise ValueError(
                    f"damping * dt must be < mass for stable integration, got "
                    f"{self.damping} * {self.dt} >= {m}"
                )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Puck2DConfig":
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown Puck2D config key(s) {unknown}")
        return cls(**raw)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class Puck2D:
    """Deterministic 2D point-mass reach task; implements robot.runner.Environment."""

    version = PUCK2D_VERSION

    def __init__(
        self, config: Union[Puck2DConfig, Mapping[str, Any], None] = None, seed: int = 0
    ) -> None:
        if config is None:
            config = Puck2DConfig()
        elif isinstance(config, Mapping):
            config = Puck2DConfig.from_mapping(config)
        elif not isinstance(config, Puck2DConfig):
            raise TypeError(f"config must be Puck2DConfig or a mapping, got {type(config).__name__}")
        self.config = config
        cfg = config
        self._lo = np.array(cfg.force_low)
        self._hi = np.array(cfg.force_high)
        self._goal = np.array(cfg.goal)
        self._alpha = 1.0 if cfg.actuator_tau == 0.0 else 1.0 - math.exp(-cfg.dt / cfg.actuator_tau)
        self._noise_std = np.array(
            [cfg.pos_noise_std, cfg.pos_noise_std, cfg.vel_noise_std, cfg.vel_noise_std]
        )
        self._impulses: dict[int, list[Impulse]] = {}
        for imp in cfg.impulses:
            self._impulses.setdefault(imp.step, []).append(imp)
        self._mass_changes = {mc.step: mc for mc in cfg.mass_changes}
        self.reset(seed)

    # -- gym-style API -----------------------------------------------------------

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode; returns ``(obs, info)``."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        cfg = self.config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pos = np.array(cfg.start_pos) + self.rng.normal(0.0, 1.0, 2) * cfg.start_pos_std
        self.vel = np.array(cfg.start_vel)
        self.force = np.zeros(2)  # force actually applied (after limits and lag)
        self.mass = cfg.mass
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.goal_reached = False
        self.collision: Optional[int] = None
        self._obs = self._draw_obs()
        info = self._info([], cfg.friction, None)
        self._last_info = copy.deepcopy(info)
        return self._obs.copy(), info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a 2D force for one ``dt``; returns ``(obs, reward, terminated, truncated, info)``."""
        if self.terminated or self.truncated:
            raise EpisodeOver("episode is over; call reset(seed)")
        u = np.asarray(action, dtype=float)
        if u.shape != (2,) or not np.all(np.isfinite(u)):
            raise ValueError(f"action must be 2 finite numbers, got {action!r}")
        cfg = self.config
        k = self.steps
        active: list[dict[str, Any]] = []

        mc = self._mass_changes.get(k)
        if mc is not None:
            self.mass = mc.mass
            active.append({"type": "mass_change", "step": k, "mass": mc.mass})

        mu = cfg.friction
        patch_index = None
        for i, patch in enumerate(cfg.friction_patches):
            if patch.contains(float(self.pos[0]), float(self.pos[1])):
                mu, patch_index = patch.friction, i
        if patch_index is not None:
            active.append({"type": "friction_patch", "index": patch_index, "friction": mu})

        u_clipped = np.minimum(np.maximum(u, self._lo), self._hi)
        clipped = bool(np.any(u_clipped != u))
        self.force = self.force + self._alpha * (u_clipped - self.force)

        m = self.mass
        v = self.vel + (self.force - cfg.damping * self.vel) * (cfg.dt / m)
        for imp in self._impulses.get(k, ()):
            v = v + np.array(imp.impulse) / m
            active.append({"type": "impulse", "step": k, "impulse": list(imp.impulse)})
        if mu > 0.0:
            speed = math.hypot(v[0], v[1])
            if speed > 0.0:
                v = v * (max(0.0, speed - mu * cfg.gravity * cfg.dt) / speed)
        p_old = self.pos
        self.vel = v
        self.pos = p_old + v * cfg.dt
        self.steps = k + 1

        self.collision = self._first_collision(p_old, self.pos)
        dist = float(np.hypot(*(self.pos - self._goal)))
        self.goal_reached = self.collision is None and dist <= cfg.goal_radius
        self.terminated = self.collision is not None or self.goal_reached
        self.truncated = not self.terminated and self.steps >= cfg.max_steps

        reward = -dist
        if self.goal_reached:
            reward += cfg.goal_bonus
        if self.collision is not None:
            reward -= cfg.collision_penalty

        self._obs = self._draw_obs()
        info = self._info(active, mu, u_clipped)
        info["action_clipped"] = clipped
        self._last_info = copy.deepcopy(info)
        return self._obs.copy(), float(reward), self.terminated, self.truncated, info

    @property
    def time(self) -> float:
        """Simulated time of the current state, ``steps * dt`` seconds."""
        return self.steps * self.config.dt

    def ground_truth(self) -> dict[str, Any]:
        """A copy of the latest ``info`` (true state, parameters, active disturbances).

        For harness metrics and telemetry only (G1-4): the cycle runner logs it
        under a ``ground_truth`` key and never passes it to a brain module.
        """
        return copy.deepcopy(self._last_info)

    # -- Model Hardware Standard ---------------------------------------------------

    def mhs(
        self,
        *,
        workspace_low: tuple[float, float] = (-2.0, -2.0),
        workspace_high: tuple[float, float] = (2.0, 2.0),
        max_command_age_s: float = 0.05,
        watchdog_timeout_s: float = 0.1,
        mass_bounds: Optional[tuple[float, float]] = None,
    ) -> MHS:
        """This body's MHS. The workspace and command timing are deployment
        settings, hence parameters. ``mass_bounds`` (low, high) must contain the
        configured mass and every scheduled mass change (default: exactly their
        min and max); a fleet-level bound avoids revealing a task's schedule."""
        cfg = self.config
        masses = [cfg.mass, *(mc.mass for mc in cfg.mass_changes)]
        lo_m, hi_m = (min(masses), max(masses)) if mass_bounds is None else (
            _num(mass_bounds[0], "mass_bounds[0]", low=0.0, strict=True),
            _num(mass_bounds[1], "mass_bounds[1]", low=0.0, strict=True),
        )
        if not lo_m <= min(masses) <= max(masses) <= hi_m:
            raise ValueError(f"mass_bounds {(lo_m, hi_m)} do not contain the masses {masses}")
        rate = 1.0 / cfg.dt

        def noise(std: float) -> NoiseModel:
            return NoiseModel("gaussian", std) if std > 0 else NoiseModel("none", None)

        sensors = tuple(
            Sensor(f"{q}_{a}", kind, units, (), rate, noise(std), "world", a)
            for q, kind, units, std in (
                ("pos", "position", "m", cfg.pos_noise_std),
                ("vel", "velocity", "m/s", cfg.vel_noise_std),
            )
            for a in AXES
        )
        actuators = tuple(
            Actuator(f"force_{a}", "force", "N", a, "world", cfg.force_low[i], cfg.force_high[i],
                     None, cfg.actuator_tau)
            for i, a in enumerate(AXES)
        )
        return MHS(
            body_name="puck2d",
            body_class="point_mass",
            actuators=actuators,
            action_layout=tuple(a.name for a in actuators),
            sensors=sensors,
            observation_layout=CHANNELS,
            body=Body(
                mass_kg=None,
                inertia_kgm2=None,
                footprint=(
                    Footprint("circle", (cfg.puck_radius,)) if cfg.puck_radius > 0
                    else Footprint("point", ())
                ),
                frames=("world", "base"),
                base_frame="base",
            ),
            control=Control(
                period_s=cfg.dt,
                max_command_age_s=float(max_command_age_s),
                watchdog_timeout_s=float(watchdog_timeout_s),
                max_cycle_latency_s=cfg.dt,
                reflexes=("force_saturation",),
            ),
            safety=SafetyEnvelope(
                workspace_frame="world",
                workspace_axes=AXES,
                workspace_low=tuple(float(v) for v in workspace_low),
                workspace_high=tuple(float(v) for v in workspace_high),
                speed_limits=None,
                mass_kg=hi_m,
                mass_lower_bound_kg=None if lo_m == hi_m else lo_m,
                braking="actuators",
                brake_decel_mps2=None,
                safe_action=(0.0, 0.0),
                estop_latching=True,
                estop_reset="operator",
                estop_action="brake",
            ),
        )

    # -- robot.runner.Environment ------------------------------------------------

    def observe(self, rng: np.random.Generator) -> Observation:
        return Observation(
            "puck2d", CHANNELS, tuple(float(x) for x in self._obs), self._uncertainty()
        )

    def actuate(self, command: tuple[float, ...], rng: np.random.Generator) -> Outcome:
        obs, reward, _, _, _ = self.step(command)
        return Outcome(
            f"puck2d-s{self.seed}-k{self.steps}",
            self.goal_reached,
            reward,
            CHANNELS,
            tuple(float(x) for x in obs),
            self._uncertainty(),
        )

    # -- internals ---------------------------------------------------------------

    def _draw_obs(self) -> np.ndarray:
        # Always draw 4 normals so the random stream does not depend on the noise level.
        noise = self.rng.normal(0.0, 1.0, 4) * self._noise_std
        return np.concatenate([self.pos, self.vel]) + noise

    def _uncertainty(self) -> Uncertainty:
        return Uncertainty(measurement=max(self.config.pos_noise_std, self.config.vel_noise_std))

    def _first_collision(self, a: np.ndarray, b: np.ndarray) -> Optional[int]:
        """Index of the first obstacle the segment a->b touches, else None."""
        d = b - a
        dd = float(d @ d)
        for i, ob in enumerate(self.config.obstacles):
            c = np.array(ob.center)
            t = 0.0 if dd == 0.0 else min(1.0, max(0.0, float((c - a) @ d) / dd))
            closest = a + t * d
            if math.hypot(*(closest - c)) <= ob.radius + self.config.puck_radius:
                return i
        return None

    def _info(self, active: list, mu: float, u_clipped: Optional[np.ndarray]) -> dict[str, Any]:
        return {
            "step": self.steps,
            "time": self.time,
            "pos": [float(x) for x in self.pos],
            "vel": [float(x) for x in self.vel],
            "applied_force": [float(x) for x in self.force],
            "commanded_force": None if u_clipped is None else [float(x) for x in u_clipped],
            "mass": self.mass,
            "friction": mu,
            "active_disturbances": active,
            "goal_reached": self.goal_reached,
            "collision": self.collision,
        }
