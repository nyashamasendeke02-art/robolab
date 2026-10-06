"""Model Hardware Standard (MHS) v0 (Gate 1, REQ-MHS, REQ-SAFE+, H5, ADR-001, ADR-003, ADR-005; ENG-0012).

The brain is the same code for every body; what differs is an :class:`MHS`, a
declared, machine-readable description of the body: identity, actuators and the
action-vector layout the brain must produce, sensors and the observation
layout, body properties, control timing and the safety envelope. The Safety
Kernel is configured from it (``SafetyConfig.from_mhs``) and the cycle runner
hands it to brain modules, which read action / observation layouts only from it.

Built on the message-contract machinery (:mod:`contracts.messages`): every class
is a frozen dataclass validated on construction (``validate()`` raises
:class:`ContractError`), JSON round-trips exactly
(``MHS.from_json(m.to_json()) == m``), floats are strict and finite, unknown
or missing JSON keys are rejected. "Unknown" body properties are ``None``
(JSON ``null``). See ``docs/mhs.md``; any change to a class or field here must
bump :data:`MHS_VERSION`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from contracts.messages import ContractError, _Contract, _non_empty, _one_of, _require

MHS_VERSION = "0.1.0"

BODY_CLASSES = ("point_mass", "wheeled", "legged", "manipulator", "aerial")
ACTUATOR_UNITS: dict[str, tuple[str, ...]] = {
    "force": ("N",),
    "torque": ("N*m",),
    "velocity": ("m/s", "rad/s"),
    "steering": ("rad",),
}
ACTUATOR_KINDS = tuple(ACTUATOR_UNITS)
SENSOR_UNITS: dict[str, tuple[str, ...]] = {
    "position": ("m",),
    "velocity": ("m/s",),
    "acceleration": ("m/s^2",),
    "orientation": ("rad",),
    "angular_velocity": ("rad/s",),
    "joint_position": ("rad", "m"),
    "joint_velocity": ("rad/s", "m/s"),
    "force": ("N",),
    "torque": ("N*m",),
}
SENSOR_KINDS = tuple(SENSOR_UNITS)
NOISE_KINDS = ("none", "gaussian", "unknown")
FOOTPRINT_DIMS: dict[str, int] = {"point": 0, "circle": 1, "box": 2}
FOOTPRINT_KINDS = tuple(FOOTPRINT_DIMS)
BRAKING_MODES = ("actuators", "none")
ESTOP_RESETS = ("operator",)
ESTOP_ACTIONS = ("brake", "safe_action", "power_off")


def _unique(values: tuple[str, ...], what: str) -> None:
    _require(len(set(values)) == len(values), f"{what} must be unique, got {list(values)}")
    _require(all(v != "" for v in values), f"{what} must be non-empty strings")


def _permutation(layout: tuple[str, ...], names: tuple[str, ...], what: str) -> None:
    _unique(layout, what)
    missing = sorted(set(names) - set(layout))
    unknown = sorted(set(layout) - set(names))
    _require(
        not missing and not unknown,
        f"{what} inconsistent: missing {missing}, unknown {unknown}",
    )


@dataclass(frozen=True)
class NoiseModel(_Contract):
    """Sensor noise: ``gaussian`` with standard deviation ``std`` (sensor units),
    ``none`` (exact readings) or ``unknown`` (``std`` must be ``None``)."""

    kind: str
    std: Optional[float]

    def _validate_semantics(self) -> None:
        _one_of(self, "kind", NOISE_KINDS)
        if self.kind == "gaussian":
            _require(self.std is not None and self.std >= 0, "NoiseModel: gaussian needs std >= 0")
        else:
            _require(self.std is None, f"NoiseModel: kind {self.kind!r} must not set std")


@dataclass(frozen=True)
class Actuator(_Contract):
    """One action-vector component. ``axis`` is the Cartesian axis of ``frame`` it
    acts along (``None`` if it has none, e.g. a joint)."""

    name: str
    kind: str
    units: str
    axis: Optional[str]
    frame: str
    low: float
    high: float
    rate_limit: Optional[float]  # units per second; None = no limit
    latency_s: float

    def _validate_semantics(self) -> None:
        _non_empty(self, "name", "frame")
        _one_of(self, "kind", ACTUATOR_KINDS)
        _require(
            self.units in ACTUATOR_UNITS[self.kind],
            f"Actuator {self.name}: unknown units {self.units!r} for kind {self.kind!r} "
            f"(allowed {list(ACTUATOR_UNITS[self.kind])})",
        )
        _require(self.axis is None or self.axis != "", f"Actuator {self.name}: empty axis")
        _require(self.low <= self.high, f"Actuator {self.name}: low > high")
        _require(
            self.rate_limit is None or self.rate_limit > 0,
            f"Actuator {self.name}: rate_limit must be > 0",
        )
        _require(self.latency_s >= 0, f"Actuator {self.name}: latency_s must be >= 0")


@dataclass(frozen=True)
class Sensor(_Contract):
    """One sensor. It yields ``prod(shape)`` observation channels, named ``name``
    for a scalar (``shape == ()``) and ``name[i]`` (row-major) otherwise."""

    name: str
    kind: str
    units: str
    shape: tuple[int, ...]
    rate_hz: float
    noise: NoiseModel
    frame: str
    axis: Optional[str]

    def _validate_semantics(self) -> None:
        _non_empty(self, "name", "frame")
        _one_of(self, "kind", SENSOR_KINDS)
        _require(
            self.units in SENSOR_UNITS[self.kind],
            f"Sensor {self.name}: unknown units {self.units!r} for kind {self.kind!r} "
            f"(allowed {list(SENSOR_UNITS[self.kind])})",
        )
        _require(all(d >= 1 for d in self.shape), f"Sensor {self.name}: shape entries must be >= 1")
        _require(self.rate_hz > 0, f"Sensor {self.name}: rate_hz must be > 0")
        _require(self.axis is None or self.axis != "", f"Sensor {self.name}: empty axis")

    @property
    def channels(self) -> tuple[str, ...]:
        n = math.prod(self.shape)
        return (self.name,) if self.shape == () else tuple(f"{self.name}[{i}]" for i in range(n))


@dataclass(frozen=True)
class Footprint(_Contract):
    """Geometry in the base frame: ``point`` (no dims), ``circle`` (radius) or
    ``box`` (length x, length y), metres."""

    kind: str
    dims: tuple[float, ...]

    def _validate_semantics(self) -> None:
        _one_of(self, "kind", FOOTPRINT_KINDS)
        n = FOOTPRINT_DIMS[self.kind]
        _require(len(self.dims) == n, f"Footprint {self.kind}: expected {n} dims, got {len(self.dims)}")
        _require(all(d > 0 for d in self.dims), "Footprint dims must be > 0")


@dataclass(frozen=True)
class Body(_Contract):
    """Physical body. ``mass_kg`` / ``inertia_kgm2`` are ``None`` when unknown;
    inertia is principal moments (3) or a row-major 3x3 tensor (9)."""

    mass_kg: Optional[float]
    inertia_kgm2: Optional[tuple[float, ...]]
    footprint: Footprint
    frames: tuple[str, ...]
    base_frame: str

    def _validate_semantics(self) -> None:
        _require(self.mass_kg is None or self.mass_kg > 0, "Body.mass_kg must be > 0 or unknown")
        if self.inertia_kgm2 is not None:
            _require(len(self.inertia_kgm2) in (3, 9), "Body.inertia_kgm2 needs 3 or 9 values")
            if len(self.inertia_kgm2) == 3:
                _require(all(i >= 0 for i in self.inertia_kgm2), "Body principal inertia must be >= 0")
        _require(len(self.frames) > 0, "Body.frames must not be empty")
        _unique(self.frames, "Body.frames")
        _require(self.base_frame in self.frames, f"Body.base_frame {self.base_frame!r} not in frames")


@dataclass(frozen=True)
class Control(_Contract):
    """Control timing and the low-level reflexes the body runs itself, below the
    embodiment adapter (e.g. ``force_saturation``)."""

    period_s: float
    max_command_age_s: float
    watchdog_timeout_s: float
    max_cycle_latency_s: float
    reflexes: tuple[str, ...]

    def _validate_semantics(self) -> None:
        for n in ("period_s", "max_command_age_s", "watchdog_timeout_s", "max_cycle_latency_s"):
            _require(getattr(self, n) > 0, f"Control.{n} must be > 0")
        _require(
            self.watchdog_timeout_s >= self.period_s,
            "Control.watchdog_timeout_s must be >= period_s",
        )
        _require(
            self.max_cycle_latency_s <= self.period_s,
            "Control.max_cycle_latency_s must be <= period_s",
        )
        _unique(self.reflexes, "Control.reflexes")


@dataclass(frozen=True)
class SafetyEnvelope(_Contract):
    """Operating domain and stopping capability, the Safety Kernel's configuration.

    ``workspace_*`` / ``speed_limits`` / ``brake_decel_mps2`` are per entry of
    ``workspace_axes`` (axes of ``workspace_frame``). ``mass_kg`` is an upper
    bound and ``mass_lower_bound_kg`` a lower bound on the true mass (``None``:
    the mass is exactly ``mass_kg``). ``safe_action`` follows the action layout.
    """

    workspace_frame: str
    workspace_axes: tuple[str, ...]
    workspace_low: tuple[float, ...]
    workspace_high: tuple[float, ...]
    speed_limits: Optional[tuple[float, ...]]  # m/s, None = not declared
    mass_kg: float
    mass_lower_bound_kg: Optional[float]
    braking: str
    brake_decel_mps2: Optional[tuple[float, ...]]  # None = as the actuators allow
    safe_action: tuple[float, ...]
    estop_latching: bool
    estop_reset: str
    estop_action: str

    def _validate_semantics(self) -> None:
        _non_empty(self, "workspace_frame")
        _require(len(self.workspace_axes) > 0, "SafetyEnvelope.workspace_axes must not be empty")
        _unique(self.workspace_axes, "SafetyEnvelope.workspace_axes")
        n = len(self.workspace_axes)
        for name in ("workspace_low", "workspace_high", "speed_limits", "brake_decel_mps2"):
            v = getattr(self, name)
            _require(v is None or len(v) == n, f"SafetyEnvelope.{name} needs {n} values")
        _require(
            all(lo < hi for lo, hi in zip(self.workspace_low, self.workspace_high)),
            "SafetyEnvelope: workspace_low must be < workspace_high",
        )
        _require(
            self.speed_limits is None or all(s > 0 for s in self.speed_limits),
            "SafetyEnvelope.speed_limits must be > 0",
        )
        _require(self.mass_kg > 0, "SafetyEnvelope.mass_kg must be > 0")
        _require(
            self.mass_lower_bound_kg is None or 0 < self.mass_lower_bound_kg <= self.mass_kg,
            "SafetyEnvelope.mass_lower_bound_kg must be in (0, mass_kg]",
        )
        _one_of(self, "braking", BRAKING_MODES)
        _require(
            self.brake_decel_mps2 is None or all(a >= 0 for a in self.brake_decel_mps2),
            "SafetyEnvelope.brake_decel_mps2 must be >= 0",
        )
        _one_of(self, "estop_reset", ESTOP_RESETS)
        _one_of(self, "estop_action", ESTOP_ACTIONS)


@dataclass(frozen=True)
class MHS(_Contract):
    """Model Hardware Standard description of one body (see module docstring)."""

    body_name: str
    body_class: str
    actuators: tuple[Actuator, ...]
    action_layout: tuple[str, ...]
    sensors: tuple[Sensor, ...]
    observation_layout: tuple[str, ...]
    body: Body
    control: Control
    safety: SafetyEnvelope
    mhs_version: str = MHS_VERSION

    def _validate_semantics(self) -> None:
        _non_empty(self, "body_name")
        _one_of(self, "body_class", BODY_CLASSES)
        _require(
            self.mhs_version == MHS_VERSION,
            f"MHS.mhs_version {self.mhs_version!r} != {MHS_VERSION!r}",
        )
        _require(len(self.actuators) > 0, "MHS.actuators must not be empty")
        _unique(tuple(a.name for a in self.actuators), "MHS actuator names")
        _permutation(self.action_layout, tuple(a.name for a in self.actuators), "MHS.action_layout")
        _unique(tuple(s.name for s in self.sensors), "MHS sensor names")
        channels = tuple(c for s in self.sensors for c in s.channels)
        _unique(channels, "MHS sensor channels")
        _permutation(self.observation_layout, channels, "MHS.observation_layout")

        frames = set(self.body.frames)
        for a in self.actuators:
            _require(a.frame in frames, f"Actuator {a.name}: frame {a.frame!r} not in Body.frames")
        for s in self.sensors:
            _require(s.frame in frames, f"Sensor {s.name}: frame {s.frame!r} not in Body.frames")
        env = self.safety
        _require(env.workspace_frame in frames, f"SafetyEnvelope.workspace_frame {env.workspace_frame!r} not in Body.frames")

        if self.body.mass_kg is not None:
            lower = env.mass_kg if env.mass_lower_bound_kg is None else env.mass_lower_bound_kg
            _require(
                lower <= self.body.mass_kg <= env.mass_kg,
                "SafetyEnvelope mass bounds must contain Body.mass_kg",
            )
        _require(
            len(env.safe_action) == len(self.action_layout),
            f"SafetyEnvelope.safe_action needs {len(self.action_layout)} values (action layout)",
        )
        for name, u in zip(self.action_layout, env.safe_action):
            a = self.actuator(name)
            _require(a.low <= u <= a.high, f"SafetyEnvelope.safe_action outside limits of {name}")
        # The kernel's workspace and stopping checks read the raw observation:
        # every workspace axis must be observable as a scalar position and velocity.
        for axis in env.workspace_axes:
            for kind in ("position", "velocity"):
                self.channel(kind, axis, env.workspace_frame)

    # -- layout queries (brain code reads layouts only through these) --------------

    def actuator(self, name: str) -> Actuator:
        for a in self.actuators:
            if a.name == name:
                return a
        raise ContractError(f"MHS: no actuator {name!r}")

    @property
    def action_actuators(self) -> tuple[Actuator, ...]:
        """Actuators in action-vector order."""
        return tuple(self.actuator(n) for n in self.action_layout)

    @property
    def action_size(self) -> int:
        return len(self.action_layout)

    def action_index(self, name: str) -> int:
        _require(name in self.action_layout, f"MHS: {name!r} not in the action layout")
        return self.action_layout.index(name)

    def observation_index(self, channel: str) -> int:
        _require(channel in self.observation_layout, f"MHS: {channel!r} not in the observation layout")
        return self.observation_layout.index(channel)

    def channel(self, kind: str, axis: str, frame: Optional[str] = None) -> str:
        """The unique scalar observation channel of sensor ``kind`` along ``axis``
        (of ``frame``, default the workspace frame); ContractError if none or several."""
        frame = self.safety.workspace_frame if frame is None else frame
        found = [
            s.name for s in self.sensors
            if s.kind == kind and s.axis == axis and s.frame == frame and s.shape == ()
        ]
        _require(
            len(found) == 1,
            f"MHS: need exactly one scalar {kind} sensor on axis {axis!r} of frame "
            f"{frame!r}, found {found}",
        )
        return found[0]
