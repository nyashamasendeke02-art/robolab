"""Deterministic Safety Kernel v1.2 (Gate 0/1, REQ-SAFE, REQ-SAFE+, ADR-003; ENG-0004, ENG-0010, ENG-0012).

Every actuator command passes :meth:`SafetyKernel.check`. The kernel is the
final authority of the control cycle and sits outside learned authority: its
limits are fixed at construction (:class:`SafetyConfig` is frozen, the kernel
offers no setter) and the emergency-stop latch can only be cleared through the
operator API (:meth:`SafetyKernel.reset_emergency_stop` with the operator key
given at construction). After construction, rebinding the limits or operator
key raises ``AttributeError``; any other attribute write from outside the
kernel (e.g. ``kernel._estop_reason = None``) is treated as tampering: it is
ignored, latches the emergency stop and is logged (decision ``"tamper"``).

Checks, in order (the first failing check decides):

1. emergency-stop latch  -> ``emergency_stop``;
2. malformed command (contracts validator, payload must be an
   :class:`ActionProposal` with one value per configured axis) -> ``reject``;
3. invalid robot state (wrong length / non-finite) -> ``reject``;
4. stale command (``now - timestamp > max_command_age_s``), a command from
   the future, or one issued before the last e-stop reset -> ``reject``;
5. per-axis action limits: ``limit_mode="clamp"`` clamps and approves with the
   violated limits listed in ``violated_constraints``; ``"reject"`` rejects;
6. workspace: the next position predicted from the action that would actually
   be applied must stay inside the workspace box, else ``reject``;
7. stopping distance (v1.1): after the command the body must still be able to
   stop inside the workspace under the braking safe action (see below), else
   ``reject`` (constraint ``stopping[axis]``); in ``limit_mode="clamp"`` the
   command is instead replaced by the braking action and approved, provided
   the braking action itself passes checks 6 and 7.

The contracts' ``SAFETY_VERDICTS`` has no ``clamp``: a clamped command is
``approve`` with the clamped ``approved_action`` and a non-empty
``violated_constraints``.

Every call returns a :class:`KernelResult` holding the contract
:class:`SafetyDecision`, a human-readable ``reason`` and ``actuator_command``,
the action that may reach the actuators: ``approved_action`` on approve and
the safe action otherwise.

Safe action (v1.1): with a known current velocity ``v`` (every :meth:`check`
with a valid state, and :meth:`tick` / :meth:`emergency_stop` when given
``velocity``) the safe action brakes: per axis
``clip(-mass_kg * v / control_dt_s, action_low, action_high)``, i.e. a force
opposing the velocity at the action limit, reduced on the last step so that it
stops the body instead of reversing it, and the configured ``safe_action``
component (default zero force) on an axis at rest. Without a known velocity
the configured ``safe_action`` is used, as in v1. Every decision is written to
the telemetry log (component ``"safety"``). If telemetry cannot be written the
kernel latches the emergency stop, since an unobservable kernel is not safe;
for the same reason an operator reset that cannot be logged is not applied.

Workspace prediction model (deterministic, not learned): each action component
is a force on the matching Cartesian axis of a point mass ``mass_kg``; over one
control period ``control_dt_s`` the predicted position is
``p' = p + v*dt + 0.5*(u/m)*dt**2`` per axis and the velocity ``v' = v + (u/m)*dt``.

Stopping-distance check (v1.1): per axis, braking deceleration
``a_brake = |action limit opposing v'| / mass_kg`` (friction and damping are
ignored, which is conservative since both only slow the body). The point where
the body comes to rest, ``p' + sign(v') * d``, must lie inside the workspace,
where ``d >= v'**2 / (2*a_brake)`` is the stopping distance in the direction of
motion. The braking action is applied in discrete control periods, so ``d`` is
the larger of the exact stopping distances of that braking sequence under the
kernel's zero-order-hold model above (from ``p'``; never shorter than
``v'**2 / (2*a_brake)``) and under semi-implicit Euler integration (``v`` is
updated before ``p``, e.g. Puck2D; there the command step itself ends at
``p + v'*dt``). Taking the worse of the two keeps the guarantee for either
plant discretisation. An axis that cannot brake in the direction of motion
(``a_brake == 0``) fails the check whenever ``v' != 0`` towards a bound.
Comparisons allow ``_TOL`` relative rounding slack (about 1e-12 of the bound),
so that exactly-tight stops computed in floating point are not rejected.
Not modelled: impulses, actuator rate limits, a ``mass_kg`` lower than the
true mass, or a plant period different from ``control_dt_s``. Actuator latency
is modelled from v1.2 on (see below).

Watchdog: :meth:`SafetyKernel.tick` must be called periodically; when no
command has been approved for longer than ``watchdog_timeout_s`` it returns a
decision whose ``actuator_command`` is the safe action (and ``None`` otherwise,
meaning the last approved command may stand).

v1.2 (ENG-0012, G1-5, APR-0003):

* :meth:`SafetyConfig.from_mhs` / :meth:`SafetyKernel.from_mhs` configure the
  kernel from a Model Hardware Standard description (``contracts.MHS``) with
  no per-body code. Kernel model requirements (one ``force`` actuator per
  workspace axis, in workspace-axis order, braking by ``actuators``, latching
  operator-reset e-stop that brakes) are checked; an MHS the kernel cannot
  model raises ``ContractError``.
* Safety independence: :meth:`SafetyKernel.observed_kinematics` reads position
  and velocity from the raw ``Observation`` through the configured channels
  (``position_channels`` / ``velocity_channels``, from the MHS sensor layout;
  default ``pos_<axis>`` / ``vel_<axis>``), so a state estimator cannot change
  what the kernel checks against. Missing channels read as NaN (``invalid_state``).
* Mass interval: ``mass_kg`` is an upper and ``mass_lower_kg`` (default
  ``mass_kg``) a lower bound on the true mass. The command step is predicted
  at both bounds (either may be the worse case); the braking deceleration
  uses the upper bound (and is capped by ``brake_decel`` when given); the
  braking safe action's last-step reduction uses the lower bound,
  ``-mass_lower_kg * v / dt``, so a lighter body is never reversed. With a
  lighter true body that last step leaves a residual speed that decays
  geometrically (ratio at most ``1 - mass_lower_kg / mass_kg``); the stopping
  distance includes that tail. With ``mass_lower_kg == mass_kg`` every check
  is exactly the v1.1 computation.
* ``max_speed`` (optional, per axis): the predicted speed after the command
  must not exceed it (constraint ``speed[axis]``; handled like ``stopping``).
* Actuator latency (``actuator_latency_s`` per axis, from the MHS actuators'
  ``latency_s``; default 0): a command (and the braking that follows it) can
  only take effect ``L`` seconds after it is issued. Until then the actuators
  keep applying an earlier force the kernel does not know, so it is taken as
  the worst case, the action limit pushing towards the bound being checked
  (or zero if no limit pushes that way). Per bound, the body is advanced
  ``ceil(L/dt)`` control periods under that force (semi-implicit Euler, the
  farther of the two integrations here) and the command step, workspace and
  stopping checks above then start from that state; a bound crossed during
  the latency fails as ``stopping[axis]``. This is exact for a dead time of
  at most ``L``; a first-order lag with time constant ``L`` (Puck2D's
  ``actuator_tau``) is approximated by it, not proven to be bounded by it.
  With ``L == 0`` every check is exactly the computation without latency.
* :meth:`SafetyKernel.no_command`: the decision for a cycle with no command
  to check (abstain, module failure): ``reject`` with constraint
  ``no_command`` and the braking safe action (or the e-stop decision). It does
  not reset the watchdog.
"""

from __future__ import annotations

import hmac
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from contracts import MHS, ActionProposal, ContractError, Envelope, Observation, SafetyDecision
from state.telemetry import TelemetryError, TelemetryLog

KERNEL_VERSION = "safety-kernel-1.2.0"
LIMIT_MODES = ("clamp", "reject")
COMPONENT = "safety"

# Constraint names used in SafetyDecision.violated_constraints.
C_ESTOP = "emergency_stop"
C_MALFORMED = "malformed_command"
C_INVALID_STATE = "invalid_state"
C_STALE = "stale_command"
C_FUTURE = "future_command"
C_PRE_RESET = "pre_reset_command"
C_WATCHDOG = "watchdog_timeout"
C_NO_COMMAND = "no_command"
C_INTERNAL = "internal_error"
C_TELEMETRY = "telemetry_failure"

# Relative floating-point slack for the stopping-distance comparison.
_TOL = 1e-12

# Kernel attributes whose rebinding raises AttributeError after construction.
_IMMUTABLE = ("_config", "_operator_key", "_telemetry", "_clock", "_sealed", "config")


def _floats(value: Any, name: str, n: Optional[int] = None) -> tuple[float, ...]:
    """Copy ``value`` into a tuple of finite floats (detaches caller lists)."""
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a sequence of numbers ({exc})") from exc
    if any(isinstance(v, bool) for v in value):
        raise ValueError(f"{name}: bools are not numbers")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name}: non-finite value in {out}")
    if n is not None and len(out) != n:
        raise ValueError(f"{name}: expected {n} values, got {len(out)}")
    return out


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected number, got {type(value).__name__}")
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return value


def _stopping_distances(
    speed: float, a_brake: float, dt: float, q: float = 0.0
) -> tuple[float, float]:
    """Distances (zero-order hold, semi-implicit Euler) covered while braking from
    ``speed`` with deceleration ``min(a_brake, speed_k / dt)`` per control period.

    ``q = 1 - mass_lower / mass_upper`` (v1.2): the reduced last-step action
    computed with the lower mass bound removes only a fraction ``>= 1 - q`` of
    the remaining speed from a heavier body. The speeds are then bounded by
    ``w_{k+1} = max(w_k - a_brake*dt, q*w_k)``; the distances below are those of
    that bound (for ``q == 0`` exactly the v1.1 formulas)."""
    if speed <= 0.0:
        return 0.0, 0.0
    if a_brake <= 0.0:
        return math.inf, math.inf
    step = a_brake * dt  # speed removed by one full braking period
    continuous = speed * speed / (2.0 * a_brake)
    if q <= 0.0:
        n = math.floor(speed / step)  # full braking periods
        rest = speed - n * step  # speed removed by the final, reduced period
        zoh = max(continuous, continuous - rest * rest / (2.0 * a_brake) + 0.5 * rest * dt)
        semi_implicit = dt * (n * speed - step * n * (n + 1) / 2.0)
        return zoh, max(0.0, semi_implicit)
    w_star = step / (1.0 - q)  # below this the bound decays geometrically
    n = max(0, math.ceil((speed - w_star) / step))  # linear braking periods
    w_n = speed - n * step
    total = n * speed - step * n * (n + 1) / 2.0 + w_n * q / (1.0 - q)  # sum_{k>=1} w_k
    return max(continuous, dt * (0.5 * speed + total)), max(0.0, dt * total)


@dataclass(frozen=True)
class SafetyConfig:
    """Immutable kernel limits. Sequences are copied into tuples of floats."""

    axis_names: tuple[str, ...]
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    workspace_low: tuple[float, ...]
    workspace_high: tuple[float, ...]
    max_command_age_s: float
    watchdog_timeout_s: float
    limit_mode: str = "reject"
    mass_kg: float = 1.0
    control_dt_s: float = 0.01
    safe_action: Optional[tuple[float, ...]] = None  # default: zero force
    kernel_version: str = KERNEL_VERSION
    mass_lower_kg: Optional[float] = None  # lower mass bound; default: mass_kg
    max_speed: Optional[tuple[float, ...]] = None  # m/s per axis; None: no speed check
    brake_decel: Optional[tuple[float, ...]] = None  # m/s^2 per axis cap; None: actuators
    position_channels: Optional[tuple[str, ...]] = None  # default: pos_<axis>
    velocity_channels: Optional[tuple[str, ...]] = None  # default: vel_<axis>
    actuator_latency_s: Optional[tuple[float, ...]] = None  # s per axis; None: no latency

    def __post_init__(self) -> None:
        names = tuple(self.axis_names)
        if not names or not all(isinstance(a, str) and a for a in names):
            raise ValueError("axis_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"axis_names must be unique, got {names}")
        n = len(names)
        mass = _positive(self.mass_kg, "mass_kg")
        fixed = {
            "mass_lower_kg": (
                mass if self.mass_lower_kg is None
                else _positive(self.mass_lower_kg, "mass_lower_kg")
            ),
            "max_speed": (
                None if self.max_speed is None else _floats(self.max_speed, "max_speed", n)
            ),
            "brake_decel": (
                None if self.brake_decel is None
                else _floats(self.brake_decel, "brake_decel", n)
            ),
            "actuator_latency_s": (
                (0.0,) * n if self.actuator_latency_s is None
                else _floats(self.actuator_latency_s, "actuator_latency_s", n)
            ),
            "position_channels": _channels(self.position_channels, "pos", names),
            "velocity_channels": _channels(self.velocity_channels, "vel", names),
            "axis_names": names,
            "action_low": _floats(self.action_low, "action_low", n),
            "action_high": _floats(self.action_high, "action_high", n),
            "workspace_low": _floats(self.workspace_low, "workspace_low", n),
            "workspace_high": _floats(self.workspace_high, "workspace_high", n),
            "max_command_age_s": _positive(self.max_command_age_s, "max_command_age_s"),
            "watchdog_timeout_s": _positive(self.watchdog_timeout_s, "watchdog_timeout_s"),
            "mass_kg": mass,
            "control_dt_s": _positive(self.control_dt_s, "control_dt_s"),
            "safe_action": (
                (0.0,) * n
                if self.safe_action is None
                else _floats(self.safe_action, "safe_action", n)
            ),
        }
        for k, v in fixed.items():
            object.__setattr__(self, k, v)
        if self.limit_mode not in LIMIT_MODES:
            raise ValueError(f"limit_mode={self.limit_mode!r} not in {list(LIMIT_MODES)}")
        if not isinstance(self.kernel_version, str) or not self.kernel_version:
            raise ValueError("kernel_version must be a non-empty str")
        for i, a in enumerate(names):
            if not self.action_low[i] <= self.action_high[i]:
                raise ValueError(f"action_low > action_high on axis {a}")
            if not self.workspace_low[i] < self.workspace_high[i]:
                raise ValueError(f"workspace_low >= workspace_high on axis {a}")
            if not self.action_low[i] <= self.safe_action[i] <= self.action_high[i]:
                raise ValueError(f"safe_action outside action limits on axis {a}")
        if self.mass_lower_kg > self.mass_kg:
            raise ValueError("mass_lower_kg must be <= mass_kg")
        if self.max_speed is not None and not all(s > 0 for s in self.max_speed):
            raise ValueError("max_speed must be > 0")
        if self.brake_decel is not None and not all(b >= 0 for b in self.brake_decel):
            raise ValueError("brake_decel must be >= 0")
        if not all(lat >= 0 for lat in self.actuator_latency_s):
            raise ValueError("actuator_latency_s must be >= 0")
        channels = self.position_channels + self.velocity_channels
        if len(set(channels)) != len(channels):
            raise ValueError(f"position/velocity channels must be unique, got {channels}")

    @property
    def n_axes(self) -> int:
        return len(self.axis_names)

    @classmethod
    def from_mhs(
        cls, mhs: MHS, *, limit_mode: str = "reject", kernel_version: str = KERNEL_VERSION
    ) -> "SafetyConfig":
        """Kernel limits from a Model Hardware Standard description (no per-body code).

        Raises ``ContractError`` for an invalid MHS or one the kernel's model
        does not cover (see the module docstring). ``limit_mode`` is kernel
        policy, not a body property, so it is passed separately.
        """
        if not isinstance(mhs, MHS):
            raise TypeError(f"mhs must be MHS, got {type(mhs).__name__}")
        mhs.validate()
        env = mhs.safety
        axes = env.workspace_axes
        acts = mhs.action_actuators
        if len(acts) != len(axes) or any(
            a.kind != "force" or a.axis != axis or a.frame != env.workspace_frame
            for a, axis in zip(acts, axes)
        ):
            raise ContractError(
                f"{KERNEL_VERSION} models one force actuator per workspace axis "
                f"{list(axes)} (frame {env.workspace_frame!r}), in workspace-axis order; "
                f"action layout {list(mhs.action_layout)} does not match"
            )
        if env.braking != "actuators":
            raise ContractError(f"{KERNEL_VERSION} needs braking='actuators', got {env.braking!r}")
        if not (env.estop_latching and env.estop_reset == "operator" and env.estop_action == "brake"):
            raise ContractError(
                f"{KERNEL_VERSION} implements a latching, operator-reset e-stop that brakes"
            )
        return cls(
            axis_names=axes,
            action_low=tuple(a.low for a in acts),
            action_high=tuple(a.high for a in acts),
            workspace_low=env.workspace_low,
            workspace_high=env.workspace_high,
            max_command_age_s=mhs.control.max_command_age_s,
            watchdog_timeout_s=mhs.control.watchdog_timeout_s,
            limit_mode=limit_mode,
            mass_kg=env.mass_kg,
            control_dt_s=mhs.control.period_s,
            safe_action=env.safe_action,
            kernel_version=kernel_version,
            mass_lower_kg=env.mass_lower_bound_kg,
            max_speed=env.speed_limits,
            brake_decel=env.brake_decel_mps2,
            position_channels=tuple(mhs.channel("position", a) for a in axes),
            velocity_channels=tuple(mhs.channel("velocity", a) for a in axes),
            actuator_latency_s=tuple(a.latency_s for a in acts),
        )


def _channels(value: Any, prefix: str, axes: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return tuple(f"{prefix}_{a}" for a in axes)
    out = tuple(value)
    if len(out) != len(axes) or not all(isinstance(c, str) and c for c in out):
        raise ValueError(f"{prefix} channels: expected {len(axes)} non-empty strings, got {out}")
    return out


@dataclass(frozen=True)
class KernelResult:
    """One kernel decision: contract message, rationale and actuator output."""

    decision: SafetyDecision
    reason: str
    actuator_command: tuple[float, ...]
    timestamp: float
    message_id: Optional[str] = None


class SafetyKernel:
    """Deterministic final-authority filter for actuator commands.

    ``clock`` (seconds, same time base as ``Envelope.timestamp``) is injected
    so behaviour is reproducible; every public call also accepts ``now``.
    """

    def __init__(
        self,
        config: SafetyConfig,
        telemetry: TelemetryLog,
        operator_key: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, SafetyConfig):
            raise TypeError(f"config must be SafetyConfig, got {type(config).__name__}")
        if not isinstance(telemetry, TelemetryLog):
            raise TypeError(f"telemetry must be TelemetryLog, got {type(telemetry).__name__}")
        if not isinstance(operator_key, str) or not operator_key:
            raise ValueError("operator_key must be a non-empty str")
        self._config = config
        self._telemetry = telemetry
        self._operator_key = operator_key
        self._clock = clock
        start = float(clock())
        self._last_valid_time = start  # watchdog reference
        self._last_reset_time = -math.inf
        self._estop_reason: Optional[str] = None
        self._last_cycle_id = 0
        self._telemetry_failures = 0
        self._sealed = True

    @classmethod
    def from_mhs(
        cls,
        mhs: MHS,
        telemetry: TelemetryLog,
        operator_key: str,
        clock: Callable[[], float] = time.time,
        *,
        limit_mode: str = "reject",
    ) -> "SafetyKernel":
        """A kernel configured from ``mhs`` (:meth:`SafetyConfig.from_mhs`)."""
        return cls(SafetyConfig.from_mhs(mhs, limit_mode=limit_mode), telemetry, operator_key, clock)

    def __setattr__(self, name: str, value: Any) -> None:
        # After construction the kernel mutates its own state only via _set.
        # Rebinding limits or credentials raises; any other outside write
        # (e.g. clearing _estop_reason) is tampering: it is not applied, it
        # latches the e-stop and it is logged.
        if not getattr(self, "_sealed", False):
            object.__setattr__(self, name, value)
        elif name in _IMMUTABLE:
            raise AttributeError(f"SafetyKernel.{name} is immutable after construction")
        else:
            self._tamper(name)

    def _set(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def _tamper(self, name: str) -> None:
        reason = f"tamper: outside write to SafetyKernel.{name}"
        if self._estop_reason is None:
            self._set("_estop_reason", reason)
        self._log_event(self._now(None), decision="tamper", reason=reason, level="error")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"SafetyKernel attributes cannot be deleted ({name})")

    # -- read-only views -----------------------------------------------------

    @property
    def config(self) -> SafetyConfig:
        return self._config

    @property
    def estopped(self) -> bool:
        return self._estop_reason is not None

    @property
    def telemetry_failures(self) -> int:
        return self._telemetry_failures

    # -- command path --------------------------------------------------------

    def observed_kinematics(
        self, observation: Any
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """(position, velocity) read from a raw :class:`Observation` through the
        configured channels. Never raises: a missing channel, duplicated channel
        names or an invalid observation read as NaN, which :meth:`check` rejects
        as ``invalid_state`` and :meth:`tick` treats as an unknown velocity."""
        cfg = self._config
        nan = (math.nan,) * cfg.n_axes
        try:
            if type(observation) is not Observation:
                return nan, nan
            observation.validate()
            if len(set(observation.channels)) != len(observation.channels):
                return nan, nan
            values = dict(zip(observation.channels, observation.values))
            pos = tuple(values.get(c, math.nan) for c in cfg.position_channels)
            vel = tuple(values.get(c, math.nan) for c in cfg.velocity_channels)
            return pos, vel
        except Exception:
            return nan, nan

    def check(
        self,
        command: Any,
        *,
        position: Sequence[float],
        velocity: Sequence[float],
        now: Optional[float] = None,
    ) -> KernelResult:
        """Decide one actuator command. Never raises; failures are rejections.

        ``command`` is an :class:`Envelope` (or its JSON text / dict form)
        carrying an :class:`ActionProposal`; ``position`` and ``velocity`` are
        the current Cartesian state used for the workspace prediction.
        """
        t0 = time.perf_counter()
        now = self._now(now)
        safe = self._config.safe_action
        try:
            safe = self._safe_action(velocity)
            result = self._evaluate(command, position, velocity, now, safe)
        except Exception as exc:  # fail safe on any internal fault
            result = self._result(
                "reject", "unknown", (), (C_INTERNAL,), f"internal error: {exc!r}", now,
                safe=safe,
            )
        if result.decision.verdict == "approve":
            self._set("_last_valid_time", now)
        return self._emit(result, t0, safe)

    def tick(
        self, now: Optional[float] = None, *, velocity: Optional[Sequence[float]] = None
    ) -> Optional[KernelResult]:
        """Watchdog. Returns the safe-action decision on timeout/e-stop, else None.

        With ``velocity`` (current Cartesian velocity) the safe action brakes;
        without it (or if it is invalid) the configured ``safe_action`` is used.
        """
        t0 = time.perf_counter()
        now = self._now(now)
        safe = self._safe_action(velocity)
        if self.estopped:
            return self._emit(self._estop_result("watchdog", now, safe=safe), t0, safe)
        silence = now - self._last_valid_time
        if silence > self._config.watchdog_timeout_s:
            return self._emit(
                self._result(
                    "reject", "watchdog", (), (C_WATCHDOG,),
                    f"no valid command for {silence:.6g}s "
                    f"> watchdog_timeout_s={self._config.watchdog_timeout_s}",
                    now, safe=safe,
                ),
                t0, safe,
            )
        return None

    def no_command(
        self,
        now: Optional[float] = None,
        *,
        velocity: Optional[Sequence[float]] = None,
        reason: str = "no command to check this cycle",
    ) -> KernelResult:
        """Decision for a cycle without a command (abstain, module failure).

        ``reject`` with constraint ``no_command`` (or the e-stop decision when
        latched) whose ``actuator_command`` is the safe action, braking when
        ``velocity`` is known. Does not reset the watchdog.
        """
        t0 = time.perf_counter()
        now = self._now(now)
        safe = self._safe_action(velocity)
        if self.estopped:
            return self._emit(self._estop_result(C_NO_COMMAND, now, safe=safe), t0, safe)
        return self._emit(
            self._result("reject", C_NO_COMMAND, (), (C_NO_COMMAND,), str(reason), now, safe=safe),
            t0, safe,
        )

    # -- emergency stop (independent of every learned module) -----------------

    def emergency_stop(
        self,
        reason: str = "operator",
        now: Optional[float] = None,
        *,
        velocity: Optional[Sequence[float]] = None,
    ) -> KernelResult:
        """Latch the e-stop. Anyone may stop; only the operator may reset.

        With ``velocity`` the returned safe action brakes (see :meth:`tick`).
        """
        t0 = time.perf_counter()
        now = self._now(now)
        safe = self._safe_action(velocity)
        if self._estop_reason is None:
            self._set("_estop_reason", str(reason) or "unspecified")
        return self._emit(self._estop_result("estop", now, safe=safe), t0, safe)

    def reset_emergency_stop(self, operator_key: str, now: Optional[float] = None) -> bool:
        """Operator API: clear the e-stop latch. Raises PermissionError on a bad key.

        Returns True when the latch was cleared. The reset is only applied once
        it has been logged: if telemetry cannot record it the latch stays set
        and False is returned (an unobservable reset is not a safe reset).
        Commands timestamped before the reset are rejected afterwards, and the
        watchdog restarts from the reset time.
        """
        now = self._now(now)
        ok = isinstance(operator_key, str) and hmac.compare_digest(
            operator_key.encode("utf-8"), self._operator_key.encode("utf-8")
        )
        logged = self._log_event(
            now,
            decision="estop_reset" if ok else "estop_reset_denied",
            reason="operator reset" if ok else "invalid operator key",
            level="warning",
        )
        if not ok:
            raise PermissionError("invalid operator key; e-stop remains latched")
        if not logged:
            if self._estop_reason is None:
                self._set("_estop_reason", C_TELEMETRY)
            return False
        self._set("_estop_reason", None)
        self._set("_last_reset_time", now)
        self._set("_last_valid_time", now)
        return True

    # -- internals -----------------------------------------------------------

    def _now(self, now: Optional[float]) -> float:
        return float(self._clock() if now is None else now)

    def _evaluate(
        self, command: Any, position: Any, velocity: Any, now: float, safe: tuple[float, ...]
    ) -> KernelResult:
        cfg = self._config
        res = lambda *a, **kw: self._result(*a, safe=safe, **kw)  # noqa: E731
        env, pid, problem = self._parse(command)
        if env is not None:
            self._set("_last_cycle_id", env.cycle_id)
        mid = None if env is None else env.message_id

        if self.estopped:
            return self._estop_result(pid, now, mid, safe=safe)
        if problem is not None:
            return res("reject", pid, (), (C_MALFORMED,), problem, now, mid)
        try:
            pos = _floats(position, "position", cfg.n_axes)
            vel = _floats(velocity, "velocity", cfg.n_axes)
        except ValueError as exc:
            return res("reject", pid, (), (C_INVALID_STATE,), str(exc), now, mid)

        age = now - env.timestamp
        if age > cfg.max_command_age_s:
            return res(
                "reject", pid, (), (C_STALE,),
                f"command age {age:.6g}s > max_command_age_s={cfg.max_command_age_s}",
                now, mid,
            )
        if age < 0:
            return res(
                "reject", pid, (), (C_FUTURE,),
                f"command timestamp {env.timestamp} is after kernel time {now}", now, mid,
            )
        if env.timestamp < self._last_reset_time:
            return res(
                "reject", pid, (), (C_PRE_RESET,),
                "command was issued before the last e-stop reset", now, mid,
            )

        action = env.payload.action
        violated = []
        applied = []
        for i, a in enumerate(cfg.axis_names):
            lo, hi = cfg.action_low[i], cfg.action_high[i]
            u = action[i]
            if u < lo or u > hi:
                violated.append(f"action_limit[{a}]")
            applied.append(min(max(u, lo), hi))
        if violated and cfg.limit_mode == "reject":
            return res(
                "reject", pid, (), tuple(violated),
                f"action {action} outside limits on {len(violated)} axis/axes", now, mid,
            )

        outside, cannot_stop = self._workspace_check(pos, vel, applied)
        if outside:
            return res(
                "reject", pid, (), tuple(violated + outside),
                "predicted next position leaves the workspace", now, mid,
            )
        if cannot_stop:
            constraints = tuple(violated + cannot_stop)
            if cfg.limit_mode == "clamp" and safe != tuple(applied):
                b_out, b_stop = self._workspace_check(pos, vel, safe)
                if not b_out and not b_stop:
                    return res(
                        "approve", pid, safe, constraints,
                        "replaced by braking action: body could not stop inside the "
                        f"workspace (or exceeded a speed limit) on {len(cannot_stop)} "
                        "axis/axes", now, mid,
                    )
            return res(
                "reject", pid, (), constraints,
                "body could not stop inside the workspace (or exceeded a speed limit) "
                "after this command", now, mid,
            )

        reason = (
            f"approved after clamping {len(violated)} axis/axes" if violated else "approved"
        )
        return res("approve", pid, tuple(applied), tuple(violated), reason, now, mid)

    def _workspace_check(
        self, pos: Sequence[float], vel: Sequence[float], applied: Sequence[float]
    ) -> tuple[list[str], list[str]]:
        """Return (axes whose next position leaves the workspace, axes that could
        not stop inside the workspace afterwards) for ``applied`` from (pos, vel)."""
        cfg = self._config
        m_hi, m_lo = cfg.mass_kg, cfg.mass_lower_kg
        # The command step is predicted at both mass bounds (equal bounds: once,
        # exactly as in v1.1).
        masses = (m_hi,) if m_lo == m_hi else (m_lo, m_hi)
        q = 1.0 - m_lo / m_hi
        outside, cannot_stop = [], []
        for i, a in enumerate(cfg.axis_names):
            found: set[str] = set()
            for m in masses:
                found |= self._axis_check(i, pos[i], vel[i], applied[i], m, q)
            if "outside" in found:
                outside.append(f"workspace[{a}]")
                continue
            if "speed" in found:
                cannot_stop.append(f"speed[{a}]")
            if "stop" in found:
                cannot_stop.append(f"stopping[{a}]")
        return outside, cannot_stop

    def _axis_check(
        self, i: int, p0: float, v0: float, u: float, m: float, q: float
    ) -> set[str]:
        """Failed checks ("outside", "speed", "stop") on axis ``i`` for force ``u``
        applied to mass ``m``; braking always uses the upper mass bound. With an
        actuator latency the checks start after it, once per bound, under the
        unknown earlier force taken as the limit pushing towards that bound."""
        cfg = self._config
        latency = cfg.actuator_latency_s[i]
        if latency == 0.0:
            return self._axis_check_now(i, p0, v0, u, m, q)
        dt = cfg.control_dt_s
        n = math.ceil(latency / dt - _TOL)  # control periods before the command acts
        found: set[str] = set()
        for f_adv, bound, sign in (
            (max(0.0, cfg.action_high[i]), cfg.workspace_high[i], 1.0),
            (min(0.0, cfg.action_low[i]), cfg.workspace_low[i], -1.0),
        ):
            a = f_adv / m
            # n semi-implicit Euler steps (>= the zero-order-hold distance towards the bound).
            p_l = p0 + n * dt * v0 + a * dt * dt * n * (n + 1) / 2.0
            v_l = v0 + a * n * dt
            if sign * (p_l - bound) > _TOL * max(1.0, abs(bound)):
                found.add("stop")
                continue
            found |= {"stop" if f == "outside" else f
                      for f in self._axis_check_now(i, p_l, v_l, u, m, q)}
        return found

    def _axis_check_now(
        self, i: int, p0: float, v0: float, u: float, m: float, q: float
    ) -> set[str]:
        """:meth:`_axis_check` for a command that acts immediately."""
        cfg = self._config
        dt = cfg.control_dt_s
        lo, hi = cfg.workspace_low[i], cfg.workspace_high[i]
        acc = u / m
        p = p0 + v0 * dt + 0.5 * acc * dt * dt
        if not lo <= p <= hi:
            return {"outside"}
        v = v0 + acc * dt
        found = set()
        if cfg.max_speed is not None and abs(v) > cfg.max_speed[i]:
            found.add("speed")
        if v == 0.0:
            return found
        if v > 0:
            bound, sign, a_brake = hi, 1.0, max(0.0, -cfg.action_low[i]) / cfg.mass_kg
        else:
            bound, sign, a_brake = lo, -1.0, max(0.0, cfg.action_high[i]) / cfg.mass_kg
        if cfg.brake_decel is not None:
            a_brake = min(a_brake, cfg.brake_decel[i])
        d_zoh, d_si = _stopping_distances(abs(v), a_brake, dt, q)
        p_si = p0 + v * dt  # semi-implicit Euler next position
        slack = _TOL * max(1.0, abs(bound))
        if not (
            sign * (p + sign * d_zoh - bound) <= slack
            and sign * (p_si + sign * d_si - bound) <= slack
        ):
            found.add("stop")
        return found

    def _safe_action(self, velocity: Any) -> tuple[float, ...]:
        """Braking safe action for ``velocity``; configured safe_action if unknown."""
        cfg = self._config
        if velocity is None:
            return cfg.safe_action
        try:
            vel = _floats(velocity, "velocity", cfg.n_axes)
        except ValueError:
            return cfg.safe_action
        out = []
        for i, v in enumerate(vel):
            if v == 0.0:
                out.append(cfg.safe_action[i])
            else:
                # Lower mass bound: the last, reduced step never reverses the body.
                u = -cfg.mass_lower_kg * v / cfg.control_dt_s
                out.append(min(max(u, cfg.action_low[i]), cfg.action_high[i]))
        return tuple(out)


    def _parse(self, command: Any) -> tuple[Optional[Envelope], str, Optional[str]]:
        """Return (envelope, proposal_id, problem); problem is None when well-formed."""
        try:
            if isinstance(command, Envelope):
                command.validate()  # re-check: frozen objects can still be forged
                env = command
            elif isinstance(command, str):
                env = Envelope.from_json(command)
            elif isinstance(command, dict):
                env = Envelope.from_dict(command)
            else:
                raise ContractError(f"unsupported command type {type(command).__name__}")
        except (ContractError, TypeError, ValueError, AttributeError) as exc:
            return None, "unknown", f"malformed command: {exc}"
        payload = env.payload
        if type(payload) is not ActionProposal:
            return env, "unknown", f"payload {env.message_type} is not an ActionProposal"
        n = self._config.n_axes
        if len(payload.action) != n:
            return (
                env, payload.proposal_id,
                f"action has {len(payload.action)} values, kernel has {n} axes",
            )
        return env, payload.proposal_id, None

    def _result(
        self,
        verdict: str,
        proposal_id: str,
        approved: tuple[float, ...],
        violated: tuple[str, ...],
        reason: str,
        now: float,
        message_id: Optional[str] = None,
        *,
        safe: Optional[tuple[float, ...]] = None,
    ) -> KernelResult:
        decision = SafetyDecision(
            verdict=verdict,
            proposal_id=proposal_id,
            approved_action=approved,
            violated_constraints=violated,
            kernel_version=self._config.kernel_version,
        )
        return KernelResult(
            decision=decision,
            reason=reason,
            actuator_command=(
                approved if verdict == "approve"
                else self._config.safe_action if safe is None else safe
            ),
            timestamp=now,
            message_id=message_id,
        )

    def _estop_result(
        self,
        proposal_id: str,
        now: float,
        mid: Optional[str] = None,
        *,
        safe: Optional[tuple[float, ...]] = None,
    ) -> KernelResult:
        return self._result(
            "emergency_stop", proposal_id, (), (C_ESTOP,),
            f"emergency stop latched ({self._estop_reason}); operator reset required",
            now, mid, safe=safe,
        )

    def _emit(
        self, result: KernelResult, t0: float, safe: Optional[tuple[float, ...]] = None
    ) -> KernelResult:
        """Log a decision. A logging failure latches the e-stop (fail safe)."""
        d = result.decision
        ok = self._log_event(
            result.timestamp,
            decision=d.verdict,
            reason=result.reason,
            level="info" if d.verdict == "approve" and not d.violated_constraints else "warning",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            data={
                "proposal_id": d.proposal_id,
                "message_id": result.message_id,
                "approved_action": list(d.approved_action),
                "actuator_command": list(result.actuator_command),
                "violated_constraints": list(d.violated_constraints),
            },
        )
        if ok:
            return result
        if self._estop_reason is None:
            self._set("_estop_reason", C_TELEMETRY)
        return self._estop_result(
            d.proposal_id, result.timestamp, result.message_id, safe=safe
        )

    def _log_event(
        self,
        now: float,
        *,
        decision: str,
        reason: str,
        level: str,
        latency_ms: float = 0.0,
        data: Optional[dict] = None,
    ) -> bool:
        try:
            self._telemetry.log(
                COMPONENT,
                cycle_id=self._last_cycle_id,
                decision=decision,
                reason=reason,
                latency_ms=max(0.0, latency_ms),
                model_version=self._config.kernel_version,
                level=level,
                timestamp=now,
                data=data,
            )
            return True
        except (TelemetryError, ContractError, OSError, ValueError):
            self._set("_telemetry_failures", self._telemetry_failures + 1)
            return False
