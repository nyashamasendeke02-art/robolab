"""Deterministic Safety Kernel v1 (Gate 0, REQ-SAFE, ADR-003; ENG-0004).

Every actuator command passes :meth:`SafetyKernel.check`. The kernel is the
final authority of the control cycle and sits outside learned authority: its
limits are fixed at construction (:class:`SafetyConfig` is frozen, the kernel
offers no setter) and the emergency-stop latch can only be cleared through the
operator API (:meth:`SafetyKernel.reset_emergency_stop` with the operator key
given at construction).

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
   be applied must stay inside the workspace box, else ``reject``.

The contracts' ``SAFETY_VERDICTS`` has no ``clamp``: a clamped command is
``approve`` with the clamped ``approved_action`` and a non-empty
``violated_constraints``.

Every call returns a :class:`KernelResult` holding the contract
:class:`SafetyDecision`, a human-readable ``reason`` and ``actuator_command``,
the action that may reach the actuators: ``approved_action`` on approve and
the configured safe action (zero force) otherwise. Every decision is written to
the telemetry log (component ``"safety"``). If telemetry cannot be written the
kernel latches the emergency stop, since an unobservable kernel is not safe.

Workspace prediction model (deterministic, not learned): each action component
is a force on the matching Cartesian axis of a point mass ``mass_kg``; over one
control period ``control_dt_s`` the predicted position is
``p + v*dt + 0.5*(u/m)*dt**2`` per axis.

Watchdog: :meth:`SafetyKernel.tick` must be called periodically; when no
command has been approved for longer than ``watchdog_timeout_s`` it returns a
decision whose ``actuator_command`` is the safe action (and ``None`` otherwise,
meaning the last approved command may stand).
"""

from __future__ import annotations

import hmac
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from contracts import ActionProposal, ContractError, Envelope, SafetyDecision
from state.telemetry import TelemetryError, TelemetryLog

KERNEL_VERSION = "safety-kernel-1.0.0"
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
C_INTERNAL = "internal_error"
C_TELEMETRY = "telemetry_failure"


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

    def __post_init__(self) -> None:
        names = tuple(self.axis_names)
        if not names or not all(isinstance(a, str) and a for a in names):
            raise ValueError("axis_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"axis_names must be unique, got {names}")
        n = len(names)
        fixed = {
            "axis_names": names,
            "action_low": _floats(self.action_low, "action_low", n),
            "action_high": _floats(self.action_high, "action_high", n),
            "workspace_low": _floats(self.workspace_low, "workspace_low", n),
            "workspace_high": _floats(self.workspace_high, "workspace_high", n),
            "max_command_age_s": _positive(self.max_command_age_s, "max_command_age_s"),
            "watchdog_timeout_s": _positive(self.watchdog_timeout_s, "watchdog_timeout_s"),
            "mass_kg": _positive(self.mass_kg, "mass_kg"),
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

    @property
    def n_axes(self) -> int:
        return len(self.axis_names)


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

    def __setattr__(self, name: str, value: Any) -> None:
        # Limits and operator credentials cannot be rebound after construction.
        if getattr(self, "_sealed", False) and name in (
            "_config", "_operator_key", "_telemetry", "_clock", "_sealed",
        ):
            raise AttributeError(f"SafetyKernel.{name} is immutable after construction")
        object.__setattr__(self, name, value)

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
        try:
            result = self._evaluate(command, position, velocity, now)
        except Exception as exc:  # fail safe on any internal fault
            result = self._result(
                "reject", "unknown", (), (C_INTERNAL,), f"internal error: {exc!r}", now
            )
        if result.decision.verdict == "approve":
            self._last_valid_time = now
        return self._emit(result, t0)

    def tick(self, now: Optional[float] = None) -> Optional[KernelResult]:
        """Watchdog. Returns the safe-action decision on timeout/e-stop, else None."""
        t0 = time.perf_counter()
        now = self._now(now)
        if self.estopped:
            return self._emit(self._estop_result("watchdog", now), t0)
        silence = now - self._last_valid_time
        if silence > self._config.watchdog_timeout_s:
            return self._emit(
                self._result(
                    "reject", "watchdog", (), (C_WATCHDOG,),
                    f"no valid command for {silence:.6g}s "
                    f"> watchdog_timeout_s={self._config.watchdog_timeout_s}",
                    now,
                ),
                t0,
            )
        return None

    # -- emergency stop (independent of every learned module) -----------------

    def emergency_stop(self, reason: str = "operator", now: Optional[float] = None) -> KernelResult:
        """Latch the e-stop. Anyone may stop; only the operator may reset."""
        t0 = time.perf_counter()
        now = self._now(now)
        if self._estop_reason is None:
            self._estop_reason = str(reason) or "unspecified"
        return self._emit(self._estop_result("estop", now), t0)

    def reset_emergency_stop(self, operator_key: str, now: Optional[float] = None) -> None:
        """Operator API: clear the e-stop latch. Raises PermissionError on a bad key.

        Commands timestamped before the reset are rejected afterwards, and the
        watchdog restarts from the reset time.
        """
        now = self._now(now)
        ok = isinstance(operator_key, str) and hmac.compare_digest(
            operator_key.encode("utf-8"), self._operator_key.encode("utf-8")
        )
        self._log_event(
            now,
            decision="estop_reset" if ok else "estop_reset_denied",
            reason="operator reset" if ok else "invalid operator key",
            level="warning",
        )
        if not ok:
            raise PermissionError("invalid operator key; e-stop remains latched")
        self._estop_reason = None
        self._last_reset_time = now
        self._last_valid_time = now

    # -- internals -----------------------------------------------------------

    def _now(self, now: Optional[float]) -> float:
        return float(self._clock() if now is None else now)

    def _evaluate(self, command: Any, position: Any, velocity: Any, now: float) -> KernelResult:
        cfg = self._config
        env, pid, problem = self._parse(command)
        if env is not None:
            self._last_cycle_id = env.cycle_id
        mid = None if env is None else env.message_id

        if self.estopped:
            return self._estop_result(pid, now, mid)
        if problem is not None:
            return self._result("reject", pid, (), (C_MALFORMED,), problem, now, mid)
        try:
            pos = _floats(position, "position", cfg.n_axes)
            vel = _floats(velocity, "velocity", cfg.n_axes)
        except ValueError as exc:
            return self._result("reject", pid, (), (C_INVALID_STATE,), str(exc), now, mid)

        age = now - env.timestamp
        if age > cfg.max_command_age_s:
            return self._result(
                "reject", pid, (), (C_STALE,),
                f"command age {age:.6g}s > max_command_age_s={cfg.max_command_age_s}",
                now, mid,
            )
        if age < 0:
            return self._result(
                "reject", pid, (), (C_FUTURE,),
                f"command timestamp {env.timestamp} is after kernel time {now}", now, mid,
            )
        if env.timestamp < self._last_reset_time:
            return self._result(
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
            return self._result(
                "reject", pid, (), tuple(violated),
                f"action {action} outside limits on {len(violated)} axis/axes", now, mid,
            )

        dt = cfg.control_dt_s
        outside = []
        for i, a in enumerate(cfg.axis_names):
            p = pos[i] + vel[i] * dt + 0.5 * (applied[i] / cfg.mass_kg) * dt * dt
            if not cfg.workspace_low[i] <= p <= cfg.workspace_high[i]:
                outside.append(f"workspace[{a}]")
        if outside:
            return self._result(
                "reject", pid, (), tuple(violated + outside),
                "predicted next position leaves the workspace", now, mid,
            )

        reason = (
            f"approved after clamping {len(violated)} axis/axes" if violated else "approved"
        )
        return self._result("approve", pid, tuple(applied), tuple(violated), reason, now, mid)

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
            actuator_command=approved if verdict == "approve" else self._config.safe_action,
            timestamp=now,
            message_id=message_id,
        )

    def _estop_result(self, proposal_id: str, now: float, mid: Optional[str] = None) -> KernelResult:
        return self._result(
            "emergency_stop", proposal_id, (), (C_ESTOP,),
            f"emergency stop latched ({self._estop_reason}); operator reset required",
            now, mid,
        )

    def _emit(self, result: KernelResult, t0: float) -> KernelResult:
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
            self._estop_reason = C_TELEMETRY
        return self._estop_result(d.proposal_id, result.timestamp, result.message_id)

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
            self._telemetry_failures += 1
            return False
