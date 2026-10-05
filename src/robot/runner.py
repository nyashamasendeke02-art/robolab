"""Deterministic synchronous cycle runner (Gate 0, ADR-001, ADR-002, REQ-STATE, REQ-S1; ENG-0005).

One call to :meth:`CycleRunner.step` runs one control cycle::

    watchdog tick -> observe -> state estimate -> world-model predict
    -> System 1 propose -> Awareness arbitrate [-> System 2 plan -> re-arbitrate]
    -> Safety Kernel -> actuate -> outcome -> telemetry

Every module sits behind a :class:`typing.Protocol` (:class:`StateEstimator`,
:class:`WorldModel`, :class:`System1`, :class:`System2`, :class:`Awareness`,
:class:`Environment`) and is chosen by :class:`Modules` - either objects or
``"package.module:Factory"`` strings via :meth:`Modules.from_config` - so a
module is swapped without touching the runner (ADR-001). The defaults are null
or pass-through implementations.

Safety (MANDATE: the Safety Kernel is the final authority):

* the only value ever passed to :meth:`Environment.actuate` is
  ``KernelResult.actuator_command`` of a :class:`KernelResult` returned by the
  :class:`SafetyKernel` in the same cycle - from :meth:`SafetyKernel.check` if a
  command was checked this cycle, else from :meth:`SafetyKernel.tick` (watchdog
  timeout / e-stop safe action). With no kernel result the environment is not
  actuated (the last approved command stands, as the kernel's ``tick`` contract
  allows until the watchdog fires);
* :meth:`SafetyKernel.tick` is called at the start of every cycle;
* System 2 output (:class:`PlanProposal`) is never actuated; only the System 1
  :class:`ActionProposal` selected by Awareness ``accept`` is sent to the kernel;
* the kernel's ``position``/``velocity`` come from the physical-layer state
  estimate (variables ``pos_<axis>`` / ``vel_<axis>``); missing values are
  passed as NaN so the kernel rejects the command as ``invalid_state``;
* a module that raises is logged at level ``error`` and its output treated as
  absent (no command this cycle); an actuator or runner-telemetry failure
  latches the kernel's emergency stop.

Determinism: the runner holds no global state; all randomness comes from the
``numpy.random.Generator`` given at construction, which is passed to every
module call in a fixed order. Message and runner event ids are derived from
``run_id`` and the cycle id. Time comes from the injected ``clock`` (it must use
the kernel's time base: it stamps command envelopes and is passed to the kernel
as ``now``). Per-module latency is measured with :func:`time.perf_counter` and
written to each telemetry record, so :meth:`TelemetryLog.summary` reports it per
component.
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Mapping, Optional, Protocol, Union, runtime_checkable

import numpy as np

from contracts import (
    ActionProposal,
    AwarenessDecision,
    Envelope,
    Observation,
    Outcome,
    PlanProposal,
    PredictionRequest,
    PredictionResult,
    StateUpdate,
    Uncertainty,
)
from safety import KernelResult, SafetyKernel
from state.telemetry import TelemetryError, TelemetryLog

RUNNER_VERSION = "cycle-runner-0.1.0"

# Telemetry component names, one per stage.
C_OBSERVE = "env_observe"
C_ESTIMATE = "state_estimator"
C_PREDICT = "world_model"
C_S1 = "system1"
C_S2 = "system2"
C_AWARENESS = "awareness"
C_ACTUATE = "env_actuate"
C_CYCLE = "runner"


# ---------------------------------------------------------------------------
# Module interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class Environment(Protocol):
    def observe(self, rng: np.random.Generator) -> Observation: ...

    def actuate(self, command: tuple[float, ...], rng: np.random.Generator) -> Outcome: ...


@runtime_checkable
class StateEstimator(Protocol):
    def estimate(self, observation: Observation, rng: np.random.Generator) -> StateUpdate: ...


@runtime_checkable
class WorldModel(Protocol):
    def predict(
        self, request: PredictionRequest, rng: np.random.Generator
    ) -> Optional[PredictionResult]: ...


@runtime_checkable
class System1(Protocol):
    def propose(
        self,
        state: StateUpdate,
        prediction: Optional[PredictionResult],
        rng: np.random.Generator,
    ) -> Optional[ActionProposal]: ...


@runtime_checkable
class System2(Protocol):
    def plan(
        self,
        state: StateUpdate,
        prediction: Optional[PredictionResult],
        proposal: Optional[ActionProposal],
        rng: np.random.Generator,
    ) -> Optional[PlanProposal]: ...


@runtime_checkable
class Awareness(Protocol):
    def arbitrate(
        self,
        state: StateUpdate,
        prediction: Optional[PredictionResult],
        proposal: Optional[ActionProposal],
        plan: Optional[PlanProposal],
        rng: np.random.Generator,
    ) -> AwarenessDecision: ...


# ---------------------------------------------------------------------------
# Null / pass-through defaults
# ---------------------------------------------------------------------------


class PassThroughStateEstimator:
    """Copies the observation channels into a physical-layer state."""

    version = "passthrough-estimator-0.1.0"

    def estimate(self, observation: Observation, rng: np.random.Generator) -> StateUpdate:
        return StateUpdate(
            layer="physical",
            variables=observation.channels,
            values=observation.values,
            uncertainty=observation.uncertainty,
            source_message_ids=(),
        )


class NullWorldModel:
    version = "null-world-model-0.1.0"

    def predict(
        self, request: PredictionRequest, rng: np.random.Generator
    ) -> Optional[PredictionResult]:
        return None


class NullSystem1:
    version = "null-system1-0.1.0"

    def propose(self, state, prediction, rng) -> Optional[ActionProposal]:
        return None


class NullSystem2:
    version = "null-system2-0.1.0"

    def plan(self, state, prediction, proposal, rng) -> Optional[PlanProposal]:
        return None


class PassThroughAwareness:
    """Accepts the System 1 proposal when there is one, else abstains."""

    version = "passthrough-awareness-0.1.0"

    def arbitrate(self, state, prediction, proposal, plan, rng) -> AwarenessDecision:
        if proposal is None:
            return AwarenessDecision("abstain", "no proposal", (), None)
        return AwarenessDecision(
            "accept", "pass-through", (proposal.proposal_id,), proposal.proposal_id
        )


class StaticEnvironment:
    """Null environment: the robot rests at the origin; commands have no effect."""

    version = "static-environment-0.1.0"

    def __init__(self, axis_names: tuple[str, ...] = ("x",)) -> None:
        self.axis_names = tuple(axis_names)

    def observe(self, rng: np.random.Generator) -> Observation:
        names = tuple(f"pos_{a}" for a in self.axis_names) + tuple(
            f"vel_{a}" for a in self.axis_names
        )
        return Observation("static", names, (0.0,) * len(names), Uncertainty())

    def actuate(self, command: tuple[float, ...], rng: np.random.Generator) -> Outcome:
        return Outcome("static", True, 0.0, (), (), Uncertainty())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_INTERFACES: dict[str, type] = {
    "environment": Environment,
    "state_estimator": StateEstimator,
    "world_model": WorldModel,
    "system1": System1,
    "system2": System2,
    "awareness": Awareness,
}


def load_component(spec: str, **kwargs: Any) -> Any:
    """Instantiate ``"package.module:Factory"`` (``src`` is on sys.path)."""
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError(f"module spec must be 'package.module:Factory', got {spec!r}")
    mod_name, attr = spec.split(":")
    factory = getattr(importlib.import_module(mod_name), attr)
    return factory(**kwargs)


@dataclass(frozen=True)
class Modules:
    """The swappable modules of one runner."""

    environment: Any = field(default_factory=StaticEnvironment)
    state_estimator: Any = field(default_factory=PassThroughStateEstimator)
    world_model: Any = field(default_factory=NullWorldModel)
    system1: Any = field(default_factory=NullSystem1)
    system2: Any = field(default_factory=NullSystem2)
    awareness: Any = field(default_factory=PassThroughAwareness)

    def __post_init__(self) -> None:
        for f in fields(self):
            proto = _INTERFACES[f.name]
            if not isinstance(getattr(self, f.name), proto):
                raise TypeError(
                    f"Modules.{f.name}: {type(getattr(self, f.name)).__name__} "
                    f"does not implement {proto.__name__}"
                )

    @classmethod
    def from_config(cls, config: Mapping[str, Union[str, Mapping[str, Any]]]) -> "Modules":
        """Build from ``{slot: "pkg.mod:Factory"}`` or ``{slot: {"factory": ..., "kwargs": {...}}}``.

        Slots that are not configured keep their null / pass-through default.
        """
        unknown = sorted(set(config) - set(_INTERFACES))
        if unknown:
            raise ValueError(f"unknown module slot(s) {unknown}; known: {sorted(_INTERFACES)}")
        built = {}
        for slot, spec in config.items():
            if isinstance(spec, Mapping):
                built[slot] = load_component(spec["factory"], **dict(spec.get("kwargs", {})))
            else:
                built[slot] = load_component(spec)
        return cls(**built)

    def replace(self, **changes: Any) -> "Modules":
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RunnerError(RuntimeError):
    """The runner cannot continue (e.g. its telemetry cannot be written)."""


@dataclass(frozen=True)
class CycleResult:
    cycle_id: int
    awareness_decision: str
    kernel_result: Optional[KernelResult]
    actuated_command: Optional[tuple[float, ...]]
    outcome: Optional[Outcome]
    latency_ms: Mapping[str, float]


class _Failed:
    """Marker for a stage whose module raised."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _version(module: Any) -> str:
    v = getattr(module, "version", None)
    return v if isinstance(v, str) and v else type(module).__name__


class CycleRunner:
    """Synchronous control loop; every actuator command passes the Safety Kernel."""

    def __init__(
        self,
        kernel: SafetyKernel,
        telemetry: TelemetryLog,
        rng: np.random.Generator,
        modules: Optional[Modules] = None,
        clock: Callable[[], float] = time.time,
        run_id: str = "run",
    ) -> None:
        if not isinstance(kernel, SafetyKernel):
            raise TypeError(f"kernel must be SafetyKernel, got {type(kernel).__name__}")
        if not isinstance(telemetry, TelemetryLog):
            raise TypeError(f"telemetry must be TelemetryLog, got {type(telemetry).__name__}")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be numpy.random.Generator, got {type(rng).__name__}")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty str")
        self.kernel = kernel
        self.telemetry = telemetry
        self.rng = rng
        self.modules = Modules() if modules is None else modules
        if not isinstance(self.modules, Modules):
            raise TypeError(f"modules must be Modules, got {type(self.modules).__name__}")
        self.clock = clock
        self.run_id = run_id
        self.cycle_id = 0
        self._event_seq = 0
        self._last_command: tuple[float, ...] = ()

    # -- public API ------------------------------------------------------------

    def run(self, n_cycles: int) -> list[CycleResult]:
        if isinstance(n_cycles, bool) or not isinstance(n_cycles, int) or n_cycles < 0:
            raise ValueError(f"n_cycles must be a non-negative int, got {n_cycles!r}")
        return [self.step() for _ in range(n_cycles)]

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per-component latency summary (count, p50_ms, p95_ms) from the telemetry log."""
        return self.telemetry.summary()

    def step(self) -> CycleResult:
        self.cycle_id += 1
        cid = self.cycle_id
        t_cycle = time.perf_counter()
        m = self.modules
        rng = self.rng
        lat: dict[str, float] = {}

        # Watchdog, every cycle.
        tick = self.kernel.tick(now=self._now())

        decision = "abstain"
        command_env: Optional[Envelope] = None
        state: Optional[StateUpdate] = None

        obs = self._stage(C_OBSERVE, m.environment, lat, lambda: m.environment.observe(rng),
                          lambda o: ("observed", {"sensor_id": o.sensor_id}))
        if not isinstance(obs, _Failed):
            state = self._stage(
                C_ESTIMATE, m.state_estimator, lat,
                lambda: m.state_estimator.estimate(obs, rng),
                lambda s: ("estimated", {"layer": s.layer, "n_variables": len(s.variables)}),
            )
        if isinstance(state, StateUpdate):
            decision, command_env = self._decide(cid, state, lat)
        else:
            state = None

        # Safety Kernel: the only source of actuator commands.
        kernel_result: Optional[KernelResult] = tick
        if command_env is not None:
            pos, vel = self._kinematics(state)
            kernel_result = self.kernel.check(
                command_env, position=pos, velocity=vel, now=self._now()
            )

        actuated: Optional[tuple[float, ...]] = None
        outcome: Optional[Outcome] = None
        if kernel_result is not None:
            command = kernel_result.actuator_command
            outcome = self._stage(
                C_ACTUATE, m.environment, lat, lambda: m.environment.actuate(command, rng),
                lambda o: ("actuated", {
                    "command": list(command), "success": o.success, "reward": o.reward,
                }),
            )
            if isinstance(outcome, _Failed):
                outcome = None
                self.kernel.emergency_stop("actuator failure", now=self._now())
            actuated = command
            self._last_command = command

        total = (time.perf_counter() - t_cycle) * 1000.0
        verdict = None if kernel_result is None else kernel_result.decision.verdict
        self._log(
            C_CYCLE, RUNNER_VERSION, decision,
            "cycle complete", total,
            data={
                "kernel_verdict": verdict,
                "kernel_source": None if kernel_result is None
                else ("check" if command_env is not None else "tick"),
                "actuated_command": None if actuated is None else list(actuated),
            },
        )
        lat[C_CYCLE] = total
        return CycleResult(cid, decision, kernel_result, actuated, outcome, lat)

    # -- internals ---------------------------------------------------------------

    def _decide(
        self, cid: int, state: StateUpdate, lat: dict[str, float]
    ) -> tuple[str, Optional[Envelope]]:
        """World model -> S1 -> Awareness (-> S2 -> Awareness). Returns (decision, command)."""
        m, rng = self.modules, self.rng
        request = PredictionRequest(
            variables=state.variables,
            state=state.values,
            action=self._last_command,
            horizon_steps=1,
            dt=self.kernel.config.control_dt_s,
        )
        prediction = self._stage(
            C_PREDICT, m.world_model, lat, lambda: m.world_model.predict(request, rng),
            lambda p: ("no_prediction", {}) if p is None
            else ("predicted", {"model_version": p.model_version}),
        )
        prediction = prediction if isinstance(prediction, PredictionResult) else None

        proposal = self._stage(
            C_S1, m.system1, lat, lambda: m.system1.propose(state, prediction, rng),
            lambda p: ("no_proposal", {}) if p is None else ("proposed", {
                "proposal_id": p.proposal_id, "action": list(p.action),
                "confidence": p.confidence,
            }),
        )
        proposal = proposal if isinstance(proposal, ActionProposal) else None

        plan: Optional[PlanProposal] = None
        arb = self._arbitrate(state, prediction, proposal, plan, lat)
        if arb is not None and arb.decision == "request_s2":
            # ADR-002: slow path only on request; a plan is advice, never a command.
            out = self._stage(
                C_S2, m.system2, lat,
                lambda: m.system2.plan(state, prediction, proposal, rng),
                lambda p: ("no_plan", {}) if p is None else ("planned", {
                    "proposal_id": p.proposal_id, "skill_ids": list(p.skill_ids),
                }),
            )
            plan = out if isinstance(out, PlanProposal) else None
            arb = self._arbitrate(state, prediction, proposal, plan, lat)
        if arb is None:
            return "abstain", None
        if (
            arb.decision != "accept"
            or proposal is None
            or arb.selected_proposal_id != proposal.proposal_id
        ):
            # Includes a second request_s2, request_prediction, replan, escalate,
            # and an accept that selects anything but the S1 ActionProposal.
            return arb.decision, None
        env = Envelope(
            message_id=f"{self.run_id}-c{cid}-cmd",
            timestamp=self._now(),
            source="awareness",
            destination="safety",
            cycle_id=cid,
            correlation_id=f"{self.run_id}-c{cid}",
            payload=proposal,
        )
        return arb.decision, env

    def _arbitrate(self, state, prediction, proposal, plan, lat) -> Optional[AwarenessDecision]:
        m, rng = self.modules, self.rng
        out = self._stage(
            C_AWARENESS, m.awareness, lat,
            lambda: m.awareness.arbitrate(state, prediction, proposal, plan, rng),
            lambda d: (d.decision, {
                "reason": d.reason, "selected_proposal_id": d.selected_proposal_id,
            }),
        )
        return out if isinstance(out, AwarenessDecision) else None

    def _stage(self, component, module, lat, call, describe):
        """Run one module call, time it and log it. Returns the output or _Failed."""
        t0 = time.perf_counter()
        try:
            out = call()
            if out is not None:
                out.validate()  # contract messages only cross module boundaries
            ms = (time.perf_counter() - t0) * 1000.0
            decision, data = describe(out)
            level, reason = "info", "ok"
        except Exception as exc:  # a faulty module must not stop the safe path
            ms = (time.perf_counter() - t0) * 1000.0
            out = _Failed(exc)
            decision, data, level, reason = "error", {}, "error", f"{type(exc).__name__}: {exc}"
        lat[component] = lat.get(component, 0.0) + ms
        self._log(component, _version(module), decision, reason, ms, data=data, level=level)
        return out

    def _kinematics(self, state: Optional[StateUpdate]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        axes = self.kernel.config.axis_names
        values: dict[str, float] = {}
        if state is not None and state.layer == "physical":
            values = dict(zip(state.variables, state.values))
        pos = tuple(values.get(f"pos_{a}", math.nan) for a in axes)
        vel = tuple(values.get(f"vel_{a}", math.nan) for a in axes)
        return pos, vel

    def _now(self) -> float:
        return float(self.clock())

    def _log(self, component, model_version, decision, reason, latency_ms, *, data, level="info"):
        self._event_seq += 1
        n = self._event_seq
        try:
            self.telemetry.log(
                component,
                cycle_id=self.cycle_id,
                decision=decision,
                reason=reason,
                latency_ms=max(0.0, latency_ms),
                model_version=model_version,
                level=level,
                event_id=f"{self.run_id}-c{self.cycle_id}-e{n}",
                timestamp=self._now(),
                data=data,
            )
        except (TelemetryError, ValueError, OSError) as exc:
            # Unobservable control is not safe: stop, then refuse to continue.
            self.kernel.emergency_stop("runner telemetry failure", now=self._now())
            raise RunnerError(f"runner telemetry failed: {exc}") from exc
