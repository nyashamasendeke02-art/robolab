"""Versioned message contracts (Gate 0, REQ-LOG, REQ-STATE, ADR-001).

Every message exchanged between components is an :class:`Envelope` carrying one
payload object. All classes are frozen dataclasses whose fields are validated on
construction, so an invalid message cannot exist in memory. JSON is the wire
format; ``Envelope.from_json(env.to_json()) == env`` for every valid envelope.

Field types are restricted to ``str``, ``int``, ``float``, ``bool``, nested
contract dataclasses, ``Optional[...]`` of those and ``tuple[X, ...]``. Mappings
are deliberately not used so that messages stay immutable and hashable; named
vectors are expressed as parallel ``names`` / ``values`` tuples of equal length.

See ``docs/contracts.md`` for the documentation of every class and field. Any
change to a class or field here must bump :data:`SCHEMA_VERSION`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
import typing
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

SCHEMA_VERSION = "1.0.0"


class ContractError(ValueError):
    """Raised when a message violates its contract."""


# ---------------------------------------------------------------------------
# Generic, annotation-driven validation and (de)coding
# ---------------------------------------------------------------------------

_HINTS_CACHE: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    if cls not in _HINTS_CACHE:
        hints = typing.get_type_hints(cls)
        _HINTS_CACHE[cls] = {f.name: hints[f.name] for f in dataclasses.fields(cls)}
    return _HINTS_CACHE[cls]


def _optional_inner(tp: Any) -> Any:
    """Return X if ``tp`` is ``Optional[X]``, else None."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(tp)) == 2:
            return args[0]
    return None


def _check(value: Any, tp: Any, path: str) -> None:
    inner = _optional_inner(tp)
    if inner is not None:
        if value is not None:
            _check(value, inner, path)
        return
    if tp is str:
        if not isinstance(value, str):
            raise ContractError(f"{path}: expected str, got {type(value).__name__}")
    elif tp is bool:
        if not isinstance(value, bool):
            raise ContractError(f"{path}: expected bool, got {type(value).__name__}")
    elif tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{path}: expected int, got {type(value).__name__}")
    elif tp is float:
        # Strict: ints (and bools) are rejected rather than silently retained,
        # so a float field always holds a float. JSON written by ``to_json``
        # always encodes floats with a decimal point / exponent.
        if not isinstance(value, float):
            raise ContractError(f"{path}: expected float, got {type(value).__name__}")
        if not math.isfinite(value):
            raise ContractError(f"{path}: non-finite number {value!r}")
    elif typing.get_origin(tp) is tuple:
        args = typing.get_args(tp)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise TypeError(f"unsupported tuple annotation {tp!r}")
        if not isinstance(value, tuple):
            raise ContractError(f"{path}: expected tuple, got {type(value).__name__}")
        for i, item in enumerate(value):
            _check(item, args[0], f"{path}[{i}]")
    elif dataclasses.is_dataclass(tp):
        if type(value) is not tp:
            raise ContractError(
                f"{path}: expected {tp.__name__}, got {type(value).__name__}"
            )
        value.validate()
    else:
        raise TypeError(f"unsupported annotation {tp!r} at {path}")


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def _decode(raw: Any, tp: Any, path: str) -> Any:
    """Convert JSON-native data to the annotated type (shape only).

    Leaf type checks are left to ``validate()`` (run by the constructors); this
    function only rebuilds tuples and nested dataclasses and reports missing or
    unexpected keys.
    """
    inner = _optional_inner(tp)
    if inner is not None:
        return None if raw is None else _decode(raw, inner, path)
    if typing.get_origin(tp) is tuple:
        if not isinstance(raw, list):
            raise ContractError(f"{path}: expected array, got {type(raw).__name__}")
        item_tp = typing.get_args(tp)[0]
        return tuple(_decode(v, item_tp, f"{path}[{i}]") for i, v in enumerate(raw))
    if dataclasses.is_dataclass(tp):
        return _decode_dataclass(raw, tp, path)
    return raw


def _decode_dataclass(raw: Any, cls: type, path: str) -> Any:
    if not isinstance(raw, dict):
        raise ContractError(f"{path}: expected object, got {type(raw).__name__}")
    hints = _hints(cls)
    missing = sorted(set(hints) - set(raw))
    if missing:
        raise ContractError(f"{path}: missing field(s) {missing}")
    unknown = sorted(set(raw) - set(hints))
    if unknown:
        raise ContractError(f"{path}: unknown field(s) {unknown}")
    kwargs = {name: _decode(raw[name], tp, f"{path}.{name}") for name, tp in hints.items()}
    return cls(**kwargs)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _same_length(obj: Any, *names: str) -> None:
    lengths = {n: len(getattr(obj, n)) for n in names}
    _require(
        len(set(lengths.values())) <= 1,
        f"{type(obj).__name__}: fields must have equal length {lengths}",
    )


def _non_empty(obj: Any, *names: str) -> None:
    for n in names:
        _require(getattr(obj, n) != "", f"{type(obj).__name__}.{n} must be non-empty")


def _one_of(obj: Any, name: str, allowed: tuple[str, ...]) -> None:
    value = getattr(obj, name)
    _require(
        value in allowed,
        f"{type(obj).__name__}.{name}={value!r} not in {list(allowed)}",
    )


class _Contract:
    """Mixin: type-check every annotated field on construction."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        cls = type(self)
        for name, tp in _hints(cls).items():
            _check(getattr(self, name), tp, f"{cls.__name__}.{name}")
        self._validate_semantics()

    def _validate_semantics(self) -> None:  # overridden where needed
        pass

    def to_json(self) -> str:
        self.validate()
        return json.dumps(_encode(self), allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Any:
        try:
            raw = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{cls.__name__}: invalid JSON: {exc}") from exc
        return _decode_dataclass(raw, cls, cls.__name__)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Uncertainty(_Contract):
    """Uncertainty components kept distinguishable (MANDATE: Contracts).

    Each component is a non-negative standard deviation in the units of the
    quantity it qualifies, or ``None`` when that kind of uncertainty is not
    applicable to / not estimated for the message.
    """

    measurement: Optional[float] = None
    estimation: Optional[float] = None
    model: Optional[float] = None
    policy: Optional[float] = None
    outcome: Optional[float] = None

    def _validate_semantics(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            _require(v is None or v >= 0, f"Uncertainty.{f.name} must be >= 0, got {v!r}")


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

STATE_LAYERS = ("physical", "belief", "task", "meta")
AWARENESS_DECISIONS = (
    "accept",
    "request_s2",
    "request_prediction",
    "replan",
    "abstain",
    "escalate",
)
SAFETY_VERDICTS = ("approve", "reject", "emergency_stop")
LEARNING_EVENT_KINDS = (
    "experience",
    "selection",
    "replay",
    "update",
    "validation",
    "deployment",
    "rollback",
)
MEMORY_QUERY_KINDS = ("episodic", "skill")
EXPERIMENT_EVENT_KINDS = ("start", "step", "end", "error")


@dataclass(frozen=True)
class Observation(_Contract):
    sensor_id: str
    channels: tuple[str, ...]
    values: tuple[float, ...]
    uncertainty: Uncertainty

    def _validate_semantics(self) -> None:
        _non_empty(self, "sensor_id")
        _same_length(self, "channels", "values")


@dataclass(frozen=True)
class StateUpdate(_Contract):
    layer: str
    variables: tuple[str, ...]
    values: tuple[float, ...]
    uncertainty: Uncertainty
    source_message_ids: tuple[str, ...]

    def _validate_semantics(self) -> None:
        _one_of(self, "layer", STATE_LAYERS)
        _same_length(self, "variables", "values")


@dataclass(frozen=True)
class PredictionRequest(_Contract):
    variables: tuple[str, ...]
    state: tuple[float, ...]
    action: tuple[float, ...]
    horizon_steps: int
    dt: float

    def _validate_semantics(self) -> None:
        _same_length(self, "variables", "state")
        _require(self.horizon_steps >= 1, "PredictionRequest.horizon_steps must be >= 1")
        _require(self.dt > 0, "PredictionRequest.dt must be > 0")


@dataclass(frozen=True)
class PredictionResult(_Contract):
    model_version: str
    variables: tuple[str, ...]
    predicted: tuple[float, ...]
    horizon_steps: int
    uncertainty: Uncertainty

    def _validate_semantics(self) -> None:
        _non_empty(self, "model_version")
        _same_length(self, "variables", "predicted")
        _require(self.horizon_steps >= 1, "PredictionResult.horizon_steps must be >= 1")


@dataclass(frozen=True)
class ActionProposal(_Contract):
    proposal_id: str
    policy_version: str
    action: tuple[float, ...]
    confidence: float
    uncertainty: Uncertainty

    def _validate_semantics(self) -> None:
        _non_empty(self, "proposal_id", "policy_version")
        _require(0.0 <= self.confidence <= 1.0, "ActionProposal.confidence must be in [0, 1]")


@dataclass(frozen=True)
class PlanProposal(_Contract):
    proposal_id: str
    planner_version: str
    goal: str
    skill_ids: tuple[str, ...]
    expected_cost: float
    uncertainty: Uncertainty

    def _validate_semantics(self) -> None:
        _non_empty(self, "proposal_id", "planner_version")


@dataclass(frozen=True)
class AwarenessDecision(_Contract):
    decision: str
    reason: str
    proposal_ids: tuple[str, ...]
    selected_proposal_id: Optional[str]

    def _validate_semantics(self) -> None:
        _one_of(self, "decision", AWARENESS_DECISIONS)
        if self.decision == "accept":
            _require(
                self.selected_proposal_id is not None,
                "AwarenessDecision: 'accept' requires selected_proposal_id",
            )


@dataclass(frozen=True)
class SafetyDecision(_Contract):
    verdict: str
    proposal_id: str
    approved_action: tuple[float, ...]
    violated_constraints: tuple[str, ...]
    kernel_version: str

    def _validate_semantics(self) -> None:
        _one_of(self, "verdict", SAFETY_VERDICTS)
        _non_empty(self, "proposal_id", "kernel_version")
        if self.verdict != "approve":
            _require(
                self.approved_action == (),
                "SafetyDecision: a non-approve verdict must not carry an approved_action",
            )


@dataclass(frozen=True)
class Outcome(_Contract):
    action_id: str
    success: bool
    reward: float
    variables: tuple[str, ...]
    observed: tuple[float, ...]
    uncertainty: Uncertainty

    def _validate_semantics(self) -> None:
        _non_empty(self, "action_id")
        _same_length(self, "variables", "observed")


@dataclass(frozen=True)
class LearningEvent(_Contract):
    kind: str
    module: str
    version_before: str
    version_after: str
    metric_names: tuple[str, ...]
    metric_values: tuple[float, ...]

    def _validate_semantics(self) -> None:
        _one_of(self, "kind", LEARNING_EVENT_KINDS)
        _non_empty(self, "module")
        _same_length(self, "metric_names", "metric_values")


@dataclass(frozen=True)
class MemoryQuery(_Contract):
    kind: str
    key_names: tuple[str, ...]
    key_values: tuple[float, ...]
    max_results: int

    def _validate_semantics(self) -> None:
        _one_of(self, "kind", MEMORY_QUERY_KINDS)
        _same_length(self, "key_names", "key_values")
        _require(self.max_results >= 1, "MemoryQuery.max_results must be >= 1")


@dataclass(frozen=True)
class MemoryResult(_Contract):
    query_message_id: str
    record_ids: tuple[str, ...]
    scores: tuple[float, ...]
    provenance: tuple[str, ...]

    def _validate_semantics(self) -> None:
        _non_empty(self, "query_message_id")
        _same_length(self, "record_ids", "scores", "provenance")


@dataclass(frozen=True)
class ExperimentEvent(_Contract):
    experiment_id: str
    kind: str
    condition: str
    seed: int
    metric_names: tuple[str, ...]
    metric_values: tuple[float, ...]

    def _validate_semantics(self) -> None:
        _non_empty(self, "experiment_id")
        _one_of(self, "kind", EXPERIMENT_EVENT_KINDS)
        _same_length(self, "metric_names", "metric_values")


PAYLOAD_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        Observation,
        StateUpdate,
        PredictionRequest,
        PredictionResult,
        ActionProposal,
        PlanProposal,
        AwarenessDecision,
        SafetyDecision,
        Outcome,
        LearningEvent,
        MemoryQuery,
        MemoryResult,
        ExperimentEvent,
    )
}


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Envelope(_Contract):
    """Metadata wrapper around one payload.

    JSON form: the envelope fields plus ``message_type`` (the payload class
    name), with ``payload`` as a nested object.
    """

    message_id: str
    timestamp: float
    source: str
    destination: str
    cycle_id: int
    correlation_id: str
    payload: Any
    schema_version: str = SCHEMA_VERSION

    _ENVELOPE_FIELDS: ClassVar[tuple[str, ...]] = (
        "message_id",
        "timestamp",
        "source",
        "destination",
        "schema_version",
        "cycle_id",
        "correlation_id",
    )

    @property
    def message_type(self) -> str:
        return type(self.payload).__name__

    def validate(self) -> None:
        _check(self.message_id, str, "Envelope.message_id")
        _check(self.timestamp, float, "Envelope.timestamp")
        _check(self.source, str, "Envelope.source")
        _check(self.destination, str, "Envelope.destination")
        _check(self.schema_version, str, "Envelope.schema_version")
        _check(self.cycle_id, int, "Envelope.cycle_id")
        _check(self.correlation_id, str, "Envelope.correlation_id")
        _require(
            self.schema_version == SCHEMA_VERSION,
            f"Envelope.schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}",
        )
        _non_empty(self, "message_id", "source", "destination", "correlation_id")
        _require(self.timestamp >= 0, "Envelope.timestamp must be >= 0")
        _require(self.cycle_id >= 0, "Envelope.cycle_id must be >= 0")
        cls = PAYLOAD_TYPES.get(self.message_type)
        _require(
            cls is not None and type(self.payload) is cls,
            f"Envelope.payload: unknown message type {self.message_type!r}",
        )
        self.payload.validate()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: getattr(self, name) for name in self._ENVELOPE_FIELDS}
        out["message_type"] = self.message_type
        out["payload"] = _encode(self.payload)
        return out

    def to_json(self) -> str:
        self.validate()
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Any) -> "Envelope":
        if not isinstance(raw, dict):
            raise ContractError(f"Envelope: expected object, got {type(raw).__name__}")
        expected = set(cls._ENVELOPE_FIELDS) | {"message_type", "payload"}
        missing = sorted(expected - set(raw))
        if missing:
            raise ContractError(f"Envelope: missing field(s) {missing}")
        unknown = sorted(set(raw) - expected)
        if unknown:
            raise ContractError(f"Envelope: unknown field(s) {unknown}")
        # Version is checked before payload decoding so that a message from a
        # different schema is reported as such rather than as a field error.
        _check(raw["schema_version"], str, "Envelope.schema_version")
        _require(
            raw["schema_version"] == SCHEMA_VERSION,
            f"Envelope.schema_version {raw['schema_version']!r} != {SCHEMA_VERSION!r}",
        )
        mtype = raw["message_type"]
        if not isinstance(mtype, str) or mtype not in PAYLOAD_TYPES:
            raise ContractError(f"Envelope: unknown message type {mtype!r}")
        payload = _decode_dataclass(raw["payload"], PAYLOAD_TYPES[mtype], mtype)
        return cls(
            payload=payload,
            **{name: raw[name] for name in cls._ENVELOPE_FIELDS},
        )

    @classmethod
    def from_json(cls, text: str) -> "Envelope":
        try:
            raw = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Envelope: invalid JSON: {exc}") from exc
        return cls.from_dict(raw)


def _reject_constant(name: str) -> Any:
    raise ContractError(f"non-finite number {name!r} in JSON")
