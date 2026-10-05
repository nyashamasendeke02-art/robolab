"""Versioned message contracts (Gate 0; docs/contracts.md is the reference).

Every message is an immutable `Envelope` carrying one typed payload. Values are
normalised on construction (lists -> tuples, dicts -> read-only mappings, ints in
float fields -> floats) and validated; any violation raises `ContractError`.
"""

from __future__ import annotations

import collections.abc
import json
import math
import types
import typing
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Mapping, Optional, Union

SCHEMA_VERSION = "1.0.0"


class ContractError(ValueError):
    """A message does not satisfy its contract."""


# --------------------------------------------------------------------------- helpers


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _decode(value: Any, tp: Any, path: str) -> Any:
    """Check `value` against type `tp` and return its normalised, immutable form."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is Union:  # Optional[T]
        inner = [a for a in args if a is not type(None)]
        if value is None:
            return None
        return _decode(value, inner[0], path)
    if value is None:
        raise ContractError(f"{path}: must not be null")

    if tp is bool:
        if not isinstance(value, bool):
            raise ContractError(f"{path}: expected bool, got {type(value).__name__}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{path}: expected int, got {type(value).__name__}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"{path}: expected float, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value):
            raise ContractError(f"{path}: non-finite number {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ContractError(f"{path}: expected str, got {type(value).__name__}")
        return value
    if origin is tuple:  # tuple[T, ...]
        if not isinstance(value, (list, tuple)):
            raise ContractError(f"{path}: expected array, got {type(value).__name__}")
        return tuple(_decode(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
    if origin is collections.abc.Mapping:  # Mapping[str, T]
        if not isinstance(value, (dict, types.MappingProxyType)):
            raise ContractError(f"{path}: expected object, got {type(value).__name__}")
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ContractError(f"{path}: keys must be str, got {type(k).__name__}")
            out[k] = _decode(v, args[1], f"{path}.{k}")
        return types.MappingProxyType(out)
    if isinstance(tp, type) and issubclass(tp, _Contract):
        if isinstance(value, tp):
            value.validate()
            return value
        if isinstance(value, dict):
            return tp.from_dict(value, path=path)
        raise ContractError(f"{path}: expected {tp.__name__}, got {type(value).__name__}")
    raise ContractError(f"{path}: unsupported contract type {_type_name(tp)}")  # pragma: no cover


def _encode(value: Any) -> Any:
    if isinstance(value, _Contract):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    if isinstance(value, (dict, types.MappingProxyType)):
        return {k: _encode(v) for k, v in value.items()}
    return value


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _non_empty(value: str, path: str) -> None:
    _require(value != "", f"{path}: must be a non-empty string")


def _unit_interval(value: float, path: str) -> None:
    _require(0.0 <= value <= 1.0, f"{path}: must be in [0, 1], got {value}")


# --------------------------------------------------------------------------- base


@dataclass(frozen=True)
class _Contract:
    """Base for every contract dataclass: typed fields, normalisation, validation."""

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def _hints(cls) -> dict[str, Any]:
        return typing.get_type_hints(cls)

    def validate(self) -> None:
        """Check every field's type and the class invariants; raise ContractError."""
        hints = self._hints()
        name = type(self).__name__
        for f in fields(self):
            normalised = _decode(getattr(self, f.name), hints[f.name], f"{name}.{f.name}")
            object.__setattr__(self, f.name, normalised)
        self._check()

    def _check(self) -> None:
        """Class-specific invariants beyond types (override)."""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _encode(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Any, path: str | None = None) -> Any:
        path = path or cls.__name__
        if not isinstance(data, dict):
            raise ContractError(f"{path}: expected object, got {type(data).__name__}")
        names = [f.name for f in fields(cls)]
        missing = [n for n in names if n not in data]
        if missing:
            raise ContractError(f"{path}: missing field(s) {missing}")
        unknown = sorted(set(data) - set(names))
        if unknown:
            raise ContractError(f"{path}: unknown field(s) {unknown}")
        hints = cls._hints()
        kwargs = {n: _decode(data[n], hints[n], f"{path}.{n}") for n in names}
        return cls(**kwargs)


# --------------------------------------------------------------------------- uncertainty


@dataclass(frozen=True)
class Uncertainty(_Contract):
    """Separate, typed uncertainty sources; None means "not estimated".

    Each value is a non-negative scalar spread (e.g. a standard deviation) in the
    units of the quantity the carrying message describes.
    """

    measurement: Optional[float] = None
    estimation: Optional[float] = None
    model: Optional[float] = None
    policy: Optional[float] = None
    outcome: Optional[float] = None

    def _check(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            _require(v is None or v >= 0.0, f"Uncertainty.{f.name}: must be >= 0, got {v}")


# --------------------------------------------------------------------------- payloads


@dataclass(frozen=True)
class Payload(_Contract):
    """Base for message payloads; MESSAGE_TYPE is the wire discriminator."""

    MESSAGE_TYPE: ClassVar[str] = ""


@dataclass(frozen=True)
class Observation(Payload):
    MESSAGE_TYPE: ClassVar[str] = "Observation"
    sensor_id: str
    modality: str
    values: tuple[float, ...]
    uncertainty: Uncertainty

    def _check(self) -> None:
        _non_empty(self.sensor_id, "Observation.sensor_id")
        _non_empty(self.modality, "Observation.modality")


@dataclass(frozen=True)
class StateUpdate(Payload):
    MESSAGE_TYPE: ClassVar[str] = "StateUpdate"
    state_id: str
    physical: Mapping[str, float]
    belief: Mapping[str, float]
    task: Mapping[str, float]
    meta: Mapping[str, float]
    uncertainty: Uncertainty

    def _check(self) -> None:
        _non_empty(self.state_id, "StateUpdate.state_id")


@dataclass(frozen=True)
class PredictionRequest(Payload):
    MESSAGE_TYPE: ClassVar[str] = "PredictionRequest"
    state_id: str
    action: tuple[float, ...]
    horizon_steps: int

    def _check(self) -> None:
        _non_empty(self.state_id, "PredictionRequest.state_id")
        _require(self.horizon_steps >= 1, "PredictionRequest.horizon_steps: must be >= 1")


@dataclass(frozen=True)
class PredictionResult(Payload):
    MESSAGE_TYPE: ClassVar[str] = "PredictionResult"
    state_id: str
    horizon_steps: int
    predicted_state: Mapping[str, float]
    model_version: str
    uncertainty: Uncertainty

    def _check(self) -> None:
        _non_empty(self.state_id, "PredictionResult.state_id")
        _non_empty(self.model_version, "PredictionResult.model_version")
        _require(self.horizon_steps >= 1, "PredictionResult.horizon_steps: must be >= 1")


@dataclass(frozen=True)
class ActionProposal(Payload):
    MESSAGE_TYPE: ClassVar[str] = "ActionProposal"
    proposer: str
    policy_version: str
    action: tuple[float, ...]
    confidence: float
    uncertainty: Uncertainty

    def _check(self) -> None:
        _non_empty(self.proposer, "ActionProposal.proposer")
        _non_empty(self.policy_version, "ActionProposal.policy_version")
        _unit_interval(self.confidence, "ActionProposal.confidence")


@dataclass(frozen=True)
class PlanProposal(Payload):
    MESSAGE_TYPE: ClassVar[str] = "PlanProposal"
    planner: str
    skill_ids: tuple[str, ...]
    expected_cost: float
    confidence: float
    uncertainty: Uncertainty

    def _check(self) -> None:
        _non_empty(self.planner, "PlanProposal.planner")
        _unit_interval(self.confidence, "PlanProposal.confidence")


AWARENESS_DECISIONS = (
    "accept",
    "request_s2",
    "request_prediction",
    "replan",
    "abstain",
    "escalate",
)


@dataclass(frozen=True)
class AwarenessDecision(Payload):
    MESSAGE_TYPE: ClassVar[str] = "AwarenessDecision"
    decision: str
    reasons: tuple[str, ...]
    proposal_message_id: Optional[str]

    def _check(self) -> None:
        _require(
            self.decision in AWARENESS_DECISIONS,
            f"AwarenessDecision.decision: {self.decision!r} not in {AWARENESS_DECISIONS}",
        )
        _require(
            self.decision != "accept" or self.proposal_message_id is not None,
            "AwarenessDecision.proposal_message_id: required when decision is 'accept'",
        )


@dataclass(frozen=True)
class SafetyDecision(Payload):
    MESSAGE_TYPE: ClassVar[str] = "SafetyDecision"
    approved: bool
    proposal_message_id: str
    command: tuple[float, ...]
    violations: tuple[str, ...]
    safe_state: bool

    def _check(self) -> None:
        _non_empty(self.proposal_message_id, "SafetyDecision.proposal_message_id")
        _require(
            not self.approved or not self.safe_state,
            "SafetyDecision: an approved command cannot also enter the safe state",
        )


@dataclass(frozen=True)
class Outcome(Payload):
    MESSAGE_TYPE: ClassVar[str] = "Outcome"
    success: bool
    reward: float
    achieved_state: Mapping[str, float]
    uncertainty: Uncertainty


@dataclass(frozen=True)
class LearningEvent(Payload):
    MESSAGE_TYPE: ClassVar[str] = "LearningEvent"
    event_kind: str
    component: str
    from_version: str
    to_version: str
    metrics: Mapping[str, float]

    def _check(self) -> None:
        _non_empty(self.event_kind, "LearningEvent.event_kind")
        _non_empty(self.component, "LearningEvent.component")


@dataclass(frozen=True)
class MemoryQuery(Payload):
    MESSAGE_TYPE: ClassVar[str] = "MemoryQuery"
    query_kind: str
    key: tuple[float, ...]
    top_k: int

    def _check(self) -> None:
        _non_empty(self.query_kind, "MemoryQuery.query_kind")
        _require(self.top_k >= 1, "MemoryQuery.top_k: must be >= 1")


@dataclass(frozen=True)
class MemoryResult(Payload):
    MESSAGE_TYPE: ClassVar[str] = "MemoryResult"
    episode_ids: tuple[str, ...]
    scores: tuple[float, ...]

    def _check(self) -> None:
        _require(
            len(self.episode_ids) == len(self.scores),
            "MemoryResult: episode_ids and scores must have equal length",
        )


@dataclass(frozen=True)
class ExperimentEvent(Payload):
    MESSAGE_TYPE: ClassVar[str] = "ExperimentEvent"
    experiment_id: str
    event_kind: str
    condition: str
    seed: int
    metrics: Mapping[str, float]

    def _check(self) -> None:
        _non_empty(self.experiment_id, "ExperimentEvent.experiment_id")
        _non_empty(self.event_kind, "ExperimentEvent.event_kind")


PAYLOAD_TYPES: Mapping[str, type[Payload]] = types.MappingProxyType(
    {
        cls.MESSAGE_TYPE: cls
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
)


# --------------------------------------------------------------------------- envelope


@dataclass(frozen=True)
class Envelope:
    """Required metadata around one payload (docs/MANDATE.md, Contracts)."""

    message_id: str
    timestamp: float
    source: str
    destination: str
    cycle_id: int
    correlation_id: str
    payload: Payload
    schema_version: str = field(default=SCHEMA_VERSION)

    _META: ClassVar[dict[str, Any]] = {
        "message_id": str,
        "timestamp": float,
        "source": str,
        "destination": str,
        "schema_version": str,
        "cycle_id": int,
        "correlation_id": str,
    }

    def __post_init__(self) -> None:
        self.validate()

    @property
    def message_type(self) -> str:
        return self.payload.MESSAGE_TYPE

    def validate(self) -> None:
        for name, tp in self._META.items():
            object.__setattr__(self, name, _decode(getattr(self, name), tp, f"Envelope.{name}"))
        _require(
            self.schema_version == SCHEMA_VERSION,
            f"Envelope.schema_version: {self.schema_version!r} != {SCHEMA_VERSION!r}",
        )
        for name in ("message_id", "source", "destination", "correlation_id"):
            _non_empty(getattr(self, name), f"Envelope.{name}")
        _require(self.timestamp >= 0.0, "Envelope.timestamp: must be >= 0")
        _require(self.cycle_id >= 0, "Envelope.cycle_id: must be >= 0")
        _require(
            type(self.payload) in PAYLOAD_TYPES.values(),
            f"Envelope.payload: unknown payload type {type(self.payload).__name__}",
        )
        self.payload.validate()

    def to_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in self._META}
        out["message_type"] = self.message_type
        out["payload"] = self.payload.to_dict()
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Any) -> Envelope:
        if not isinstance(data, dict):
            raise ContractError(f"Envelope: expected object, got {type(data).__name__}")
        expected = set(cls._META) | {"message_type", "payload"}
        missing = sorted(expected - set(data))
        if missing:
            raise ContractError(f"Envelope: missing field(s) {missing}")
        unknown = sorted(set(data) - expected)
        if unknown:
            raise ContractError(f"Envelope: unknown field(s) {unknown}")
        # Check the version before interpreting the payload under this schema.
        version = _decode(data["schema_version"], str, "Envelope.schema_version")
        _require(
            version == SCHEMA_VERSION,
            f"Envelope.schema_version: {version!r} != {SCHEMA_VERSION!r}",
        )
        message_type = data["message_type"]
        if not isinstance(message_type, str) or message_type not in PAYLOAD_TYPES:
            raise ContractError(f"Envelope.message_type: unknown message type {message_type!r}")
        payload = PAYLOAD_TYPES[message_type].from_dict(data["payload"])
        return cls(payload=payload, **{n: data[n] for n in cls._META})

    @classmethod
    def from_json(cls, text: str) -> Envelope:
        if not isinstance(text, (str, bytes, bytearray)):
            raise ContractError(f"Envelope: expected JSON text, got {type(text).__name__}")
        try:
            data = json.loads(text)
        except (ValueError, RecursionError) as exc:
            raise ContractError(f"Envelope: invalid JSON ({exc})") from exc
        return cls.from_dict(data)
