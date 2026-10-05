"""Per-cycle structured telemetry log (Gate 0, REQ-LOG; ENG-0003).

:class:`TelemetryLog` is an append-only JSONL writer: one JSON object per line,
one line per :class:`TelemetryRecord`. Every record carries the fixed fields
``timestamp, component, level, cycle_id, event_id, model_version, decision,
reason, latency_ms``; any extra, component-specific fields go under ``data``.

Guarantees:

* append-only: the file is opened in append mode and never rewritten; records
  already present (e.g. from an earlier session) are kept and count towards
  the size cap, as do bytes appended by other writers after this log was
  opened (the file size is re-read from the OS before every write);
* each record is validated before anything is written, so an invalid record
  never reaches the file (``ContractError`` from :mod:`contracts`);
* a configurable byte cap: a write that would make the file exceed
  ``max_bytes`` raises :class:`TelemetryCapExceeded` and writes nothing;
* after :meth:`TelemetryLog.close` every write raises :class:`TelemetryClosed`.

:meth:`TelemetryLog.summary` reports per-component latency p50/p95 using linear
interpolation between closest ranks (the "inclusive" / numpy-default method):
for sorted values ``x[0..n-1]`` the q-quantile is at fractional index
``q * (n - 1)``.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from contracts import ContractError

LEVELS = ("debug", "info", "warning", "error", "critical")
RECORD_FIELDS = (
    "timestamp",
    "component",
    "level",
    "cycle_id",
    "event_id",
    "model_version",
    "decision",
    "reason",
    "latency_ms",
    "data",
)
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class TelemetryError(RuntimeError):
    """Base class for telemetry writer failures."""


class TelemetryCapExceeded(TelemetryError):
    """A write would make the log exceed its configured size cap."""


class TelemetryClosed(TelemetryError):
    """A write was attempted after :meth:`TelemetryLog.close`."""


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{path}: expected number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"{path}: non-finite number {value!r}")
    return value


@dataclass(frozen=True)
class TelemetryRecord:
    """One telemetry line. ``data`` holds JSON-serialisable extra fields."""

    timestamp: float
    component: str
    level: str
    cycle_id: int
    event_id: str
    model_version: str
    decision: str
    reason: str
    latency_ms: float
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise numbers (ints accepted for convenience, stored as float).
        object.__setattr__(self, "timestamp", _finite_number(self.timestamp, "timestamp"))
        object.__setattr__(self, "latency_ms", _finite_number(self.latency_ms, "latency_ms"))
        for name in ("component", "level", "event_id", "model_version", "decision", "reason"):
            v = getattr(self, name)
            if not isinstance(v, str):
                raise ContractError(f"{name}: expected str, got {type(v).__name__}")
        for name in ("component", "event_id", "model_version"):
            if getattr(self, name) == "":
                raise ContractError(f"{name} must be non-empty")
        if self.level not in LEVELS:
            raise ContractError(f"level={self.level!r} not in {list(LEVELS)}")
        if isinstance(self.cycle_id, bool) or not isinstance(self.cycle_id, int):
            raise ContractError(f"cycle_id: expected int, got {type(self.cycle_id).__name__}")
        if self.cycle_id < 0:
            raise ContractError(f"cycle_id must be >= 0, got {self.cycle_id}")
        if self.latency_ms < 0:
            raise ContractError(f"latency_ms must be >= 0, got {self.latency_ms}")
        if not isinstance(self.data, Mapping):
            raise ContractError(f"data: expected mapping, got {type(self.data).__name__}")
        if any(not isinstance(k, str) for k in self.data):
            raise ContractError("data: keys must be str")
        # Detach from the caller's mapping and prove it serialises cleanly.
        object.__setattr__(self, "data", json.loads(self._dump_data()))

    def _dump_data(self) -> str:
        try:
            return json.dumps(dict(self.data), allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"data: not JSON-serialisable: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in RECORD_FIELDS}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Any) -> "TelemetryRecord":
        if not isinstance(raw, dict):
            raise ContractError(f"record: expected object, got {type(raw).__name__}")
        missing = sorted(set(RECORD_FIELDS) - set(raw))
        if missing:
            raise ContractError(f"record: missing field(s) {missing}")
        unknown = sorted(set(raw) - set(RECORD_FIELDS))
        if unknown:
            raise ContractError(f"record: unknown field(s) {unknown}")
        return cls(**raw)

    @classmethod
    def from_json(cls, text: str) -> "TelemetryRecord":
        try:
            raw = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"record: invalid JSON: {exc}") from exc
        return cls.from_dict(raw)


def read_records(path: Union[str, os.PathLike]) -> list[TelemetryRecord]:
    """Read every record of a JSONL telemetry file, in file order."""
    with open(path, "r", encoding="utf-8") as fh:
        return [TelemetryRecord.from_json(line) for line in fh if line.strip()]


def percentile(values: list[float], q: float) -> float:
    """q-quantile (0 <= q <= 1) by linear interpolation between closest ranks."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class TelemetryLog:
    """Append-only JSONL telemetry writer with a byte cap."""

    def __init__(
        self,
        path: Union[str, os.PathLike],
        max_bytes: int = DEFAULT_MAX_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError(f"max_bytes must be a positive int, got {max_bytes!r}")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._clock = clock
        # Binary append: byte counts are exact and no newline translation occurs.
        self._fh = open(self.path, "ab")
        self._size = self._fh.seek(0, os.SEEK_END)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def size_bytes(self) -> int:
        if not self._closed:
            self._size = self._current_size()
        return self._size

    def write(self, record: TelemetryRecord) -> TelemetryRecord:
        """Append one validated record; all-or-nothing."""
        if self._closed:
            raise TelemetryClosed(f"telemetry log {self.path} is closed")
        if not isinstance(record, TelemetryRecord):
            raise ContractError(f"expected TelemetryRecord, got {type(record).__name__}")
        line = (record.to_json() + "\n").encode("utf-8")
        # Re-read the real file size: other writers may have appended since
        # open, and a cached counter would let this instance bypass the cap.
        self._size = self._current_size()
        if self._size + len(line) > self.max_bytes:
            raise TelemetryCapExceeded(
                f"writing {len(line)} bytes would exceed cap {self.max_bytes} "
                f"(current size {self._size}) for {self.path}"
            )
        self._fh.write(line)
        self._fh.flush()
        self._size = self._current_size()
        return record

    def _current_size(self) -> int:
        """Size of the file behind the open handle, as seen by the OS."""
        self._fh.flush()
        return os.fstat(self._fh.fileno()).st_size

    def log(
        self,
        component: str,
        *,
        cycle_id: int,
        decision: str,
        reason: str,
        latency_ms: float,
        model_version: str,
        level: str = "info",
        event_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        data: Optional[Mapping[str, Any]] = None,
    ) -> TelemetryRecord:
        """Build and append a record; ``event_id``/``timestamp`` default to uuid4/clock."""
        if self._closed:
            raise TelemetryClosed(f"telemetry log {self.path} is closed")
        record = TelemetryRecord(
            timestamp=self._clock() if timestamp is None else timestamp,
            component=component,
            level=level,
            cycle_id=cycle_id,
            event_id=uuid.uuid4().hex if event_id is None else event_id,
            model_version=model_version,
            decision=decision,
            reason=reason,
            latency_ms=latency_ms,
            data={} if data is None else data,
        )
        return self.write(record)

    def records(self) -> list[TelemetryRecord]:
        """All records in the file, in write order (works after close)."""
        if not self._closed:
            self._fh.flush()
        return read_records(self.path)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per-component ``{"count", "p50_ms", "p95_ms"}`` over the whole file."""
        latencies: dict[str, list[float]] = {}
        for rec in self.records():
            latencies.setdefault(rec.component, []).append(rec.latency_ms)
        return {
            comp: {
                "count": len(xs),
                "p50_ms": percentile(xs, 0.50),
                "p95_ms": percentile(xs, 0.95),
            }
            for comp, xs in sorted(latencies.items())
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._fh.close()

    def __enter__(self) -> "TelemetryLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
