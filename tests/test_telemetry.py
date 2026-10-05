"""Tests for src/state/telemetry.py (Gate 0, REQ-LOG, ENG-0003)."""

import json
import statistics

import pytest

from contracts import ContractError
from state.telemetry import (
    RECORD_FIELDS,
    TelemetryCapExceeded,
    TelemetryClosed,
    TelemetryLog,
    TelemetryRecord,
    percentile,
    read_records,
)


def _log(tl, i, component="s1", latency=1.0, **kw):
    return tl.log(
        component,
        cycle_id=i,
        event_id=f"e{i}",
        model_version="wm-0.1",
        decision="accept",
        reason=f"r{i}",
        latency_ms=latency,
        timestamp=1000.0 + i,
        **kw,
    )


# 1. Round-trip and order ---------------------------------------------------


def test_records_round_trip_in_order(tmp_path):
    path = tmp_path / "t.jsonl"
    with TelemetryLog(path) as tl:
        written = [
            _log(tl, i, component=["s1", "wm", "safety"][i % 3], latency=0.5 * i,
                 data={"i": i, "vec": [1.0, 2.0], "note": "ü"})
            for i in range(20)
        ]
        assert tl.records() == written
    assert read_records(path) == written
    assert [r.event_id for r in read_records(path)] == [f"e{i}" for i in range(20)]


def test_jsonl_line_shape(tmp_path):
    path = tmp_path / "t.jsonl"
    with TelemetryLog(path) as tl:
        _log(tl, 3, data={"x": 1})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert set(obj) == set(RECORD_FIELDS)
    assert obj["cycle_id"] == 3 and obj["data"] == {"x": 1}
    assert obj["level"] == "info"


def test_append_only_keeps_previous_session(tmp_path):
    path = tmp_path / "t.jsonl"
    with TelemetryLog(path) as tl:
        a = _log(tl, 0)
    with TelemetryLog(path) as tl:
        b = _log(tl, 1)
        assert tl.records() == [a, b]


def test_data_is_detached_from_caller(tmp_path):
    payload = {"k": [1]}
    with TelemetryLog(tmp_path / "t.jsonl") as tl:
        rec = _log(tl, 0, data=payload)
        payload["k"].append(2)
        assert tl.records()[0].data == {"k": [1]} == rec.data


@pytest.mark.parametrize(
    "kw",
    [
        {"latency": float("nan")},
        {"latency": -1.0},
        {"latency": True},
        {"level": "loud"},
        {"component": ""},
        {"data": {"x": float("inf")}},
        {"data": {"x": object()}},
        {"data": {1: "x"}},
    ],
)
def test_invalid_records_rejected_and_not_written(tmp_path, kw):
    path = tmp_path / "t.jsonl"
    with TelemetryLog(path) as tl:
        with pytest.raises(ContractError):
            _log(tl, 0, **kw)
        assert tl.size_bytes == 0
    assert path.read_bytes() == b""


def test_from_json_rejects_unknown_and_missing_fields():
    rec = TelemetryRecord(1.0, "c", "info", 0, "e", "v", "d", "r", 1.0)
    obj = rec.to_dict()
    assert TelemetryRecord.from_json(rec.to_json()) == rec
    with pytest.raises(ContractError):
        TelemetryRecord.from_dict({**obj, "extra": 1})
    del obj["reason"]
    with pytest.raises(ContractError):
        TelemetryRecord.from_dict(obj)
    with pytest.raises(ContractError):
        TelemetryRecord.from_json("{not json")


# 2. Size cap ---------------------------------------------------------------


def test_size_cap_enforced(tmp_path):
    path = tmp_path / "t.jsonl"
    probe = TelemetryRecord(1000.0, "s1", "info", 0, "e0", "wm-0.1", "accept", "r0", 1.0)
    line_len = len((probe.to_json() + "\n").encode("utf-8"))
    cap = 3 * line_len  # exactly three identical-size lines fit
    with TelemetryLog(path, max_bytes=cap) as tl:
        for _ in range(3):
            tl.write(probe)
        assert tl.size_bytes == cap
        with pytest.raises(TelemetryCapExceeded):
            tl.write(probe)
        assert tl.size_bytes == cap
    assert path.stat().st_size == cap
    assert len(read_records(path)) == 3


def test_size_cap_counts_existing_file(tmp_path):
    path = tmp_path / "t.jsonl"
    with TelemetryLog(path) as tl:
        _log(tl, 0)
    existing = path.stat().st_size
    with TelemetryLog(path, max_bytes=existing + 10) as tl:
        with pytest.raises(TelemetryCapExceeded):
            _log(tl, 1)
    assert path.stat().st_size == existing


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_cap_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        TelemetryLog(tmp_path / "t.jsonl", max_bytes=bad)


# 3. Writes after close -----------------------------------------------------


def test_writes_after_close_refused(tmp_path):
    path = tmp_path / "t.jsonl"
    tl = TelemetryLog(path)
    rec = _log(tl, 0)
    tl.close()
    assert tl.closed
    with pytest.raises(TelemetryClosed):
        _log(tl, 1)
    with pytest.raises(TelemetryClosed):
        tl.write(rec)
    tl.close()  # idempotent
    assert read_records(path) == [rec]
    assert tl.summary()["s1"]["count"] == 1  # reading still works


# 4. p50 / p95 --------------------------------------------------------------


def test_percentile_known_samples():
    xs = [float(i) for i in range(1, 101)]
    assert percentile(xs, 0.50) == pytest.approx(50.5)
    assert percentile(xs, 0.95) == pytest.approx(95.05)
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == pytest.approx(4.8)
    assert percentile([7.0], 0.95) == 7.0
    # Independent cross-check against the stdlib "inclusive" method.
    sample = [3.1, 0.2, 9.9, 4.4, 4.4, 12.0, 0.0, 7.5]
    q = statistics.quantiles(sample, n=100, method="inclusive")
    assert percentile(sample, 0.50) == pytest.approx(q[49])
    assert percentile(sample, 0.95) == pytest.approx(q[94])
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_summary_per_component(tmp_path):
    with TelemetryLog(tmp_path / "t.jsonl") as tl:
        # Interleave and shuffle order so the summary must sort per component.
        for i, v in enumerate([5, 1, 4, 2, 3]):
            _log(tl, i, component="s1", latency=float(v))
            _log(tl, 100 + i, component="wm", latency=10.0 * (i + 1))
        s = tl.summary()
    assert set(s) == {"s1", "wm"}
    assert s["s1"] == {"count": 5, "p50_ms": pytest.approx(3.0), "p95_ms": pytest.approx(4.8)}
    assert s["wm"] == {"count": 5, "p50_ms": pytest.approx(30.0), "p95_ms": pytest.approx(48.0)}


def test_summary_empty(tmp_path):
    with TelemetryLog(tmp_path / "t.jsonl") as tl:
        assert tl.summary() == {}
