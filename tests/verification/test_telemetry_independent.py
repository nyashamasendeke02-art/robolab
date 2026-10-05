"""Independent checks for telemetry invariants."""

import json

import pytest

from state.telemetry import TelemetryCapExceeded, TelemetryLog


def test_cap_accounts_for_bytes_appended_after_open(tmp_path):
    path = tmp_path / "events.jsonl"
    first = {
        "timestamp": 1.0,
        "component": "sensor",
        "level": "info",
        "cycle_id": 1,
        "event_id": "e1",
        "model_version": "v1",
        "decision": "ok",
        "reason": "ready",
        "latency_ms": 2.0,
        "data": {},
    }
    second = dict(first, event_id="e2", cycle_id=2)
    first_line = (json.dumps(first, sort_keys=True) + "\n").encode()
    second_line = (json.dumps(second, sort_keys=True) + "\n").encode()
    cap = len(first_line) + len(second_line) - 1

    log = TelemetryLog(path, max_bytes=cap)
    try:
        path.write_bytes(first_line)
        with pytest.raises(TelemetryCapExceeded):
            log.log(
                "sensor", cycle_id=2, event_id="e2", model_version="v1",
                decision="ok", reason="ready", latency_ms=2.0, timestamp=1.0,
            )
    finally:
        log.close()
