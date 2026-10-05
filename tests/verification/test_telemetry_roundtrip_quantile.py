"""Independent checks for the telemetry record contract and summary."""

import json

from state.telemetry import TelemetryLog, TelemetryRecord, read_records


def test_jsonl_round_trip_and_per_component_quantiles(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    entries = [
        TelemetryRecord(1.0, "planner", "info", i, f"event-{i}", "v1",
                        "continue", "ok", latency)
        for i, latency in enumerate((10.0, 20.0, 30.0, 40.0, 50.0))
    ]
    with TelemetryLog(path) as log:
        for record in entries:
            log.write(record)
        assert log.summary()["planner"] == {
            "count": 5, "p50_ms": 30.0, "p95_ms": 48.0
        }

    assert read_records(path) == entries
    assert [json.loads(line)["event_id"] for line in path.read_text().splitlines()] == [
        record.event_id for record in entries
    ]
