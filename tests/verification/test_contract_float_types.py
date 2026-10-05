"""Independent checks for the G0-1 wire contracts."""

import sys
from pathlib import Path

import pytest

# The verifier invokes pytest with -P -E; make the repository's src package
# discoverable without relying on site customization or pytest configuration.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from contracts import ContractError, Envelope, Observation, Uncertainty


def test_valid_float_timestamp_round_trips():
    payload = Observation(sensor_id="sensor", channels=("x",), values=(1.0,), uncertainty=Uncertainty())
    env = Envelope("m1", 1.0, "sensor", "state", 0, "c1", payload)
    assert Envelope.from_json(env.to_json()) == env


def test_float_annotated_fields_reject_integer_values():
    """Wire numeric fields declared float must not silently retain ints."""
    payload = Observation(sensor_id="sensor", channels=("x",), values=(1.0,), uncertainty=Uncertainty())
    with pytest.raises(ContractError):
        Envelope("m1", 1, "sensor", "state", 0, "c1", payload)
