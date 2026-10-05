"""Independent envelope round-trip checks for every registered payload."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from contracts import (
    SCHEMA_VERSION,
    ActionProposal,
    AwarenessDecision,
    ContractError,
    Envelope,
    ExperimentEvent,
    LearningEvent,
    MemoryQuery,
    MemoryResult,
    Observation,
    Outcome,
    PAYLOAD_TYPES,
    PlanProposal,
    PredictionRequest,
    PredictionResult,
    SafetyDecision,
    StateUpdate,
    Uncertainty,
)


SAMPLES = (
    Observation("s", ("x",), (1.0,), Uncertainty(measurement=0.1)),
    StateUpdate("belief", ("x",), (1.0,), Uncertainty(estimation=0.1), ("m0",)),
    PredictionRequest(("x",), (1.0,), (0.0,), 1, 0.1),
    PredictionResult("wm1", ("x",), (1.1,), 1, Uncertainty(model=0.1)),
    ActionProposal("p1", "policy1", (0.0,), 0.9, Uncertainty(policy=0.1)),
    PlanProposal("p2", "planner1", "goal", (), 1.0, Uncertainty(outcome=0.1)),
    AwarenessDecision("accept", "selected", ("p1",), "p1"),
    SafetyDecision("approve", "p1", (0.0,), (), "kernel1"),
    Outcome("p1", True, 1.0, ("x",), (1.1,), Uncertainty(outcome=0.1)),
    LearningEvent("update", "world_model", "v1", "v2", (), ()),
    MemoryQuery("episodic", (), (), 1),
    MemoryResult("q1", (), (), ()),
    ExperimentEvent("e1", "end", "baseline", 1, (), ()),
)


def test_every_registered_payload_round_trips_in_versioned_envelope():
    assert {type(sample) for sample in SAMPLES} == set(PAYLOAD_TYPES.values())
    for index, payload in enumerate(SAMPLES):
        envelope = Envelope(
            f"m{index}", float(index), "source", "destination", index, "corr", payload
        )
        decoded = Envelope.from_json(envelope.to_json())
        assert decoded == envelope
        assert decoded.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("bad_version", ["0.9.0", "2.0.0"])
def test_envelope_decoder_rejects_other_schema_versions(bad_version):
    payload = SAMPLES[0]
    wire = Envelope("m", 0.0, "s", "d", 0, "c", payload).to_dict()
    wire["schema_version"] = bad_version
    import json

    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(wire))
