"""Contract tests for src/contracts (ENG-0001, Gate 0)."""

import dataclasses
import json
import math
from pathlib import Path

import pytest

from contracts import (
    PAYLOAD_TYPES,
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
    PlanProposal,
    PredictionRequest,
    PredictionResult,
    SafetyDecision,
    StateUpdate,
    Uncertainty,
)

U = Uncertainty(measurement=0.01, estimation=0.02, model=0.03, policy=None, outcome=0.05)

SAMPLES = {
    "Observation": Observation(
        sensor_id="imu0", modality="imu", values=(0.1, -0.2, 9.81), uncertainty=U
    ),
    "StateUpdate": StateUpdate(
        state_id="s-1",
        physical={"x": 1.0, "v": 0.5},
        belief={"p_contact": 0.2},
        task={"progress": 0.4},
        meta={"staleness_s": 0.01},
        uncertainty=U,
    ),
    "PredictionRequest": PredictionRequest(state_id="s-1", action=(0.3,), horizon_steps=5),
    "PredictionResult": PredictionResult(
        state_id="s-1",
        horizon_steps=5,
        predicted_state={"x": 1.2},
        model_version="wm-0",
        uncertainty=U,
    ),
    "ActionProposal": ActionProposal(
        proposer="system1", policy_version="s1-0", action=(0.3, -0.1), confidence=0.9,
        uncertainty=U,
    ),
    "PlanProposal": PlanProposal(
        planner="system2", skill_ids=("reach", "grasp"), expected_cost=2.5, confidence=0.6,
        uncertainty=U,
    ),
    "AwarenessDecision": AwarenessDecision(
        decision="accept", reasons=("low_surprise",), proposal_message_id="m-1"
    ),
    "SafetyDecision": SafetyDecision(
        approved=True, proposal_message_id="m-1", command=(0.3, -0.1), violations=(),
        safe_state=False,
    ),
    "Outcome": Outcome(success=True, reward=1.0, achieved_state={"x": 1.19}, uncertainty=U),
    "LearningEvent": LearningEvent(
        event_kind="update", component="system1", from_version="s1-0", to_version="s1-1",
        metrics={"heldout_loss": 0.12},
    ),
    "MemoryQuery": MemoryQuery(query_kind="nearest", key=(0.1, 0.2), top_k=3),
    "MemoryResult": MemoryResult(episode_ids=("e1", "e2"), scores=(0.9, 0.7)),
    "ExperimentEvent": ExperimentEvent(
        experiment_id="E1", event_kind="trial_end", condition="s1_only", seed=7,
        metrics={"success": 1.0},
    ),
}


def make_envelope(payload, **overrides):
    meta = dict(
        message_id="m-42",
        timestamp=12.5,
        source="system1",
        destination="awareness",
        cycle_id=3,
        correlation_id="c-3",
    )
    meta.update(overrides)
    return Envelope(payload=payload, **meta)


def valid_dict(name="Observation"):
    return json.loads(make_envelope(SAMPLES[name]).to_json())


def test_every_message_class_has_a_sample():
    assert set(SAMPLES) == set(PAYLOAD_TYPES)
    assert len(PAYLOAD_TYPES) == 13


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_round_trip(name):
    env = make_envelope(SAMPLES[name])
    text = env.to_json()
    back = Envelope.from_json(text)
    assert back == env
    assert type(back.payload) is PAYLOAD_TYPES[name]
    assert back.message_type == name
    assert back.schema_version == SCHEMA_VERSION
    assert back.to_json() == text
    back.validate()


def test_uncertainty_sources_stay_distinct():
    back = Envelope.from_json(make_envelope(SAMPLES["Observation"]).to_json())
    u = back.payload.uncertainty
    assert (u.measurement, u.estimation, u.model, u.policy, u.outcome) == (
        0.01, 0.02, 0.03, None, 0.05,
    )


def test_messages_are_immutable():
    env = make_envelope(SAMPLES["StateUpdate"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.cycle_id = 4
    with pytest.raises(TypeError):
        env.payload.physical["x"] = 2.0


# ----------------------------------------------------------------- malformed messages


@pytest.mark.parametrize("field", ["message_id", "timestamp", "cycle_id", "payload",
                                   "message_type", "schema_version", "correlation_id"])
def test_missing_envelope_field(field):
    data = valid_dict()
    del data[field]
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_missing_payload_field(name):
    data = valid_dict(name)
    del data["payload"][sorted(data["payload"])[0]]
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))


def test_missing_uncertainty_subfield():
    data = valid_dict()
    del data["payload"]["uncertainty"]["model"]
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))


@pytest.mark.parametrize(
    "path, bad",
    [
        (("timestamp",), "12.5"),
        (("cycle_id",), 3.0),
        (("cycle_id",), True),
        (("source",), 5),
        (("payload", "values"), "0.1"),
        (("payload", "values"), [0.1, "x"]),
        (("payload", "sensor_id"), None),
        (("payload", "uncertainty"), 0.1),
        (("payload", "uncertainty", "measurement"), "low"),
        (("payload",), [1, 2]),
    ],
)
def test_wrong_type(path, bad):
    data = valid_dict()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))


def test_wrong_type_in_mapping_and_int_fields():
    data = valid_dict("StateUpdate")
    data["payload"]["physical"]["x"] = "fast"
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))
    data = valid_dict("ExperimentEvent")
    data["payload"]["seed"] = 7.5
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))


def test_wrong_type_on_direct_construction():
    with pytest.raises(ContractError):
        MemoryQuery(query_kind="nearest", key=(0.1,), top_k="3")
    with pytest.raises(ContractError):
        make_envelope({"sensor_id": "imu0"})


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize(
    "path",
    [("timestamp",), ("payload", "values", 0), ("payload", "uncertainty", "estimation")],
)
def test_non_finite_numbers_in_json(path, value):
    data = valid_dict()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = f"__{value}__"
    text = json.dumps(data).replace(f'"__{value}__"', value)
    with pytest.raises(ContractError):
        Envelope.from_json(text)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_on_construction(bad):
    with pytest.raises(ContractError):
        Observation(sensor_id="imu0", modality="imu", values=(bad,), uncertainty=U)
    with pytest.raises(ContractError):
        Uncertainty(model=bad)
    with pytest.raises(ContractError):
        make_envelope(SAMPLES["Outcome"], timestamp=bad)
    with pytest.raises(ContractError):
        StateUpdate(state_id="s", physical={"x": bad}, belief={}, task={}, meta={},
                    uncertainty=U)


@pytest.mark.parametrize("bad", ["Teleport", "", None, 3, "observation"])
def test_unknown_message_type(bad):
    data = valid_dict()
    data["message_type"] = bad
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))


def test_message_type_must_match_payload_shape():
    data = valid_dict("Observation")
    data["message_type"] = "Outcome"
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))


@pytest.mark.parametrize("bad", ["0.9.9", "2.0.0", "", "1.0", 1])
def test_wrong_schema_version(bad):
    data = valid_dict()
    data["schema_version"] = bad
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))
    with pytest.raises(ContractError):
        make_envelope(SAMPLES["Observation"], schema_version=bad)


def test_unknown_fields_rejected():
    data = valid_dict()
    data["extra"] = 1
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))
    data = valid_dict()
    data["payload"]["extra"] = 1
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(data))


@pytest.mark.parametrize("text", ["", "{", "[]", "null", "42"])
def test_not_an_envelope(text):
    with pytest.raises(ContractError):
        Envelope.from_json(text)


def test_validate_catches_tampering_after_construction():
    env = make_envelope(SAMPLES["Observation"])
    object.__setattr__(env.payload, "values", (math.nan,))
    with pytest.raises(ContractError):
        env.validate()


def test_value_invariants():
    with pytest.raises(ContractError):
        Uncertainty(measurement=-0.1)
    with pytest.raises(ContractError):
        dataclasses.replace(SAMPLES["ActionProposal"], confidence=1.5)
    with pytest.raises(ContractError):
        dataclasses.replace(SAMPLES["AwarenessDecision"], decision="ignore")
    with pytest.raises(ContractError):
        dataclasses.replace(SAMPLES["AwarenessDecision"], proposal_message_id=None)
    with pytest.raises(ContractError):
        dataclasses.replace(SAMPLES["SafetyDecision"], safe_state=True)
    with pytest.raises(ContractError):
        MemoryResult(episode_ids=("e1",), scores=())
    with pytest.raises(ContractError):
        make_envelope(SAMPLES["Outcome"], timestamp=-1.0)
    with pytest.raises(ContractError):
        make_envelope(SAMPLES["Outcome"], source="")


# ----------------------------------------------------------------- documentation


def test_docs_cover_every_class_and_field():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "contracts.md").read_text(
        encoding="utf-8"
    )
    assert SCHEMA_VERSION in doc
    classes = [Envelope, Uncertainty, *PAYLOAD_TYPES.values()]
    for cls in classes:
        assert f"### {cls.__name__}" in doc, cls.__name__
        section = doc.split(f"### {cls.__name__}", 1)[1].split("\n### ", 1)[0]
        for f in dataclasses.fields(cls):
            assert f"`{f.name}`" in section, f"{cls.__name__}.{f.name}"
    assert "`message_type`" in doc.split("### Envelope", 1)[1].split("\n### ", 1)[0]
