"""Contract tests for src/contracts (Gate 0, ENG-0002)."""

import dataclasses
import json
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

U = Uncertainty(measurement=0.01, estimation=0.02, model=0.03, policy=None, outcome=0.5)

SAMPLES = {
    "Observation": Observation(
        sensor_id="joint_encoders", channels=("q0", "q1"), values=(0.1, -0.2),
        uncertainty=Uncertainty(measurement=0.001),
    ),
    "StateUpdate": StateUpdate(
        layer="belief", variables=("x", "v"), values=(1.0, 0.0), uncertainty=U,
        source_message_ids=("m-1", "m-2"),
    ),
    "PredictionRequest": PredictionRequest(
        variables=("x", "v"), state=(1.0, 0.0), action=(0.5,), horizon_steps=10, dt=0.01,
    ),
    "PredictionResult": PredictionResult(
        model_version="wm-0.1", variables=("x", "v"), predicted=(1.05, 0.1),
        horizon_steps=10, uncertainty=Uncertainty(model=0.02, estimation=0.01),
    ),
    "ActionProposal": ActionProposal(
        proposal_id="p-1", policy_version="s1-0.1", action=(0.5, -0.5), confidence=0.8,
        uncertainty=Uncertainty(policy=0.1),
    ),
    "PlanProposal": PlanProposal(
        proposal_id="p-2", planner_version="s2-0.1", goal="reach(target)",
        skill_ids=("reach", "grasp"), expected_cost=3.5, uncertainty=Uncertainty(outcome=0.4),
    ),
    "AwarenessDecision": AwarenessDecision(
        decision="accept", reason="confidence above threshold",
        proposal_ids=("p-1", "p-2"), selected_proposal_id="p-1",
    ),
    "SafetyDecision": SafetyDecision(
        verdict="approve", proposal_id="p-1", approved_action=(0.5, -0.5),
        violated_constraints=(), kernel_version="sk-0.1",
    ),
    "Outcome": Outcome(
        action_id="p-1", success=True, reward=1.0, variables=("x",), observed=(1.04,),
        uncertainty=Uncertainty(measurement=0.001, outcome=0.05),
    ),
    "LearningEvent": LearningEvent(
        kind="update", module="world_model", version_before="wm-0.1", version_after="wm-0.2",
        metric_names=("heldout_mse",), metric_values=(0.012,),
    ),
    "MemoryQuery": MemoryQuery(
        kind="episodic", key_names=("x",), key_values=(1.0,), max_results=5,
    ),
    "MemoryResult": MemoryResult(
        query_message_id="m-9", record_ids=("ep-1", "ep-2"), scores=(0.9, 0.7),
        provenance=("run-1/cycle-3", "run-1/cycle-8"),
    ),
    "ExperimentEvent": ExperimentEvent(
        experiment_id="EXP-1", kind="end", condition="baseline", seed=7,
        metric_names=("success_rate",), metric_values=(0.75,),
    ),
}


def make_envelope(payload, **overrides):
    kwargs = dict(
        message_id="m-1", timestamp=12.5, source="system1", destination="awareness",
        cycle_id=3, correlation_id="c-1", payload=payload,
    )
    kwargs.update(overrides)
    return Envelope(**kwargs)


def envelope_dict(name="ActionProposal"):
    return json.loads(make_envelope(SAMPLES[name]).to_json())


def test_samples_cover_every_message_class():
    assert set(SAMPLES) == set(PAYLOAD_TYPES)
    assert len(PAYLOAD_TYPES) == 13


# --- round trips -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_payload_round_trip(name):
    payload = SAMPLES[name]
    assert type(payload).from_json(payload.to_json()) == payload


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_envelope_round_trip(name):
    env = make_envelope(SAMPLES[name])
    text = env.to_json()
    back = Envelope.from_json(text)
    assert back == env
    assert back.message_type == name
    assert back.schema_version == SCHEMA_VERSION
    assert back.to_json() == text


def test_uncertainty_components_stay_distinct_after_round_trip():
    env = Envelope.from_json(make_envelope(SAMPLES["StateUpdate"]).to_json())
    u = env.payload.uncertainty
    assert (u.measurement, u.estimation, u.model, u.policy, u.outcome) == (
        0.01, 0.02, 0.03, None, 0.5,
    )


def test_messages_are_frozen():
    env = make_envelope(SAMPLES["Observation"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.cycle_id = 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.payload.sensor_id = "other"


# --- malformed messages ----------------------------------------------------


def test_missing_envelope_field_raises():
    d = envelope_dict()
    del d["correlation_id"]
    with pytest.raises(ContractError, match="missing"):
        Envelope.from_dict(d)


def test_missing_payload_field_raises():
    d = envelope_dict()
    del d["payload"]["confidence"]
    with pytest.raises(ContractError, match="missing"):
        Envelope.from_json(json.dumps(d))


def test_missing_uncertainty_component_raises():
    d = envelope_dict()
    del d["payload"]["uncertainty"]["policy"]
    with pytest.raises(ContractError, match="missing"):
        Envelope.from_json(json.dumps(d))


def test_unknown_field_raises():
    d = envelope_dict()
    d["payload"]["extra"] = 1
    with pytest.raises(ContractError, match="unknown field"):
        Envelope.from_json(json.dumps(d))


@pytest.mark.parametrize(
    "path, value",
    [
        (("timestamp",), "12.5"),
        (("cycle_id",), 3.0),
        (("cycle_id",), True),
        (("source",), 5),
        (("payload", "confidence"), "high"),
        (("payload", "confidence"), True),
        (("payload", "action"), 0.5),
        (("payload", "action"), [0.5, "x"]),
        (("payload", "proposal_id"), None),
        (("payload", "uncertainty"), 0.1),
        (("payload", "uncertainty", "policy"), "0.1"),
    ],
)
def test_wrong_type_raises(path, value):
    d = envelope_dict()
    target = d
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(d))


def test_wrong_type_on_construction_raises():
    with pytest.raises(ContractError, match="expected int"):
        MemoryQuery(kind="skill", key_names=(), key_values=(), max_results=2.0)
    with pytest.raises(ContractError, match="expected tuple"):
        Observation(sensor_id="s", channels=["a"], values=(1.0,), uncertainty=Uncertainty())


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_in_json_raises(token):
    text = make_envelope(SAMPLES["ActionProposal"]).to_json()
    text = text.replace('"confidence": 0.8', f'"confidence": {token}')
    assert token in text
    with pytest.raises(ContractError, match="non-finite"):
        Envelope.from_json(text)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_on_construction_raises(bad):
    with pytest.raises(ContractError, match="non-finite"):
        Uncertainty(model=bad)
    with pytest.raises(ContractError, match="non-finite"):
        make_envelope(SAMPLES["Observation"], timestamp=bad)
    with pytest.raises(ContractError, match="non-finite"):
        Observation(sensor_id="s", channels=("a",), values=(bad,), uncertainty=Uncertainty())


def test_unknown_message_type_raises():
    d = envelope_dict()
    d["message_type"] = "Teleport"
    with pytest.raises(ContractError, match="unknown message type"):
        Envelope.from_json(json.dumps(d))


def test_unregistered_payload_class_raises():
    @dataclasses.dataclass(frozen=True)
    class Teleport:
        where: str

    with pytest.raises(ContractError, match="unknown message type"):
        make_envelope(Teleport(where="moon"))


def test_payload_type_mismatch_raises():
    d = envelope_dict("ActionProposal")
    d["message_type"] = "Observation"
    with pytest.raises(ContractError):
        Envelope.from_json(json.dumps(d))


@pytest.mark.parametrize("version", ["0.9.0", "2.0.0", ""])
def test_schema_version_mismatch_raises(version):
    d = envelope_dict()
    d["schema_version"] = version
    with pytest.raises(ContractError, match="schema_version"):
        Envelope.from_json(json.dumps(d))
    with pytest.raises(ContractError, match="schema_version"):
        make_envelope(SAMPLES["ActionProposal"], schema_version=version)


def test_invalid_json_raises():
    with pytest.raises(ContractError):
        Envelope.from_json("{not json")
    with pytest.raises(ContractError):
        Envelope.from_json("[]")


# --- semantic constraints --------------------------------------------------


def test_semantic_constraints():
    with pytest.raises(ContractError):
        Uncertainty(measurement=-0.1)
    with pytest.raises(ContractError):
        StateUpdate(layer="dream", variables=(), values=(), uncertainty=U, source_message_ids=())
    with pytest.raises(ContractError):
        Observation(sensor_id="s", channels=("a", "b"), values=(1.0,), uncertainty=U)
    with pytest.raises(ContractError):
        ActionProposal(proposal_id="p", policy_version="v", action=(), confidence=1.5,
                       uncertainty=U)
    with pytest.raises(ContractError):
        AwarenessDecision(decision="accept", reason="", proposal_ids=(), selected_proposal_id=None)
    with pytest.raises(ContractError):
        SafetyDecision(verdict="reject", proposal_id="p", approved_action=(0.1,),
                       violated_constraints=("joint_limit",), kernel_version="sk")
    with pytest.raises(ContractError):
        make_envelope(SAMPLES["Observation"], timestamp=-1.0)


# --- documentation ---------------------------------------------------------


def test_docs_document_every_class_and_field():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "contracts.md").read_text("utf-8")
    assert SCHEMA_VERSION in doc
    classes = [Envelope, Uncertainty, *PAYLOAD_TYPES.values()]
    for cls in classes:
        assert f"### {cls.__name__}" in doc, cls.__name__
        section = doc.split(f"### {cls.__name__}", 1)[1].split("\n### ", 1)[0]
        for f in dataclasses.fields(cls):
            assert f"`{f.name}`" in section, f"{cls.__name__}.{f.name}"
    assert "`message_type`" in doc
