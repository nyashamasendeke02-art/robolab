# Message contracts

Implementation: `src/contracts/messages.py` (import as `from contracts import ...`).
Mandate refs: Gate 0, REQ-LOG, REQ-STATE, ADR-001 (docs/MANDATE.md, "Contracts").

Current schema version: **`1.1.0`** (`contracts.SCHEMA_VERSION`). Any change to a class
or field below must bump the version and update this document.

Version history:

- `1.0.0` - the 13 payload classes, `Envelope` and `Uncertainty` (Gate 0).
- `1.1.0` - adds the Model Hardware Standard (`contracts.MHS` and its parts `Actuator`,
  `Sensor`, `NoiseModel`, `Body`, `Footprint`, `Control`, `SafetyEnvelope`; G1-5,
  ENG-0012), versioned separately by `contracts.MHS_VERSION` and documented in
  [`docs/mhs.md`](mhs.md). The message classes below are unchanged; envelopes written
  under `1.0.0` are rejected by the version check, as for any schema change.

The MHS is not a message (it is never wrapped in an `Envelope`): it is the declared
description of a body, read once at construction by the Safety Kernel
(`SafetyKernel.from_mhs`) and handed to brain modules by the cycle runner. It uses the
same validation and JSON rules as the messages (see General rules).

## General rules

- Every message is an `Envelope` carrying exactly one payload object of one of the 13
  payload classes. All classes are frozen dataclasses.
- Validation runs on construction (`__post_init__` calls `validate()`), so an invalid
  message cannot be created. `validate()` may also be called explicitly. Every
  violation raises `contracts.ContractError` (a `ValueError` subclass).
- Allowed field types: `str`, `int`, `float`, `bool`, `Uncertainty`, `Optional[...]`
  and `tuple[X, ...]`. `bool` is not accepted where an `int` or `float` is expected, and
  an `int` is not accepted where a `float` is expected (write `1.0`, not `1`, both in
  Python and in JSON; `float` and `int` are never coerced). Every `float` must be finite (NaN and
  ±Infinity are rejected both in objects and in JSON input).
- Named vectors are expressed as parallel tuples (for example `variables` / `values`)
  that must have equal length. Mappings are not used, so messages stay immutable.
- Wire format is JSON. `Envelope.to_json()` / `Envelope.from_json()` round-trip exactly.
  Each payload class also has `to_json()` / `from_json()` for the bare payload. Tuples
  are encoded as arrays, `None` as `null`. Decoding rejects missing fields, unknown
  extra fields, wrong types, non-finite numbers, unknown message types and schema
  version mismatches. `from_json()` accepts only `str`; any other input (`None`,
  numbers, `bytes`, already-parsed dicts/lists) and malformed or excessively nested
  JSON raise `ContractError` rather than leaking `TypeError`/`RecursionError`.
- Free-form string enumerations are closed: values outside the listed set are rejected.

### Envelope

Metadata wrapper (MANDATE: required metadata). The JSON form also contains
`message_type`, which is derived from the payload class name (also available as the
`Envelope.message_type` property) and selects the payload class when decoding.

| Field | Type | Meaning |
|---|---|---|
| `message_id` | str, non-empty | Unique id of this message. |
| `timestamp` | float, finite, ≥ 0 | Simulation-clock time in seconds when the message was created. |
| `source` | str, non-empty | Sending component (e.g. `system1`). |
| `destination` | str, non-empty | Receiving component (e.g. `awareness`). |
| `cycle_id` | int, ≥ 0 | Control cycle this message belongs to. |
| `correlation_id` | str, non-empty | Links related messages (e.g. a request and its result). |
| `payload` | one payload class | The message body. |
| `schema_version` | str | Must equal `SCHEMA_VERSION`; defaults to it. |

### Uncertainty

Uncertainty components are kept as separate typed fields so that they stay
distinguishable. Each is a non-negative, finite standard deviation in the units of the
quantity it qualifies, or `null`/`None` when not applicable or not estimated. All five
keys must be present in JSON.

| Field | Type | Meaning |
|---|---|---|
| `measurement` | Optional[float] ≥ 0 | Sensor/measurement noise. |
| `estimation` | Optional[float] ≥ 0 | State-estimation (belief) uncertainty. |
| `model` | Optional[float] ≥ 0 | Predictive-model (World Model) uncertainty. |
| `policy` | Optional[float] ≥ 0 | Uncertainty of a policy's action choice. |
| `outcome` | Optional[float] ≥ 0 | Uncertainty about the result of acting. |

## Payload classes

### Observation

A raw sensor reading.

| Field | Type | Meaning |
|---|---|---|
| `sensor_id` | str, non-empty | Sensor identifier. |
| `channels` | tuple[str, ...] | Channel names. |
| `values` | tuple[float, ...] | Readings; same length as `channels`. |
| `uncertainty` | Uncertainty | Typically `measurement`. |

### StateUpdate

An update of one state layer (REQ-STATE).

| Field | Type | Meaning |
|---|---|---|
| `layer` | str in `physical`, `belief`, `task`, `meta` | State layer updated. |
| `variables` | tuple[str, ...] | State variable names. |
| `values` | tuple[float, ...] | Values; same length as `variables`. |
| `uncertainty` | Uncertainty | Typically `measurement` / `estimation`. |
| `source_message_ids` | tuple[str, ...] | Ids of the messages this update was derived from (provenance). |

### PredictionRequest

Request to the World Model to roll a state forward under an action.

| Field | Type | Meaning |
|---|---|---|
| `variables` | tuple[str, ...] | State variable names. |
| `state` | tuple[float, ...] | Initial state; same length as `variables`. |
| `action` | tuple[float, ...] | Action applied over the horizon. |
| `horizon_steps` | int ≥ 1 | Number of steps to predict. |
| `dt` | float > 0 | Step length in seconds. |

### PredictionResult

World Model prediction answering a `PredictionRequest` (linked by `correlation_id`).

| Field | Type | Meaning |
|---|---|---|
| `model_version` | str, non-empty | Version of the predicting model. |
| `variables` | tuple[str, ...] | State variable names. |
| `predicted` | tuple[float, ...] | Predicted state at the horizon; same length as `variables`. |
| `horizon_steps` | int ≥ 1 | Horizon that was predicted. |
| `uncertainty` | Uncertainty | Typically `model` (and propagated `estimation`). |

### ActionProposal

A low-level action proposed by System 1.

| Field | Type | Meaning |
|---|---|---|
| `proposal_id` | str, non-empty | Id referenced by awareness/safety decisions. |
| `policy_version` | str, non-empty | Version of the proposing policy. |
| `action` | tuple[float, ...] | Proposed actuator command vector. |
| `confidence` | float in [0, 1] | Policy confidence. |
| `uncertainty` | Uncertainty | Typically `policy`. |

### PlanProposal

A plan / skill sequence proposed by System 2 (System 2 never commands actuators).

| Field | Type | Meaning |
|---|---|---|
| `proposal_id` | str, non-empty | Id referenced by awareness decisions. |
| `planner_version` | str, non-empty | Version of the planner. |
| `goal` | str | Goal the plan addresses. |
| `skill_ids` | tuple[str, ...] | Ordered skills to execute. |
| `expected_cost` | float | Planner's expected cost of the plan. |
| `uncertainty` | Uncertainty | Typically `outcome`. |

### AwarenessDecision

Arbitration result of the Awareness Harness.

| Field | Type | Meaning |
|---|---|---|
| `decision` | str in `accept`, `request_s2`, `request_prediction`, `replan`, `abstain`, `escalate` | Arbitration outcome. |
| `reason` | str | Human-readable rationale (interpretable rules). |
| `proposal_ids` | tuple[str, ...] | Proposals that were considered. |
| `selected_proposal_id` | Optional[str] | Selected proposal; required when `decision` is `accept`. |

### SafetyDecision

Verdict of the Safety Kernel (final authority).

| Field | Type | Meaning |
|---|---|---|
| `verdict` | str in `approve`, `reject`, `emergency_stop` | Kernel verdict. A rejection is final for the cycle. |
| `proposal_id` | str, non-empty | Proposal that was checked. |
| `approved_action` | tuple[float, ...] | Action allowed to reach actuators; must be empty unless `verdict` is `approve`. |
| `violated_constraints` | tuple[str, ...] | Names of violated constraints. |
| `kernel_version` | str, non-empty | Safety Kernel version. |

### Outcome

Observed result of executing an action.

| Field | Type | Meaning |
|---|---|---|
| `action_id` | str, non-empty | Proposal/action that was executed. |
| `success` | bool | Whether the action achieved its intent. |
| `reward` | float | Scalar reward/score. |
| `variables` | tuple[str, ...] | Observed variable names. |
| `observed` | tuple[float, ...] | Observed values; same length as `variables`. |
| `uncertainty` | Uncertainty | Typically `measurement` / `outcome`. |

### LearningEvent

A step in the continual-learning pipeline (experience → selection → replay → update →
validation → deployment, plus rollback).

| Field | Type | Meaning |
|---|---|---|
| `kind` | str in `experience`, `selection`, `replay`, `update`, `validation`, `deployment`, `rollback` | Pipeline stage. |
| `module` | str, non-empty | Learned module affected. |
| `version_before` | str | Module version before the event. |
| `version_after` | str | Module version after the event. |
| `metric_names` | tuple[str, ...] | Metric names. |
| `metric_values` | tuple[float, ...] | Metric values; same length as `metric_names`. |

### MemoryQuery

Query to episodic memory or the skill library.

| Field | Type | Meaning |
|---|---|---|
| `kind` | str in `episodic`, `skill` | Store to query. |
| `key_names` | tuple[str, ...] | Query key names. |
| `key_values` | tuple[float, ...] | Query key values; same length as `key_names`. |
| `max_results` | int ≥ 1 | Maximum number of records to return. |

### MemoryResult

Answer to a `MemoryQuery`.

| Field | Type | Meaning |
|---|---|---|
| `query_message_id` | str, non-empty | `message_id` of the answered query. |
| `record_ids` | tuple[str, ...] | Retrieved record ids. |
| `scores` | tuple[float, ...] | Retrieval scores; same length as `record_ids`. |
| `provenance` | tuple[str, ...] | Origin of each record; same length as `record_ids`. |

### ExperimentEvent

Experiment record for telemetry (REQ-LOG).

| Field | Type | Meaning |
|---|---|---|
| `experiment_id` | str, non-empty | Experiment identifier. |
| `kind` | str in `start`, `step`, `end`, `error` | Event kind. |
| `condition` | str | Experimental condition name. |
| `seed` | int | Seed of the run. |
| `metric_names` | tuple[str, ...] | Metric names. |
| `metric_values` | tuple[float, ...] | Metric values; same length as `metric_names`. |
