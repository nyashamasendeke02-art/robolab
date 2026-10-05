# Message contracts

Implementation: `src/contracts/messages.py` (import as `from contracts import ...`).
Tests: `tests/test_contracts.py`. Traces to: Gate 0, REQ-LOG, REQ-STATE, ADR-001
(docs/MANDATE.md, "Contracts").

**Schema version:** `SCHEMA_VERSION = "1.0.0"`. Any change to a class or field below
bumps this version and updates this document (no undocumented contracts).

## Rules common to all contracts

- Every contract is a frozen dataclass. Construction normalises values (JSON arrays and
  Python lists become tuples, objects become read-only mappings, integers in `float`
  fields become floats) and then runs `validate()`, so an invalid message cannot be
  built.
- `validate()` raises `ContractError` (a `ValueError` subclass) for: a missing field, an
  unknown (extra) field, a wrong type, a `null` in a non-optional field, a non-finite
  number (`NaN`, `Infinity`, `-Infinity`), an unknown `message_type`, a
  `schema_version` other than `SCHEMA_VERSION`, and the per-class invariants listed
  below.
- Types are strict: `bool` is not accepted as `int`/`float`, a float is not accepted as
  `int`, strings are never parsed as numbers.
- Notation: `float` is a finite float; `tuple[T]` is a JSON array of `T`;
  `map[str, float]` is a JSON object with string keys and finite float values;
  `T | null` is optional (`null` allowed but the key is still required).
- Wire format: `Envelope.to_json()` emits JSON with sorted keys and refuses non-finite
  numbers; `Envelope.from_json(text)` parses, checks the envelope, then the version,
  then the message type, then the payload.

## Envelope

### Envelope

Required metadata around exactly one payload.

| Field | Type | Meaning / constraint |
|---|---|---|
| `message_id` | str | Unique id of this message; non-empty. |
| `timestamp` | float | Simulation-clock time in seconds; finite, >= 0. |
| `source` | str | Sending component (e.g. `system1`); non-empty. |
| `destination` | str | Receiving component; non-empty. |
| `cycle_id` | int | Control-cycle index; >= 0. |
| `correlation_id` | str | Links messages belonging to one request/decision chain; non-empty. |
| `payload` | Payload | One of the 13 payload classes below. |
| `schema_version` | str | Must equal `SCHEMA_VERSION`; defaults to it on construction. |
| `message_type` | str | Wire-only discriminator (the payload class name); derived from `payload`, exposed as the `message_type` property. Unknown values are rejected. |

## Uncertainty

### Uncertainty

Uncertainty sources stay separate and typed (they are never summed into one number).
Each value is a non-negative spread (e.g. a standard deviation) in the units of the
quantity described by the carrying message; `null` means "not estimated". All five keys
are required on the wire.

| Field | Type | Meaning / constraint |
|---|---|---|
| `measurement` | float \| null | Sensor noise. >= 0. |
| `estimation` | float \| null | State-estimation / belief uncertainty. >= 0. |
| `model` | float \| null | World-model / predictive-model uncertainty. >= 0. |
| `policy` | float \| null | Uncertainty of the policy's chosen action. >= 0. |
| `outcome` | float \| null | Uncertainty about the result of acting. >= 0. |

## Payloads

`message_type` equals the class name.

### Observation

A sensor reading.

| Field | Type | Meaning / constraint |
|---|---|---|
| `sensor_id` | str | Sensor identifier; non-empty. |
| `modality` | str | Sensor modality (e.g. `imu`, `joint_encoder`); non-empty. |
| `values` | tuple[float] | Raw reading vector. |
| `uncertainty` | Uncertainty | Typically `measurement`. |

### StateUpdate

State/belief update covering the four state layers (REQ-STATE).

| Field | Type | Meaning / constraint |
|---|---|---|
| `state_id` | str | Identifier of this state estimate; non-empty. |
| `physical` | map[str, float] | Physical layer (positions, velocities, ...). |
| `belief` | map[str, float] | Belief layer (probabilities, latent estimates). |
| `task` | map[str, float] | Task layer (goal progress, sub-goal indicators). |
| `meta` | map[str, float] | Meta layer (staleness, health, budget). |
| `uncertainty` | Uncertainty | Typically `measurement` and `estimation`. |

### PredictionRequest

Request to the World Model to roll a state forward under an action.

| Field | Type | Meaning / constraint |
|---|---|---|
| `state_id` | str | State to predict from; non-empty. |
| `action` | tuple[float] | Action vector to apply. |
| `horizon_steps` | int | Number of steps to predict; >= 1. |

### PredictionResult

World Model answer to a PredictionRequest (match via `correlation_id`).

| Field | Type | Meaning / constraint |
|---|---|---|
| `state_id` | str | State predicted from; non-empty. |
| `horizon_steps` | int | Steps predicted; >= 1. |
| `predicted_state` | map[str, float] | Predicted state variables. |
| `model_version` | str | World Model version; non-empty. |
| `uncertainty` | Uncertainty | Typically `model`. |

### ActionProposal

A candidate action (normally from System 1). Not a command: it must pass Awareness and
the Safety Kernel.

| Field | Type | Meaning / constraint |
|---|---|---|
| `proposer` | str | Proposing component; non-empty. |
| `policy_version` | str | Versioned policy id; non-empty. |
| `action` | tuple[float] | Proposed action vector. |
| `confidence` | float | Proposer confidence in [0, 1]. |
| `uncertainty` | Uncertainty | Typically `policy`. |

### PlanProposal

A plan / skill sequence from System 2. System 2 never commands actuators.

| Field | Type | Meaning / constraint |
|---|---|---|
| `planner` | str | Planning component; non-empty. |
| `skill_ids` | tuple[str] | Ordered skill identifiers. |
| `expected_cost` | float | Planner's expected cost of the plan. |
| `confidence` | float | Planner confidence in [0, 1]. |
| `uncertainty` | Uncertainty | Typically `policy` and `outcome`. |

### AwarenessDecision

Arbitration result of the Awareness Harness.

| Field | Type | Meaning / constraint |
|---|---|---|
| `decision` | str | One of `accept`, `request_s2`, `request_prediction`, `replan`, `abstain`, `escalate`. |
| `reasons` | tuple[str] | Interpretable reasons (rule ids) for the decision. |
| `proposal_message_id` | str \| null | `message_id` of the accepted proposal; required when `decision` is `accept`. |

### SafetyDecision

Final verdict of the Safety Kernel for a proposal in this cycle.

| Field | Type | Meaning / constraint |
|---|---|---|
| `approved` | bool | Whether `command` may be sent to actuators. |
| `proposal_message_id` | str | `message_id` of the reviewed proposal; non-empty. |
| `command` | tuple[float] | Command actually released (possibly clamped); ignored when not approved. |
| `violations` | tuple[str] | Limit/watchdog ids that were violated or triggered clamping. |
| `safe_state` | bool | Whether the configured safe state is entered; cannot be true when `approved` is true. |

### Outcome

Result observed after acting.

| Field | Type | Meaning / constraint |
|---|---|---|
| `success` | bool | Task-level success for this step/episode. |
| `reward` | float | Scalar reward/score. |
| `achieved_state` | map[str, float] | State variables actually reached. |
| `uncertainty` | Uncertainty | Typically `outcome` and `measurement`. |

### LearningEvent

A record of a learning step (update, validation, deployment, rollback, ...).

| Field | Type | Meaning / constraint |
|---|---|---|
| `event_kind` | str | Kind of event (e.g. `update`, `rollback`); non-empty. |
| `component` | str | Learning component; non-empty. |
| `from_version` | str | Version before the event. |
| `to_version` | str | Version after the event. |
| `metrics` | map[str, float] | Associated metrics (e.g. held-out loss, forgetting). |

### MemoryQuery

Query to episodic memory.

| Field | Type | Meaning / constraint |
|---|---|---|
| `query_kind` | str | Retrieval mode (e.g. `nearest`); non-empty. |
| `key` | tuple[float] | Query key vector. |
| `top_k` | int | Maximum results; >= 1. |

### MemoryResult

Answer to a MemoryQuery (match via `correlation_id`).

| Field | Type | Meaning / constraint |
|---|---|---|
| `episode_ids` | tuple[str] | Retrieved episode ids (provenance). |
| `scores` | tuple[float] | Score per episode; same length as `episode_ids`. |

### ExperimentEvent

Experiment telemetry record (REQ-LOG).

| Field | Type | Meaning / constraint |
|---|---|---|
| `experiment_id` | str | Experiment id (e.g. `E1`); non-empty. |
| `event_kind` | str | Kind of event (e.g. `trial_start`, `trial_end`); non-empty. |
| `condition` | str | Experimental condition name. |
| `seed` | int | Seed of the run. |
| `metrics` | map[str, float] | Metrics recorded with the event. |
