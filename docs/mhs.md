# Model Hardware Standard (MHS) v0

Implementation: `src/contracts/mhs.py` (import as `from contracts import MHS, ...`).
Mandate refs: Gate 1, REQ-MHS, REQ-SAFE+, H5, ADR-001, ADR-003, ADR-005; task ENG-0012 (G1-5).

Current MHS version: **`0.1.0`** (`contracts.MHS_VERSION`); contracts schema `1.1.0`
(`docs/contracts.md`). Any change to a class or field below must bump `MHS_VERSION` and
update this document.

## Purpose

The brain is the same code for every body. What differs is the MHS: a declared,
machine-readable description of one body. It is the embodiment adapter's contract
(MANDATE: body-specific code stays below an embodiment adapter). It is a
*declaration*, i.e. a datasheet, not ground truth: true or disturbed parameters
(e.g. Puck2D's current mass) are not in it (G1-4).

Consumers:

- **Safety Kernel**: `SafetyConfig.from_mhs(mhs, limit_mode=...)` /
  `SafetyKernel.from_mhs(mhs, telemetry, operator_key, clock, limit_mode=...)`. No
  per-body code. `limit_mode` is kernel policy, not a body property.
- **Kernel kinematics**: `SafetyKernel.observed_kinematics(observation)` reads
  position and velocity from the raw `Observation` through the channels the MHS
  declares (`MHS.channel("position", axis)`, `MHS.channel("velocity", axis)`). The
  cycle runner passes these, not the state estimate, to `check`, `tick`,
  `no_command` and `emergency_stop`, so a faulty or learned estimator cannot change
  what the kernel checks against (safety independence).
- **Brain modules**: the `CycleRunner` takes `mhs=` (default: `environment.mhs()` if
  the environment publishes one) and, at construction, calls `bind_mhs(mhs)` on each
  brain module (state estimator, world model, System 1, System 2, awareness) that
  has one. Brain code reads action and observation layouts only from the MHS
  (`action_layout`, `action_actuators`, `action_size`, `action_index`,
  `observation_layout`, `observation_index`, `channel`). `tests/test_mhs.py`
  statically checks that brain packages contain no Puck2D layout names or constants.

## Rules

Same as the message contracts: frozen dataclasses validated on construction
(`validate()` raises `ContractError`), strict typing (`float` fields need floats, all
finite), closed enumerations, JSON round trip `MHS.from_json(m.to_json()) == m`,
missing / unknown JSON keys rejected. "Unknown" is `None` (JSON `null`).

## Classes

### MHS

| Field | Type | Meaning |
|---|---|---|
| `body_name` | str, non-empty | Body identity. |
| `body_class` | str in `point_mass`, `wheeled`, `legged`, `manipulator`, `aerial` | Kind of body. |
| `actuators` | tuple[Actuator, ...], non-empty, unique names | Actuators. |
| `action_layout` | tuple[str, ...] | Actuator names in the order of the action vector the brain must produce; a permutation of the actuator names. |
| `sensors` | tuple[Sensor, ...], unique names | Sensors. |
| `observation_layout` | tuple[str, ...] | Observation channel order; a permutation of all sensor channels. |
| `body` | Body | Physical properties. |
| `control` | Control | Timing and reflexes. |
| `safety` | SafetyEnvelope | Operating domain and stopping capability. |
| `mhs_version` | str | Must equal `MHS_VERSION`; defaults to it. |

Cross-field checks: every actuator / sensor / workspace frame is in `body.frames`;
`safety.safe_action` has one value per action-layout entry, each inside that
actuator's `[low, high]`; the safety mass bounds contain `body.mass_kg` when it is
known; every workspace axis has exactly one scalar `position` and one scalar
`velocity` sensor on that axis in the workspace frame (the kernel must be able to
observe what it enforces).

### Actuator

| Field | Type | Meaning |
|---|---|---|
| `name` | str, non-empty | Actuator name (used in `action_layout`). |
| `kind` | str in `force`, `torque`, `velocity`, `steering` | What the command sets. |
| `units` | str | `force`: `N`; `torque`: `N*m`; `velocity`: `m/s` or `rad/s`; `steering`: `rad`. Others are rejected. |
| `axis` | Optional[str] | Cartesian axis of `frame` it acts along; `None` if none (e.g. a joint). |
| `frame` | str | Frame of the command. |
| `low` | float | Minimum command (required: no unlimited actuators). |
| `high` | float ≥ `low` | Maximum command. |
| `rate_limit` | Optional[float] > 0 | Max change per second; `None` = not limited. |
| `latency_s` | float ≥ 0 | Command latency (Puck2D: the first-order lag time constant `actuator_tau`). |

### Sensor

| Field | Type | Meaning |
|---|---|---|
| `name` | str, non-empty | Sensor name. |
| `kind` | str in `position`, `velocity`, `acceleration`, `orientation`, `angular_velocity`, `joint_position`, `joint_velocity`, `force`, `torque` | Measured quantity. |
| `units` | str | `m`, `m/s`, `m/s^2`, `rad`, `rad/s`, `rad`/`m`, `rad/s`/`m/s`, `N`, `N*m` respectively. |
| `shape` | tuple[int, ...], entries ≥ 1 | `()` = scalar: one channel named `name`; otherwise `prod(shape)` channels `name[i]` (row-major). |
| `rate_hz` | float > 0 | Sample rate. |
| `noise` | NoiseModel | Noise model. |
| `frame` | str | Frame of the reading. |
| `axis` | Optional[str] | Axis of `frame` a scalar reading is along; `None` if none. |

### NoiseModel

| Field | Type | Meaning |
|---|---|---|
| `kind` | str in `none`, `gaussian`, `unknown` | Noise model. |
| `std` | Optional[float] | Standard deviation (sensor units); required ≥ 0 for `gaussian`, `None` otherwise. |

### Body

| Field | Type | Meaning |
|---|---|---|
| `mass_kg` | Optional[float] > 0 | Nominal mass, `None` = unknown. |
| `inertia_kgm2` | Optional[tuple[float, ...]] | Principal moments (3, ≥ 0) or row-major 3x3 tensor (9); `None` = unknown. |
| `footprint` | Footprint | Geometry. |
| `frames` | tuple[str, ...], non-empty, unique | Frames used by this MHS. |
| `base_frame` | str in `frames` | Body frame. |

### Footprint

| Field | Type | Meaning |
|---|---|---|
| `kind` | str in `point`, `circle`, `box` | Shape in the base frame. |
| `dims` | tuple[float, ...] > 0 | `point`: none; `circle`: radius; `box`: length x, length y (m). |

### Control

| Field | Type | Meaning |
|---|---|---|
| `period_s` | float > 0 | Control period (kernel `control_dt_s`). |
| `max_command_age_s` | float > 0 | Oldest command the body may execute. |
| `watchdog_timeout_s` | float ≥ `period_s` | Max time without an approved command before the safe action. |
| `max_cycle_latency_s` | float in (0, `period_s`] | Required end-to-end cycle latency. |
| `reflexes` | tuple[str, ...], unique | Low-level reflexes the body provides itself below the adapter (Puck2D: `force_saturation`). |

### SafetyEnvelope

| Field | Type | Meaning |
|---|---|---|
| `workspace_frame` | str in `body.frames` | Frame of the workspace. |
| `workspace_axes` | tuple[str, ...], non-empty, unique | Constrained axes. |
| `workspace_low` / `workspace_high` | tuple[float, ...] per axis, low < high | Workspace box (operating domain). |
| `speed_limits` | Optional[tuple[float, ...]] > 0 per axis | m/s; `None` = not declared. Enforced by the kernel (`speed[axis]`). |
| `mass_kg` | float > 0 | **Upper bound** on the true mass (APR-0003). |
| `mass_lower_bound_kg` | Optional[float] in (0, `mass_kg`] | Lower bound; `None` = the mass is exactly `mass_kg`. |
| `braking` | str in `actuators`, `none` | How the body brakes (`actuators`: by actuator command opposing the motion). |
| `brake_decel_mps2` | Optional[tuple[float, ...]] ≥ 0 per axis | Cap on braking deceleration (e.g. traction); `None` = as the actuators allow at `mass_kg`. |
| `safe_action` | tuple[float, ...] | Action for a body at rest when the velocity is unknown (action layout). |
| `estop_latching` | bool | E-stop latches until reset. |
| `estop_reset` | str in `operator` | Who may reset. |
| `estop_action` | str in `brake`, `safe_action`, `power_off` | What an e-stop does. |

## Safety Kernel mapping (kernel v1.2)

`SafetyConfig.from_mhs` maps: `axis_names` = `workspace_axes`; action limits from the
actuators in action-layout order; workspace; `max_command_age_s`,
`watchdog_timeout_s`, `control_dt_s` = `period_s`; `mass_kg` (upper) and
`mass_lower_kg`; `max_speed`; `brake_decel`; `safe_action`; position / velocity
channels via `MHS.channel`; `actuator_latency_s` = each action actuator's `latency_s`. It raises `ContractError` for an MHS outside the kernel's
model: the action layout must be one `force` actuator per workspace axis, in
workspace-axis order, in the workspace frame; `braking == "actuators"`; e-stop
latching, operator reset, `brake`. Not modelled by the kernel (documented, not
rejected): actuator `rate_limit`, sensor noise, impulses.

Actuator latency: a command, and the braking after it, takes effect only `latency_s`
after it is issued; until then the actuators keep an earlier force the kernel does
not know. Per workspace bound the kernel advances the body `ceil(latency_s / dt)`
periods under the action limit pushing towards that bound (zero if none does), then
runs the command-step, workspace and stopping checks from there. A bound crossed in
that time is `stopping[axis]`. This bounds a dead time of at most `latency_s`; for a
first-order lag of time constant `latency_s` (Puck2D) it is an approximation, not a
proof. `latency_s = 0` gives exactly the checks without latency. A large latency
relative to the workspace margin makes the kernel reject nearly every command (safe,
but the body can then only brake).

Mass interval: the command step is predicted at both mass bounds; braking
deceleration uses the upper bound; the braking safe action's final reduced step uses
the lower bound (`-mass_lower * v / dt`) so a lighter body is not reversed, and the
stopping distance includes the resulting geometric residual. With equal bounds the
kernel computes exactly what v1.1 computed (`tests/test_mhs.py` compares a kernel
from Puck2D's MHS with the hand-configured one).

## Puck2D

`Puck2D.mhs(workspace_low, workspace_high, max_command_age_s, watchdog_timeout_s,
mass_bounds)` publishes: `body_name="puck2d"`, `point_mass`; actuators `force_x`,
`force_y` (N, world frame, force limits, latency = `actuator_tau`); scalar sensors
`pos_x`, `pos_y` (position, m), `vel_x`, `vel_y` (velocity, m/s) at `1/dt` Hz with
Gaussian or no noise; `Body.mass_kg = None` (the true mass is ground truth);
footprint circle of `puck_radius` (or point); period `dt`; reflex
`force_saturation`; safe action zero force; mass bounds default to the min / max of
the configured mass and its scheduled changes. The episode harness passes a
fleet-level bound (min / max over the eval set) so the MHS does not reveal one
task's disturbance schedule.
