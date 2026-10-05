# Lab Mandate: AI Robotics Lab

**Status:** adopted 2026-10-04 by the human researcher as the foundation of this lab
(decision D20): the research mandate and the engineering mandate.
**Source of truth:** [`AI_Robotics_Full_Documentation.pdf`](../AI_Robotics_Full_Documentation.pdf)
(9 pages). This file is a structured, traceable digest of it. If the two disagree,
the PDF wins and this file is corrected.

Evidence labels (project taxonomy): every architectural element below is an
**ENGINEERING_DECISION** or a **HYPOTHESIS** of the mandate. None of it is
ESTABLISHED; the PDF itself says "the architecture is a research hypothesis, not an
assumed result".

---

## 1. Research mandate

**Thesis.** Resource-efficient embodied intelligence: a modular robot brain built from
a physics-informed World Model, a fast System 1, a slower System 2, an Awareness
Harness, memory/skills and continual learning. Long-term direction: an
embodiment-agnostic robot brain.

**Central question.** Can modular embodied intelligence learn from experience and
allocate computation adaptively under resource constraints?

**Research questions.** Which frontier abstractions transfer into constrained robotics?
How does continual learning compare with train-then-deploy? What is the minimum
useful compute? Can S1 and S2 cooperate efficiently? Can prediction error identify
untrustworthy situations? Can skills transfer? What representation supports shared
intelligence without losing grounding?

| ID | Hypothesis |
|---|---|
| H1 | S1+S2 broadens task capability at acceptable compute. |
| H2 | Awareness reduces unnecessary S2 use while improving difficult cases. |
| H3 | Prediction error helps detect states requiring reconsideration. |
| H4 | Continual learning improves future performance while controlling forgetting. |
| H5 | A shared representation plus embodiment adapters enables measurable cross-embodiment transfer. |

| ID | Experiment | Tests |
|---|---|---|
| E1 | S1 baseline | reference for everything else |
| E2 | S1+S2 | H1 |
| E3 | awareness-guided escalation | H2 (and H3) |
| E4 | World Model utility | H3, ADR-004 |
| E5 | continual learning | H4 |
| E6 | embodiment transfer | H5 |

**Baselines:** task-specific/conventional where applicable, S1 only, S1+S2,
S1+World Model, S1+S2+Awareness, full system.
**Ablations:** Awareness, World Model, S2, Memory, continual learning, and the
novelty, surprise, stakes and budget inputs.
**Metrics:** performance (success, time, failure, recovery); decisions (S2
invocations, correct/unnecessary escalation, abstention); prediction (error,
calibration); resources (latency, memory, compute, energy where measurable);
learning (forgetting, transfer). Every result reports its baseline, sample count,
variability, configuration and a failure analysis.
**Runbook:** define hypothesis → select baseline → freeze versions → set seed →
validate environment → pilot → run trials → collect telemetry → analyse → failure
analysis → archive → conclude. Evaluation definitions are never changed after
results without recording the change.

**Out of scope (initial system):** claims of consciousness, unrestricted
self-modification, learned bypass of safety, premature universal embeddings.
"Awareness" means self-monitoring.

## 2. Engineering mandate

**Control cycle.** Sensors → State/Belief → World Model → Prediction; S1 → candidate
action; S2 → plan/skill; Awareness arbitrates (accept, request S2, request
prediction, replan, abstain, escalate) → **Safety Kernel (final authority)** →
Actuators → Environment → Outcome. S2 never commands actuators. A safety rejection
is final for that cycle. Timeouts enter a configured safe state. Learned modules can
never disable hard safety constraints.

| ID | Requirement |
|---|---|
| REQ-STATE | state and belief representation |
| REQ-WM | prediction and uncertainty |
| REQ-S1 | low-latency control and confidence |
| REQ-S2 | planning and skill selection |
| REQ-AH | monitoring and arbitration |
| REQ-SAFE | limits, watchdogs, emergency handling |
| REQ-MEM | episodic memory and skills |
| REQ-CL | online learning and forgetting control |
| REQ-SIM | deterministic simulation |
| REQ-ROS | robot integration |
| REQ-LOG | telemetry and experiment records |

| ID | Architecture decision |
|---|---|
| ADR-001 | modular architecture, so components can be ablated and replaced |
| ADR-002 | fast/slow control for efficient deliberation |
| ADR-003 | deterministic Safety Kernel outside learned authority |
| ADR-004 | World Model as a first-class predictive subsystem |
| ADR-005 | shared representation is experimental, not assumed universal |

**Acceptance gates.** Gate 0 repository/contracts · 1 simulation · 2 World Model ·
3 System 1 · 4 Awareness · 5 System 2 · 6 learning · 7 hardware · 8 research
evaluation. Each gate needs implementation, tests, observability, documentation and
reproducibility evidence.

**Contracts.** Messages: Observation, StateUpdate, PredictionRequest/Result,
ActionProposal, PlanProposal, AwarenessDecision, SafetyDecision, Outcome,
LearningEvent, MemoryQuery/Result, ExperimentEvent. Required metadata: message ID,
timestamp, source, destination, schema version, cycle ID, correlation ID, payload.
State layers: physical, belief, task, meta. Measurement, estimation, model, policy
and outcome uncertainty stay distinguishable. Schemas are versioned; no
undocumented contracts.

**Component specs.** World Model: object-centric latent physical variables,
hand-written integrator, learned residuals for friction/contact, compared
progressively with more learned physics. System 1: explicit action space,
confidence and calibration, bounded latency, versioned policy, safe fallback.
Awareness v1: interpretable rules (learned arbitration is a research question).
Memory: episodic retrieval and replay with observable provenance; skills are
versioned and every change is evaluated. Continual learning: experience → selection
→ replay → update → validation → deployment, with held-out regression, replay,
versioning, rollback and safety gates.

**Robotics and safety.** Simulation first (MuJoCo/MJX or equivalent): deterministic
seeds, configurable physics, sensor/actuator models, disturbances, domain
randomisation, reproducible logs. Body-specific code stays below an embodiment
adapter, and core logic stays testable without ROS2. Every actuator command passes
the Safety Kernel. Watchdogs, an emergency stop that is independent of learned
decision quality, and fault injection (sensor dropout, stale state, malformed
messages, actuator disconnect, component failure, timeouts), each with an expected
safe behaviour and evidence.

**Engineering practice.** Repository layout: `src/{contracts, state, world_model,
system1, system2, awareness, memory, skills, learning, safety, simulation, robot}`,
`tests`, `experiments`, `configs`, `docs`, `scripts`, `docker`, `.claude`. Order of
work: contract → tests → implementation → integration → ADR if needed → report.
Test levels: unit, contract, component, integration, simulation, fault injection,
regression, research benchmark. Pinned dependencies and reproducible environments.
Experiences are immutable after collection, and low-quality data is never silently
discarded. Deployment stages: simulation → HIL → bench → constrained motion →
supervised low-risk → expanded, each with acceptance evidence and rollback.
Roadmap: Phase 1 foundation → 2 online continual learning → 3 multi-task/shared
models → 4 scaling.

**Agents.** 13 roles (Project Lead, Research Governor, Contracts, Simulation, World
Model, System 1, Awareness, System 2, Memory & Skills, Safety, Continual Learning,
Testing & Evaluation, Integration & DevOps). Agents implement the architecture and
never silently redesign it. Forbidden: undocumented contract drift, skipped tests,
unsupported completion claims, changing safety limits to pass tests, metric
manipulation. Research changes need Research Governor review, safety changes need
Safety review, contract changes need Contracts review.

## 3. Reconciliation with CLAUDE.md (D20)

The mandate sets the *direction*; CLAUDE.md's research invariants set the *method*.
Both apply.

1. **Components are earned, not assumed.** Every module (World Model, S2, Awareness,
   memory, continual learning) enters the system as an *intervention* in an
   experiment against the previous baseline, with the mandate's ablations. A module
   that does not beat its ablation is recorded as a negative result, not kept
   because it "sounds intelligent".
2. **Sequencing follows the gates and the first milestone.** Gate 0 (contracts) and
   Gate 1 (simulation), then E1, an S1 baseline in a small deterministic
   simulation, as a reproducible baseline. This *is* CLAUDE.md's first milestone (a
   minimal predictive agent with a reproducible baseline). Nothing above S1 is built
   before E1 exists.
3. **Emergence claims.** CLAUDE.md asks whether general capability can emerge. Under
   this mandate, a capability may be called emergent only if it was not hard-coded
   in any component and alternative explanations were examined (CLAUDE.md
   invariant 9).
4. **Safety Kernel is not a research variable.** It is never ablated in experiments
   that actuate; it is tested by fault injection.
5. **Agent roles.** The 13 mandate roles are *responsibilities*. In autolab they map
   onto the scientist (Research Governor, research design), the engineer
   (component roles, Contracts, Simulation, DevOps), the verifier (Testing &
   Evaluation, adversarial review) and the human (Project Lead authority). Safety
   review of Safety Kernel changes requires the human.
6. **Compute.** The simulation starts in pure Python/NumPy and is sized for this
   machine (8 GB RAM). MuJoCo comes in when a gate requires contact physics,
   recorded as a decision.

## 4. Open questions raised by the mandate

See project_state/OPEN_QUESTIONS.md (items 8-12).
