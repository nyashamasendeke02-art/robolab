# Rules for agents working in this repository

This repository is managed by the Autonomous Research Lab controller.

- Do not run git; the controller commits your changes with provenance.
- `protocols/` holds frozen, pre-registered protocols: never edit.
- `tests/verification/` belongs to the independent verifier: the engineer must not edit it; the verifier may ONLY edit it.
- Entrypoint contract: `<entrypoint> --condition NAME --seed N --out DIR --params JSON` writes `DIR/metrics.json` (flat object of finite numbers), deterministic given the seed.
- Tests run with `python -m pytest -q` from the repo root and are required.
- Never change code, tests or metrics to make a hypothesis look supported.

## AI Robotics Lab rules (docs/MANDATE.md is the mandate)

- Layout: `src/<package>` for contracts, state, world_model, system1, system2, awareness,
  memory, skills, learning, safety, simulation, robot; `tests/`; `experiments/` (entrypoints);
  `configs/`. Import packages by name (`from contracts import ...`); `src` is on sys.path.
- Order of work: contract -> tests -> implementation -> integration. No undocumented
  contracts; schema changes bump the schema version.
- Every actuator command passes the Safety Kernel. Never weaken a safety limit, a test or a
  metric to make something pass.
- Pure Python + NumPy. Every random draw uses an explicitly passed numpy Generator (no
  global RNG). Simulation and evaluation are deterministic given the seed.
- Experiences/telemetry are immutable once written; never silently drop data.
- Do not edit docs/MANDATE.md (controller-owned).
