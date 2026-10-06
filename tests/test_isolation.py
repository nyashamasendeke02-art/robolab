"""G1-4 ground-truth isolation (Gate 1, REQ-ISO, REQ-SIM, ADR-001, ADR-004; ENG-0011).

1. A spy fills every brain slot (StateEstimator, WorldModel, System1, System2,
   Awareness) over seeded Puck2D episodes with every disturbance kind and
   records everything it is given; no ground-truth field, object or value may
   reach it, while telemetry does carry ground truth under ``ground_truth``.
2. A static test: no brain package imports ``simulation`` (directly or through
   another ``src`` module).
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pytest

import contracts
from contracts import ActionProposal, AwarenessDecision, Uncertainty
from robot.runner import C_ACTUATE, PassThroughAwareness, PassThroughStateEstimator
from simulation.harness import EvalSet, EvalSetSpec, TaskBrief, TaskInstance, build_eval_set, run_episodes
from simulation.policies import PDController
from state.telemetry import TelemetryLog, read_records

SRC = Path(__file__).resolve().parents[1] / "src"
BRAIN_PACKAGES = ("world_model", "system1", "system2", "awareness", "memory", "skills", "learning")

# Keys of Puck2D ``info`` and TaskInstance/Puck2DConfig ground-truth fields.
FORBIDDEN_NAMES = {
    "ground_truth", "pos", "vel", "applied_force", "commanded_force", "mass", "friction",
    "active_disturbances", "goal_reached", "collision", "action_clipped",
    "obstacles", "impulses", "mass_changes", "friction_patches", "start", "start_pos",
    "start_vel", "damping", "seed", "config",
}
CONTRACT_TYPES = tuple(getattr(contracts, n) for n in contracts.__all__ if isinstance(getattr(contracts, n), type))
ALLOWED_TYPES = CONTRACT_TYPES + (TaskBrief, np.random.Generator, str, int, float, bool, type(None), tuple)

RICH = EvalSetSpec(
    n_tasks=4, max_obstacles=2, max_impulses=2, mass_change_prob=1.0, friction_patch_prob=1.0,
    max_disturbance_step=20,
)
NOISY = {"pos_noise_std": 0.01, "vel_noise_std": 0.01, "max_steps": 150}


def walk(value: Any, path: str = "arg") -> Iterator[tuple[str, Any]]:
    """Yield ``(path, node)`` for a value and everything reachable inside it."""
    yield path, value
    if isinstance(value, np.random.Generator):
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from walk(getattr(value, f.name), f"{path}.{f.name}")
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield f"{path}[{k!r}]", k
            yield from walk(v, f"{path}[{k!r}]")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")
    elif hasattr(value, "__dict__"):
        for k, v in vars(value).items():
            yield from walk(v, f"{path}.{k}")


def leaks(value: Any, path: str = "arg") -> list[str]:
    """Ground-truth carriers in ``value``: forbidden field names or non-allowed types."""
    out = []
    for p, node in walk(value, path):
        if not isinstance(node, ALLOWED_TYPES):
            out.append(f"{p}: type {type(node).__name__}")
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            out += [f"{p}.{f.name}" for f in dataclasses.fields(node) if f.name in FORBIDDEN_NAMES]
        if isinstance(node, str) and p.endswith("]") and node in FORBIDDEN_NAMES:
            out.append(f"{p}: key {node!r}")
    return out


class Spy:
    """Implements every brain interface; records all it receives, delegating behaviour."""

    version = "spy-0"

    def __init__(self) -> None:
        self.received: list[tuple[str, tuple]] = []
        self._pd = PDController()
        self._est = PassThroughStateEstimator()
        self._aw = PassThroughAwareness()
        self._ask_s2 = True
        self.calls: dict[str, int] = {}

    def _record(self, method: str, *args: Any) -> None:
        self.calls[method] = self.calls.get(method, 0) + 1
        self.received.append((method, args))

    def reset(self, task=None):
        self._record("reset", task)
        self._pd.reset(task)

    def estimate(self, observation, rng):
        self._record("estimate", observation, rng)
        return self._est.estimate(observation, rng)

    def predict(self, request, rng):
        self._record("predict", request, rng)
        return None

    def propose(self, state, prediction, rng):
        self._record("propose", state, prediction, rng)
        return self._pd.propose(state, prediction, rng)

    def plan(self, state, prediction, proposal, rng):
        self._record("plan", state, prediction, proposal, rng)
        return None

    def arbitrate(self, state, prediction, proposal, plan, rng) -> AwarenessDecision:
        self._record("arbitrate", state, prediction, proposal, plan, rng)
        # Alternate request_s2 / pass-through so System 2 is exercised every cycle.
        self._ask_s2 = not self._ask_s2
        if not self._ask_s2:
            return AwarenessDecision("request_s2", "spy", (), None)
        return self._aw.arbitrate(state, prediction, proposal, plan, rng)

    def floats(self) -> set[float]:
        """Every state value received (Observation/StateUpdate values, PredictionRequest state)."""
        return {
            float(n) for _, args in self.received for p, n in walk(args)
            if isinstance(n, float) and (".values[" in p or ".state[" in p)
        }


def _true_floats(records) -> set[float]:
    out = set()
    for r in records:
        gt = r.data.get("ground_truth")
        if gt:
            out.update(float(x) for x in gt["pos"] + gt["vel"])
    return out


def _run_spy(tmp_path, env_params, seed):
    es = build_eval_set(seed, RICH, env_params=env_params)
    spy = Spy()
    path = tmp_path / f"spy-{seed}.jsonl"
    with TelemetryLog(path) as log:
        metrics = run_episodes(
            es, spy, log, seed=seed,
            brain={"state_estimator": spy, "world_model": spy, "system2": spy, "awareness": spy},
        )
    return es, spy, metrics, read_records(path)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spy_never_receives_ground_truth(tmp_path, seed):
    es, spy, metrics, records = _run_spy(tmp_path, NOISY, seed)
    # Every brain interface was exercised, every episode.
    for method in ("reset", "estimate", "predict", "propose", "plan", "arbitrate"):
        assert spy.calls.get(method, 0) > 0, method
    assert spy.calls["reset"] == len(metrics) == len(es.tasks)
    assert sum(m.steps for m in metrics) > 0
    # Disturbances actually happened (so there was ground truth to leak).
    gts = [r.data["ground_truth"] for r in records if r.component == C_ACTUATE]
    assert gts and any(g["active_disturbances"] for g in gts)

    # No ground-truth field names, objects (Puck2D, TaskInstance, info dicts) ...
    found = [x for i, (m, args) in enumerate(spy.received) for x in leaks(args, f"{m}#{i}")]
    assert found == []
    # ... and no true state value: brain inputs are noisy observations.
    truth = _true_floats(records) - {0.0}
    assert truth
    assert spy.floats() & truth == set()
    # The reset brief is the task instruction only.
    briefs = [args[0] for m, args in spy.received if m == "reset"]
    assert briefs == [TaskBrief(t.task_id, t.goal) for t in es.tasks]


def test_ground_truth_reaches_telemetry_only(tmp_path):
    es, spy, metrics, records = _run_spy(tmp_path, NOISY, 3)
    starts = [r for r in records if r.decision == "episode_start"]
    assert len(starts) == len(es.tasks)
    assert all("ground_truth" in r.data for r in starts)
    act = [r for r in records if r.component == C_ACTUATE]
    assert len(act) == sum(m.steps for m in metrics)
    assert all({"pos", "vel", "mass", "friction", "active_disturbances"} <= set(r.data["ground_truth"])
               for r in act)
    # The harness's final distance metric is the ground truth's, not the observation's.
    last = act[-1].data["ground_truth"]
    goal = es.tasks[-1].goal
    assert metrics[-1].final_distance == pytest.approx(math.dist(last["pos"], goal), abs=1e-12)


def test_spy_detectors_are_not_vacuous(tmp_path):
    # Without sensor noise the observation equals the true state: the value check must fire.
    _, spy, _, records = _run_spy(tmp_path, {"max_steps": 50}, 0)
    assert spy.floats() & (_true_floats(records) - {0.0})
    # The structural check flags the things the runner must never pass.
    task = TaskInstance("t", 0, (0.0, 0.0), (1.0, 0.0))
    assert leaks(task)
    assert leaks({"ground_truth": {"pos": [0.0, 0.0]}})
    from simulation.puck2d import Puck2D

    assert leaks(Puck2D())
    assert leaks(Puck2D().ground_truth())
    ok = ActionProposal("p", "v", (1.0, 2.0), 0.5, Uncertainty())
    assert leaks((ok, TaskBrief("t", (1.0, 0.0)), np.random.default_rng(0))) == []


def test_brain_slots_validated(tmp_path):
    es = EvalSet("one", 0, RICH, {"max_steps": 5}, (TaskInstance("t", 0, (0.0, 0.0), (1.0, 0.0)),))
    with TelemetryLog(tmp_path / "t.jsonl") as log:
        with pytest.raises(ValueError):
            run_episodes(es, PDController(), log, seed=0, brain={"environment": object()})


# -- static: brain packages must not import simulation -------------------------------------


def _module_name(path: Path, src: Path) -> str:
    rel = path.relative_to(src).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def imported_modules(path: Path, src: Path = SRC) -> set[str]:
    """Absolute module names imported by a file (incl. importlib / __import__ string literals)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pkg = _module_name(path, src)
    if path.name != "__init__.py":
        pkg = pkg.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".") if pkg else []
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("import_module", "__import__") and isinstance(node.args[0].value, str):
                out.add(node.args[0].value)
    return {m for m in out if m}


def _src_file(module: str, src: Path) -> list[Path]:
    p = src.joinpath(*module.split("."))
    return [f for f in (p.with_suffix(".py"), p / "__init__.py") if f.is_file()]


def simulation_import_chains(packages, src: Path = SRC) -> list[list[str]]:
    """Import chains from a brain package file to ``simulation`` (transitive within ``src``)."""
    chains = []
    for pkg in packages:
        for start in sorted((src / pkg).rglob("*.py")):
            stack, seen = [(start, [str(start.relative_to(src))])], set()
            while stack:
                f, chain = stack.pop()
                if f in seen:
                    continue
                seen.add(f)
                for mod in sorted(imported_modules(f, src)):
                    if mod == "simulation" or mod.startswith("simulation."):
                        chains.append(chain + [mod])
                        continue
                    for g in _src_file(mod, src):
                        stack.append((g, chain + [mod]))
    return chains


def test_brain_packages_exist():
    for pkg in BRAIN_PACKAGES:
        assert (SRC / pkg / "__init__.py").is_file(), pkg


def test_no_brain_package_imports_simulation():
    assert simulation_import_chains(BRAIN_PACKAGES) == []


@pytest.mark.parametrize("source", [
    "import simulation",
    "import simulation.puck2d as p",
    "from simulation import puck2d",
    "from simulation.harness import run_episodes",
    "import importlib\nimportlib.import_module('simulation.puck2d')",
    "__import__('simulation')",
    "from .helper import x",               # helper imports simulation
    "from world_model.helper import x",    # transitive through src
])
def test_static_check_detects_simulation_imports(tmp_path, source):
    src = tmp_path / "src"
    (src / "world_model").mkdir(parents=True)
    (src / "simulation").mkdir()
    (src / "simulation" / "__init__.py").write_text("", encoding="utf-8")
    (src / "world_model" / "__init__.py").write_text("", encoding="utf-8")
    (src / "world_model" / "helper.py").write_text("from simulation import puck2d\nx = 1\n", encoding="utf-8")
    (src / "world_model" / "model.py").write_text(source + "\n", encoding="utf-8")
    chains = simulation_import_chains(["world_model"], src)
    assert any(c[0].endswith("model.py") for c in chains), chains


def test_static_check_passes_clean_package(tmp_path):
    src = tmp_path / "src"
    (src / "world_model").mkdir(parents=True)
    (src / "world_model" / "__init__.py").write_text(
        "import numpy\nfrom contracts import Observation\nNAME = 'simulation'\n", encoding="utf-8"
    )
    assert simulation_import_chains(["world_model"], src) == []
