"""Independent G1-4 checks for the runner boundary and brain import graph."""

import ast
from pathlib import Path
import sys
import tempfile

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contracts import ActionProposal, Observation, Outcome, StateUpdate, Uncertainty
from robot.runner import CycleRunner, Modules, PassThroughAwareness, PassThroughStateEstimator
from safety import SafetyConfig, SafetyKernel
from state.telemetry import TelemetryLog, read_records


class TruthEnvironment:
    """Environment whose private truth is observable only through telemetry callback."""

    def __init__(self):
        self.truth = {"pos": [0.25], "mass": 7.0, "active_disturbances": [{"step": 1}]}

    def observe(self, rng):
        return Observation("sensor", ("pos_x", "vel_x"), (0.2, 0.0), Uncertainty())

    def actuate(self, command, rng):
        return Outcome("out", False, 0.0, (), (), Uncertainty())

    def ground_truth(self):
        return self.truth


class CaptureEstimator:
    def __init__(self):
        self.seen = []

    def estimate(self, observation, rng):
        self.seen.append(observation)
        return StateUpdate("physical", observation.channels, observation.values,
                           observation.uncertainty, ())


class QuietPolicy:
    def propose(self, state, prediction, rng):
        return ActionProposal("p", "quiet", (0.0,), 1.0, Uncertainty())


def test_runner_keeps_ground_truth_out_of_brain_calls_and_logs_it():
    env, estimator = TruthEnvironment(), CaptureEstimator()
    config = SafetyConfig(axis_names=("x",), action_low=(-1.0,), action_high=(1.0,),
                          workspace_low=(-1.0,), workspace_high=(1.0,),
                          max_command_age_s=1.0, watchdog_timeout_s=2.0)
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
        path = Path(directory) / "telemetry.jsonl"
        with TelemetryLog(path) as log:
            kernel = SafetyKernel(config, log, "operator", clock=lambda: 0.0)
            runner = CycleRunner(kernel, log, np.random.default_rng(4),
                                 Modules(environment=env, state_estimator=estimator,
                                         system1=QuietPolicy(), awareness=PassThroughAwareness()),
                                 clock=lambda: 0.0)
            runner.step()
        records = read_records(path)

    assert len(estimator.seen) == 1
    assert estimator.seen[0].values == (0.2, 0.0)
    assert not hasattr(estimator.seen[0], "truth")
    actuations = [r for r in records if r.component == "env_actuate"]
    assert len(actuations) == 1, [(r.component, r.decision, r.reason) for r in records]
    assert actuations[0].data["ground_truth"] == env.truth


def test_brain_source_import_graph_does_not_reach_simulation():
    src = SRC
    brains = ("world_model", "system1", "system2", "awareness", "memory", "skills", "learning")
    seen = set()
    pending = []
    for package in brains:
        pending.extend((p, package) for p in (src / package).rglob("*.py"))
    while pending:
        path, root = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert module != "simulation" and not module.startswith("simulation."), (path, module)
                if module and not module.startswith("."):
                    candidate = src.joinpath(*module.split("."))
                    pending.extend((p, root) for p in (candidate.with_suffix(".py"), candidate / "__init__.py") if p.is_file())
