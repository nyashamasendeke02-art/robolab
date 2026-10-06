"""Episode harness: run policies on Puck2D through the cycle runner (Gate 1, REQ-SIM,
REQ-LOG, REQ-REPRO; ENG-0008).

:func:`run_episodes` runs one episode per :class:`~simulation.evalset.TaskInstance`.
Each episode builds, from the task and :class:`HarnessConfig`:

* a :class:`~simulation.puck2d.Puck2D` (seeded with ``task.seed``) as the
  runner's environment;
* a fresh :class:`~safety.SafetyKernel` whose limits come from the environment
  (force limits, nominal mass, ``dt``) and the configured workspace; it is the
  only source of actuator commands (the runner sends every accepted proposal to
  :meth:`SafetyKernel.check`);
* a :class:`~robot.runner.CycleRunner` with the policy as its System 1 module
  and ``numpy.random.default_rng(task.seed)`` as its Generator;
* one append-only telemetry file ``<out_dir>/<run_id>-<task_id>.jsonl`` shared
  by the runner and the kernel.

Simulation clock: the runner, the kernel and the telemetry log share one
:class:`SimClock` that reads ``k * dt`` during the ``k``-th cycle (``k = 0, 1,
...``; one cycle = one environment step whenever the cycle actuates), so
staleness and watchdog decisions depend only on the episode, not on wall time.
The clock advances once per cycle even when nothing is actuated, so the
watchdog still fires if a policy goes silent.

An episode ends when the environment terminates (goal or collision) or
truncates, or after ``env.max_steps`` cycles (counted as truncated).

Per-episode metrics (:class:`EpisodeMetrics`): ``success`` (goal reached),
``steps`` (environment steps), ``cycles``, ``time_to_goal`` (``steps * dt`` at
success, else None), ``path_length`` (sum of true-position step lengths),
``collisions`` (0/1: a collision ends a Puck2D episode), ``safety_interventions``
(cycles whose actuating :class:`KernelResult` was not a plain approve: verdict
``reject`` / ``emergency_stop``, or ``approve`` with violated constraints, i.e.
a clamp) with the per-verdict split, ``final_distance`` and ``estopped``.
The metrics of all episodes are also written to ``<out_dir>/<run_id>-episodes.json``.

Determinism: same tasks + same policy configuration give identical metrics and
identical :func:`decision_sequence` of the telemetry. ``latency_ms`` (wall-clock
module timing) and the kernel's ``event_id`` (``uuid4``, assigned inside
:mod:`safety`) are not deterministic and are excluded from the sequence.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from robot.runner import CycleRunner, Modules
from safety import SafetyConfig, SafetyKernel
from simulation.evalset import TaskInstance
from simulation.puck2d import Puck2D, Puck2DConfig
from state.telemetry import TelemetryLog, TelemetryRecord

HARNESS_VERSION = "episode-harness-0.1.0"
AXES = ("x", "y")


class SimClock:
    """Simulation time in seconds; set by the harness, read by runner, kernel and log."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


@dataclass(frozen=True)
class HarnessConfig:
    """Environment base config and Safety Kernel settings for every episode.

    ``max_command_age_s`` / ``watchdog_timeout_s`` default to ``env.dt`` and
    ``5 * env.dt``.
    """

    env: Puck2DConfig = field(default_factory=Puck2DConfig)
    workspace_low: tuple[float, float] = (-1.5, -1.5)
    workspace_high: tuple[float, float] = (1.5, 1.5)
    limit_mode: str = "reject"
    max_command_age_s: Optional[float] = None
    watchdog_timeout_s: Optional[float] = None

    def safety_config(self) -> SafetyConfig:
        dt = self.env.dt
        return SafetyConfig(
            axis_names=AXES,
            action_low=self.env.force_low,
            action_high=self.env.force_high,
            workspace_low=self.workspace_low,
            workspace_high=self.workspace_high,
            max_command_age_s=dt if self.max_command_age_s is None else self.max_command_age_s,
            watchdog_timeout_s=5 * dt if self.watchdog_timeout_s is None else self.watchdog_timeout_s,
            limit_mode=self.limit_mode,
            mass_kg=self.env.mass,
            control_dt_s=dt,
        )


@dataclass(frozen=True)
class EpisodeMetrics:
    task_id: str
    seed: int
    policy_version: str
    success: bool
    steps: int
    cycles: int
    time_to_goal: Optional[float]
    path_length: float
    collisions: int
    safety_interventions: int
    kernel_rejects: int
    kernel_estops: int
    kernel_clamps: int
    final_distance: float
    terminated: bool
    truncated: bool
    estopped: bool
    telemetry_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_episode(
    policy: Any,
    task: TaskInstance,
    telemetry_path: Union[str, os.PathLike],
    config: Optional[HarnessConfig] = None,
    run_id: str = "episode",
) -> EpisodeMetrics:
    """Run one episode of ``task`` with ``policy`` as System 1 (see module doc)."""
    config = HarnessConfig() if config is None else config
    path = Path(telemetry_path)
    if path.exists():
        # Telemetry is append-only and immutable: never mix two episodes in one file.
        raise FileExistsError(f"telemetry file already exists: {path}")
    env = Puck2D(task.env_config(config.env), seed=task.seed)
    dt = env.config.dt
    clock = SimClock(0.0)
    if hasattr(policy, "reset"):
        policy.reset(task)

    with TelemetryLog(path, clock=clock) as log:
        # Fresh kernel per episode: its watchdog and e-stop state start at t = 0.
        # The operator key is never used by the harness, so an e-stop stays latched.
        kernel = SafetyKernel(config.safety_config(), log, secrets.token_hex(16), clock=clock)
        runner = CycleRunner(
            kernel, log, np.random.default_rng(task.seed),
            modules=Modules(environment=env, system1=policy),
            clock=clock, run_id=f"{run_id}-{task.task_id}",
        )
        cycles = rejects = estops = clamps = 0
        path_length = 0.0
        while not (env.terminated or env.truncated) and cycles < env.config.max_steps:
            clock.t = cycles * dt
            p_before = env.pos.copy()
            result = runner.step()
            cycles += 1
            path_length += float(np.hypot(*(env.pos - p_before)))
            kr = result.kernel_result
            if kr is not None:
                verdict = kr.decision.verdict
                rejects += verdict == "reject"
                estops += verdict == "emergency_stop"
                clamps += verdict == "approve" and bool(kr.decision.violated_constraints)
        estopped = kernel.estopped

    success = bool(env.goal_reached)
    return EpisodeMetrics(
        task_id=task.task_id,
        seed=task.seed,
        policy_version=str(getattr(policy, "version", type(policy).__name__)),
        success=success,
        steps=env.steps,
        cycles=cycles,
        time_to_goal=env.time if success else None,
        path_length=path_length,
        collisions=int(env.collision is not None),
        safety_interventions=rejects + estops + clamps,
        kernel_rejects=rejects,
        kernel_estops=estops,
        kernel_clamps=clamps,
        final_distance=float(np.hypot(*(env.pos - np.array(env.config.goal)))),
        terminated=bool(env.terminated),
        truncated=not env.terminated,
        estopped=estopped,
        telemetry_path=str(path),
    )


def run_episodes(
    policy: Any,
    tasks: Iterable[TaskInstance],
    out_dir: Union[str, os.PathLike],
    config: Optional[HarnessConfig] = None,
    run_id: str = "run",
) -> list[EpisodeMetrics]:
    """Run one episode per task (in order); write telemetry and ``<run_id>-episodes.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"{run_id}-episodes.json"
    if summary_path.exists():
        raise FileExistsError(f"episode record already exists: {summary_path}")
    metrics = [
        run_episode(policy, t, out / f"{run_id}-{t.task_id}.jsonl", config, run_id)
        for t in tasks
    ]
    record = {
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "policy_version": str(getattr(policy, "version", type(policy).__name__)),
        "summary": summarize(metrics),
        "episodes": [m.to_dict() for m in metrics],
    }
    with open(summary_path, "x", encoding="utf-8") as fh:
        json.dump(record, fh, allow_nan=False, sort_keys=True, indent=1)
    return metrics


def summarize(metrics: list[EpisodeMetrics]) -> dict[str, Any]:
    """Aggregate counts and means over episodes (None where undefined)."""
    n = len(metrics)
    ttg = [m.time_to_goal for m in metrics if m.time_to_goal is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "n_episodes": n,
        "n_success": sum(m.success for m in metrics),
        "success_rate": None if n == 0 else sum(m.success for m in metrics) / n,
        "mean_time_to_goal": mean(ttg),
        "mean_path_length": mean([m.path_length for m in metrics]),
        "total_collisions": sum(m.collisions for m in metrics),
        "total_safety_interventions": sum(m.safety_interventions for m in metrics),
    }


def decision_sequence(records: Iterable[TelemetryRecord]) -> list[tuple]:
    """The reproducible part of a telemetry stream, record by record.

    Drops ``latency_ms`` (wall-clock timing) and ``event_id`` (the kernel uses uuid4).
    """
    return [
        (r.component, r.cycle_id, r.timestamp, r.level, r.model_version, r.decision,
         r.reason, json.dumps(r.data, sort_keys=True))
        for r in records
    ]
