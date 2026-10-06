"""Episode harness and frozen evaluation sets for Puck2D (Gate 1, REQ-SIM, REQ-LOG, REQ-REPRO; ENG-0009).

Evaluation sets
---------------
:func:`build_eval_set` draws a list of :class:`TaskInstance` (start, goal,
obstacles, disturbance schedule: impulses, mass changes, friction patches) from
``numpy.random.default_rng(seed)`` according to an :class:`EvalSetSpec`. The
result, an :class:`EvalSet`, is frozen: it is saved as JSON
(:meth:`EvalSet.save`) and evaluation uses the saved file, so a later change of
the builder cannot silently change a set that results were computed on.
``EvalSet.from_json(s.to_json()) == s``. ``env_params`` holds the shared
Puck2D physics (mass, friction, dt, max_steps, noise, ...); each task supplies
the task-specific Puck2D fields and its environment seed.

Episodes
--------
:func:`run_episodes` runs N episodes. Each episode builds a fresh
:class:`Puck2D`, :class:`SafetyKernel` and :class:`CycleRunner` (with the policy
in the runner's ``system1`` slot) that share one :class:`TelemetryLog`, so every
action reaches the environment only as a kernel ``actuator_command``. The
kernel's limits are the environment's: action limits = Puck2D force limits,
``mass_kg`` = initial mass, ``control_dt_s`` = ``dt``; the workspace, command
age, watchdog timeout and limit mode come from :class:`HarnessConfig`.

Simulation clock: the runner and the kernel share a :class:`SimClock` that
reads ``cycle * dt``, with ``cycle`` the 0-based control-cycle index set by the
harness before each :meth:`CycleRunner.step`. Every cycle in which the kernel
returns a result steps Puck2D exactly once, so while every cycle actuates (the
case for both reference policies) the clock equals ``Puck2D.time`` (episode
step x dt) at the start of each cycle. A cycle that does not actuate (no
command and no watchdog result) still consumes one control period, so the
watchdog sees time pass instead of stalling; such cycles are counted in
``idle_cycles``. An episode ends when Puck2D terminates or truncates, or after
``max_steps`` cycles.

Metrics per episode (:class:`EpisodeMetrics`): success (goal reached), steps
(Puck2D steps), time_to_goal (``steps * dt`` on success, else ``None``),
path_length (sum of ground-truth displacement norms), collisions (0/1; Puck2D
terminates on contact), and safety_interventions: the number of cycles whose
:class:`KernelResult` was not a clean approval, broken down into clamps
(approve with violated constraints), rejections, watchdog timeouts and
emergency stops.

Determinism (REQ-REPRO): environment seeds come from the tasks, the runner's
Generator is ``default_rng([seed, episode])`` and time is simulated, so the
same eval set, policy config and seed give identical metrics and identical
telemetry decision sequences (:func:`decision_trace`). ``latency_ms`` (measured
wall time) and the kernel's ``event_id`` (uuid4) are excluded from the trace.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from robot.runner import CycleRunner, Modules
from safety import KernelResult, SafetyConfig, SafetyKernel
from simulation.puck2d import (
    FrictionPatch,
    Impulse,
    MassChange,
    Obstacle,
    Puck2D,
    Puck2DConfig,
    _items,
    _vec2,
)
from state.telemetry import TelemetryLog, TelemetryRecord

HARNESS_VERSION = "episode-harness-0.1.0"
EVAL_SET_VERSION = "puck2d-evalset-0.1.0"
COMPONENT = "harness"

# Puck2D fields owned by a TaskInstance (not allowed in EvalSet.env_params).
TASK_FIELDS = ("start_pos", "goal", "obstacles", "impulses", "mass_changes", "friction_patches")


def _json_normal(value: Any) -> Any:
    """The value as it reads back from JSON (tuples become lists)."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _check_keys(raw: Any, cls: type, name: str) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name}: expected object, got {type(raw).__name__}")
    names = {f.name for f in fields(cls)}
    missing = sorted(names - set(raw))
    unknown = sorted(set(raw) - names)
    if missing or unknown:
        raise ValueError(f"{name}: missing {missing}, unknown {unknown}")


# ---------------------------------------------------------------------------
# Evaluation sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskInstance:
    """One task: Puck2D start/goal/obstacles/disturbances plus the environment seed."""

    task_id: str
    seed: int
    start: tuple[float, float]
    goal: tuple[float, float]
    obstacles: tuple[Obstacle, ...] = ()
    impulses: tuple[Impulse, ...] = ()
    mass_changes: tuple[MassChange, ...] = ()
    friction_patches: tuple[FrictionPatch, ...] = ()

    def __post_init__(self) -> None:
        s = lambda n, v: object.__setattr__(self, n, v)  # noqa: E731
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("TaskInstance.task_id must be a non-empty str")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"TaskInstance.seed must be a non-negative int, got {self.seed!r}")
        s("start", _vec2(self.start, "TaskInstance.start"))
        s("goal", _vec2(self.goal, "TaskInstance.goal"))
        s("obstacles", _items(self.obstacles, Obstacle, "obstacles"))
        s("impulses", _items(self.impulses, Impulse, "impulses"))
        s("mass_changes", _items(self.mass_changes, MassChange, "mass_changes"))
        s("friction_patches", _items(self.friction_patches, FrictionPatch, "friction_patches"))

    def env_config(self, env_params: Optional[Mapping[str, Any]] = None) -> Puck2DConfig:
        """Puck2D config: shared ``env_params`` plus this task's fields."""
        base = Puck2DConfig.from_mapping(dict(env_params or {}))
        return replace(
            base,
            start_pos=self.start,
            goal=self.goal,
            obstacles=self.obstacles,
            impulses=self.impulses,
            mass_changes=self.mass_changes,
            friction_patches=self.friction_patches,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_normal(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskInstance":
        _check_keys(raw, cls, "TaskInstance")
        return cls(**raw)


@dataclass(frozen=True)
class EvalSetSpec:
    """Builder parameters. Ranges are ``(low, high)``; counts are inclusive maxima."""

    n_tasks: int = 20
    arena_low: tuple[float, float] = (-1.0, -1.0)
    arena_high: tuple[float, float] = (1.0, 1.0)
    min_start_goal_distance: float = 0.5
    max_obstacles: int = 2
    obstacle_radius: tuple[float, float] = (0.05, 0.15)
    obstacle_clearance: float = 0.1     # free margin around start and goal disc
    max_impulses: int = 1
    impulse_max: float = 1.0            # N s, per axis
    max_disturbance_step: int = 200     # disturbances occur at steps [0, max)
    mass_change_prob: float = 0.2
    mass_range: tuple[float, float] = (0.5, 2.0)
    friction_patch_prob: float = 0.2
    friction_range: tuple[float, float] = (0.0, 0.3)
    patch_radius: tuple[float, float] = (0.1, 0.3)

    def __post_init__(self) -> None:
        for n in ("n_tasks", "max_obstacles", "max_impulses", "max_disturbance_step"):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"EvalSetSpec.{n} must be a non-negative int, got {v!r}")
        if self.max_disturbance_step < 1:
            raise ValueError("EvalSetSpec.max_disturbance_step must be >= 1")
        for n in ("arena_low", "arena_high", "obstacle_radius", "mass_range",
                  "friction_range", "patch_radius"):
            object.__setattr__(self, n, _vec2(getattr(self, n), f"EvalSetSpec.{n}"))
        for n in ("min_start_goal_distance", "obstacle_clearance", "impulse_max",
                  "mass_change_prob", "friction_patch_prob"):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and v >= 0):
                raise ValueError(f"EvalSetSpec.{n} must be a finite number >= 0, got {v!r}")
            object.__setattr__(self, n, float(v))
        if not all(lo < hi for lo, hi in zip(self.arena_low, self.arena_high)):
            raise ValueError("EvalSetSpec: arena_low must be < arena_high")
        for n in ("obstacle_radius", "mass_range", "patch_radius", "friction_range"):
            lo, hi = getattr(self, n)
            strict = n != "friction_range"
            if lo > hi or lo < 0 or (strict and lo == 0):
                raise ValueError(f"EvalSetSpec.{n} must be 0 {'<' if strict else '<='} low <= high")
        if self.mass_change_prob > 1 or self.friction_patch_prob > 1:
            raise ValueError("EvalSetSpec probabilities must be <= 1")
        diag = math.dist(self.arena_low, self.arena_high)
        if self.min_start_goal_distance >= diag:
            raise ValueError("EvalSetSpec.min_start_goal_distance must be < the arena diagonal")


@dataclass(frozen=True)
class EvalSet:
    """A frozen, versioned list of tasks generated from ``seed`` and ``spec``."""

    name: str
    seed: int
    spec: EvalSetSpec
    env_params: Mapping[str, Any]
    tasks: tuple[TaskInstance, ...]
    version: str = EVAL_SET_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("EvalSet.name must be a non-empty str")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"EvalSet.seed must be a non-negative int, got {self.seed!r}")
        if isinstance(self.spec, Mapping):
            _check_keys(self.spec, EvalSetSpec, "EvalSet.spec")
            object.__setattr__(self, "spec", EvalSetSpec(**self.spec))
        if not isinstance(self.spec, EvalSetSpec):
            raise ValueError("EvalSet.spec must be an EvalSetSpec")
        owned = sorted(set(self.env_params) & set(TASK_FIELDS))
        if owned:
            raise ValueError(f"EvalSet.env_params must not set task fields {owned}")
        Puck2DConfig.from_mapping(dict(self.env_params))  # validate
        object.__setattr__(self, "env_params", _json_normal(dict(self.env_params)))
        tasks = tuple(
            t if isinstance(t, TaskInstance) else TaskInstance.from_dict(t) for t in self.tasks
        )
        ids = [t.task_id for t in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("EvalSet task ids must be unique")
        object.__setattr__(self, "tasks", tasks)
        if self.version != EVAL_SET_VERSION:
            raise ValueError(f"EvalSet.version {self.version!r} != {EVAL_SET_VERSION!r}")

    def __len__(self) -> int:
        return len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "version": self.version,
            "spec": _json_normal(asdict(self.spec)),
            "env_params": _json_normal(dict(self.env_params)),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvalSet":
        _check_keys(raw, cls, "EvalSet")
        return cls(**raw)

    @classmethod
    def from_json(cls, text: str) -> "EvalSet":
        return cls.from_dict(json.loads(text))

    def save(self, path: Union[str, os.PathLike]) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "EvalSet":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())


def _r(x: float) -> float:
    return round(float(x), 6)


def build_eval_set(
    seed: int,
    spec: Optional[EvalSetSpec] = None,
    env_params: Optional[Mapping[str, Any]] = None,
    name: str = "puck2d-eval",
) -> EvalSet:
    """Draw ``spec.n_tasks`` tasks from ``default_rng(seed)``; deterministic in its inputs."""
    spec = EvalSetSpec() if spec is None else spec
    env_params = dict(env_params or {})
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    base = Puck2DConfig.from_mapping(env_params)
    rng = np.random.default_rng(seed)
    lo, hi = np.array(spec.arena_low), np.array(spec.arena_high)

    def point() -> tuple[float, float]:
        p = rng.uniform(lo, hi)
        return (_r(p[0]), _r(p[1]))

    tasks = []
    for i in range(spec.n_tasks):
        task_seed = int(rng.integers(0, 2**31 - 1))
        start = point()
        goal = point()
        for _ in range(1000):
            if math.dist(start, goal) >= spec.min_start_goal_distance:
                break
            goal = point()
        else:
            raise RuntimeError("could not place a goal far enough from the start")

        obstacles = []
        for _ in range(int(rng.integers(0, spec.max_obstacles + 1))):
            for _ in range(100):
                c = point()
                r = _r(rng.uniform(*spec.obstacle_radius))
                pad = r + base.puck_radius + spec.obstacle_clearance
                if math.dist(c, start) > pad and math.dist(c, goal) > pad + base.goal_radius:
                    obstacles.append(Obstacle(c, r))
                    break

        impulses = [
            Impulse(
                int(rng.integers(0, spec.max_disturbance_step)),
                tuple(_r(v) for v in rng.uniform(-spec.impulse_max, spec.impulse_max, 2)),
            )
            for _ in range(int(rng.integers(0, spec.max_impulses + 1)))
        ]
        mass_changes = []
        if rng.random() < spec.mass_change_prob:
            mass_changes.append(
                MassChange(int(rng.integers(0, spec.max_disturbance_step)),
                           _r(rng.uniform(*spec.mass_range)))
            )
        patches = []
        if rng.random() < spec.friction_patch_prob:
            patches.append(
                FrictionPatch(
                    friction=_r(rng.uniform(*spec.friction_range)),
                    center=point(),
                    radius=_r(rng.uniform(*spec.patch_radius)),
                )
            )
        task = TaskInstance(
            task_id=f"{name}-s{seed}-t{i:04d}",
            seed=task_seed,
            start=start,
            goal=goal,
            obstacles=tuple(obstacles),
            impulses=tuple(impulses),
            mass_changes=tuple(mass_changes),
            friction_patches=tuple(patches),
        )
        task.env_config(env_params)  # the task must make a valid environment
        tasks.append(task)
    return EvalSet(name=name, seed=seed, spec=spec, env_params=env_params, tasks=tuple(tasks))


# ---------------------------------------------------------------------------
# Episode harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessConfig:
    """Safety Kernel settings not derived from the environment."""

    workspace_low: tuple[float, float] = (-2.0, -2.0)
    workspace_high: tuple[float, float] = (2.0, 2.0)
    max_command_age_s: float = 0.05
    watchdog_timeout_s: float = 0.1
    limit_mode: str = "clamp"
    operator_key: str = "harness-operator"

    def safety_config(self, env: Puck2DConfig) -> SafetyConfig:
        return SafetyConfig(
            axis_names=("x", "y"),
            action_low=env.force_low,
            action_high=env.force_high,
            workspace_low=self.workspace_low,
            workspace_high=self.workspace_high,
            max_command_age_s=self.max_command_age_s,
            watchdog_timeout_s=self.watchdog_timeout_s,
            limit_mode=self.limit_mode,
            mass_kg=env.mass,
            control_dt_s=env.dt,
        )


class SimClock:
    """Simulation time ``cycle * dt`` (seconds); ``cycle`` is set by the harness."""

    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        self.cycle = 0

    def __call__(self) -> float:
        return self.cycle * self.dt


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    task_id: str
    success: bool
    steps: int
    cycles: int
    idle_cycles: int
    time_to_goal: Optional[float]
    path_length: float
    final_distance: float
    collisions: int
    truncated: bool
    safety_interventions: int
    kernel_clamps: int
    kernel_rejections: int
    watchdog_timeouts: int
    emergency_stops: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify(result: Optional[KernelResult]) -> Optional[str]:
    """Intervention kind of one cycle's KernelResult, or None for none / clean approval."""
    if result is None:
        return None
    d = result.decision
    if d.verdict == "approve":
        return "clamp" if d.violated_constraints else None
    if d.verdict == "emergency_stop":
        return "emergency_stop"
    return "watchdog" if "watchdog_timeout" in d.violated_constraints else "reject"


def run_episode(
    eval_set: EvalSet,
    task: TaskInstance,
    policy: Any,
    telemetry: TelemetryLog,
    *,
    seed: int,
    episode: int = 0,
    config: Optional[HarnessConfig] = None,
    run_id: str = "harness",
) -> EpisodeMetrics:
    """Run one episode of ``task`` with ``policy`` as System 1 through the cycle runner."""
    cfg = HarnessConfig() if config is None else config
    env = Puck2D(task.env_config(eval_set.env_params), seed=task.seed)
    ecfg = env.config
    clock = SimClock(ecfg.dt)
    kernel = SafetyKernel(cfg.safety_config(ecfg), telemetry, cfg.operator_key, clock=clock)
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset(task)
    ep_id = f"{run_id}-ep{episode}"
    runner = CycleRunner(
        kernel,
        telemetry,
        np.random.default_rng([seed, episode]),
        Modules(environment=env, system1=policy),
        clock=clock,
        run_id=ep_id,
    )
    policy_version = getattr(policy, "version", type(policy).__name__)
    telemetry.log(
        COMPONENT, cycle_id=0, decision="episode_start", reason="ok", latency_ms=0.0,
        model_version=HARNESS_VERSION, event_id=f"{ep_id}-start", timestamp=clock(),
        data={"episode": episode, "task_id": task.task_id, "env_seed": task.seed,
              "runner_seed": [seed, episode], "policy_version": policy_version,
              "eval_set": eval_set.name, "eval_set_version": eval_set.version},
    )

    counts = {"clamp": 0, "reject": 0, "watchdog": 0, "emergency_stop": 0}
    path = 0.0
    cycles = idle = 0
    while cycles < ecfg.max_steps and not (env.terminated or env.truncated):
        clock.cycle = cycles
        p0 = env.pos.copy()
        result = runner.step()
        cycles += 1
        path += float(np.hypot(*(env.pos - p0)))
        if result.actuated_command is None:
            idle += 1
        kind = classify(result.kernel_result)
        if kind is not None:
            counts[kind] += 1
    clock.cycle = cycles

    success = bool(env.goal_reached)
    metrics = EpisodeMetrics(
        episode=episode,
        task_id=task.task_id,
        success=success,
        steps=env.steps,
        cycles=cycles,
        idle_cycles=idle,
        time_to_goal=env.time if success else None,
        path_length=path,
        final_distance=float(np.hypot(*(env.pos - np.array(ecfg.goal)))),
        collisions=0 if env.collision is None else 1,
        truncated=not env.terminated,
        safety_interventions=sum(counts.values()),
        kernel_clamps=counts["clamp"],
        kernel_rejections=counts["reject"],
        watchdog_timeouts=counts["watchdog"],
        emergency_stops=counts["emergency_stop"],
    )
    telemetry.log(
        COMPONENT, cycle_id=runner.cycle_id, decision="episode_end",
        reason="success" if success else ("collision" if metrics.collisions else "failure"),
        latency_ms=0.0, model_version=HARNESS_VERSION, event_id=f"{ep_id}-end",
        timestamp=clock(), data=metrics.to_dict(),
    )
    return metrics


def run_episodes(
    eval_set: EvalSet,
    policy: Any,
    telemetry: TelemetryLog,
    *,
    seed: int,
    n_episodes: Optional[int] = None,
    config: Optional[HarnessConfig] = None,
    run_id: str = "harness",
) -> list[EpisodeMetrics]:
    """Run ``n_episodes`` (default: one per task); episode ``i`` uses task ``i % len``."""
    if not eval_set.tasks:
        raise ValueError("eval set has no tasks")
    n = len(eval_set.tasks) if n_episodes is None else n_episodes
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n_episodes must be a non-negative int, got {n_episodes!r}")
    return [
        run_episode(eval_set, eval_set.tasks[i % len(eval_set.tasks)], policy, telemetry,
                    seed=seed, episode=i, config=config, run_id=run_id)
        for i in range(n)
    ]


def summarize(metrics: Sequence[EpisodeMetrics]) -> dict[str, Any]:
    """Aggregate episode metrics; ``mean_time_to_goal`` is over successes (None if none)."""
    n = len(metrics)
    if n == 0:
        raise ValueError("no episodes to summarize")
    ttg = [m.time_to_goal for m in metrics if m.time_to_goal is not None]
    mean = lambda xs: float(sum(xs) / len(xs))  # noqa: E731
    return {
        "n_episodes": n,
        "success_rate": mean([float(m.success) for m in metrics]),
        "collision_rate": mean([float(m.collisions) for m in metrics]),
        "mean_steps": mean([m.steps for m in metrics]),
        "mean_path_length": mean([m.path_length for m in metrics]),
        "mean_safety_interventions": mean([m.safety_interventions for m in metrics]),
        "mean_time_to_goal": mean(ttg) if ttg else None,
    }


def decision_trace(records: Iterable[TelemetryRecord]) -> list[tuple]:
    """Reproducible view of a telemetry log: every field except latency_ms and event_id."""
    return [
        (r.component, r.cycle_id, r.timestamp, r.level, r.model_version, r.decision,
         r.reason, json.dumps(dict(r.data), sort_keys=True))
        for r in records
    ]
