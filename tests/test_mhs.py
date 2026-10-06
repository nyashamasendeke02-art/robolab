"""Model Hardware Standard v0 (G1-5, REQ-MHS, REQ-SAFE+, H5, ADR-001, ADR-003, ADR-005; ENG-0012).

AC1: round trip and malformed MHS raise ContractError.
AC2: a SafetyKernel built from Puck2D's MHS equals / behaves as the hand-configured one.
AC3: brain packages contain no Puck2D constants; brain modules get layouts from the MHS.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from contracts import (
    MHS,
    ActionProposal,
    Actuator,
    Body,
    ContractError,
    Control,
    Envelope,
    Footprint,
    NoiseModel,
    Observation,
    SafetyEnvelope,
    Sensor,
    Uncertainty,
)
from robot.runner import CycleRunner, Modules
from safety import SafetyConfig, SafetyKernel
from simulation.harness import HarnessConfig
from simulation.puck2d import Puck2D, Puck2DConfig
from state.telemetry import TelemetryLog

SRC = Path(__file__).resolve().parents[1] / "src"
BRAIN_PACKAGES = ("state", "world_model", "system1", "system2", "awareness", "memory", "skills", "learning")
KEY = "operator"
T0 = 100.0


def _raw(m: MHS) -> dict:
    return json.loads(m.to_json())


def _decode(raw: dict) -> MHS:
    return MHS.from_json(json.dumps(raw))


def wheeled_mhs() -> MHS:
    """A non-Puck2D body: unknown mass, velocity + steering actuators, vector IMU."""
    return MHS(
        body_name="rover",
        body_class="wheeled",
        actuators=(
            Actuator("drive", "velocity", "m/s", None, "base", -1.0, 1.0, 2.0, 0.05),
            Actuator("steer", "steering", "rad", None, "base", -0.5, 0.5, None, 0.02),
        ),
        action_layout=("steer", "drive"),
        sensors=(
            Sensor("odom_x", "position", "m", (), 50.0, NoiseModel("unknown", None), "map", "x"),
            Sensor("odom_vx", "velocity", "m/s", (), 50.0, NoiseModel("gaussian", 0.01), "map", "x"),
            Sensor("imu_acc", "acceleration", "m/s^2", (3,), 200.0, NoiseModel("gaussian", 0.1), "base", None),
        ),
        observation_layout=("imu_acc[0]", "imu_acc[1]", "imu_acc[2]", "odom_x", "odom_vx"),
        body=Body(None, None, Footprint("box", (0.5, 0.3)), ("map", "base"), "base"),
        control=Control(0.02, 0.05, 0.1, 0.01, ("wheel_current_limit",)),
        safety=SafetyEnvelope(
            workspace_frame="map", workspace_axes=("x",), workspace_low=(-5.0,),
            workspace_high=(5.0,), speed_limits=(1.0,), mass_kg=20.0, mass_lower_bound_kg=10.0,
            braking="actuators", brake_decel_mps2=(2.0,), safe_action=(0.0, 0.0),
            estop_latching=True, estop_reset="operator", estop_action="brake",
        ),
    )


# -- AC1: round trip -------------------------------------------------------------------


@pytest.mark.parametrize("make", [
    lambda: Puck2D().mhs(),
    lambda: Puck2D(Puck2DConfig(pos_noise_std=0.01, puck_radius=0.05, actuator_tau=0.1,
                                mass_changes=[{"step": 3, "mass": 2.0}])).mhs(),
    wheeled_mhs,
])
def test_mhs_json_round_trip(make):
    m = make()
    back = MHS.from_json(m.to_json())
    assert back == m
    assert back.to_json() == m.to_json()


def test_puck2d_publishes_its_layouts():
    m = Puck2D(Puck2DConfig(dt=0.05, force_low=(-3.0, -4.0), force_high=(5.0, 6.0))).mhs()
    assert m.body_class == "point_mass"
    assert m.action_layout == ("force_x", "force_y")
    assert m.observation_layout == ("pos_x", "pos_y", "vel_x", "vel_y")
    assert [(a.low, a.high) for a in m.action_actuators] == [(-3.0, 5.0), (-4.0, 6.0)]
    assert m.control.period_s == 0.05
    assert m.body.mass_kg is None  # true mass is ground truth (G1-4)
    assert m.safety.mass_kg == 1.0 and m.safety.mass_lower_bound_kg is None
    # Observations really use the declared layout.
    assert Puck2D().observe(np.random.default_rng(0)).channels == m.observation_layout


def test_puck2d_mass_bounds_contain_every_mass():
    cfg = Puck2DConfig(mass=1.0, mass_changes=[{"step": 2, "mass": 0.5}, {"step": 9, "mass": 1.5}])
    m = Puck2D(cfg).mhs()
    assert (m.safety.mass_lower_bound_kg, m.safety.mass_kg) == (0.5, 1.5)
    assert Puck2D(cfg).mhs(mass_bounds=(0.25, 3.0)).safety.mass_kg == 3.0
    with pytest.raises(ValueError):
        Puck2D(cfg).mhs(mass_bounds=(0.5, 1.2))  # not an upper bound on the true mass


# -- AC1: malformed MHS ------------------------------------------------------------------


def _mutations():
    """(name, function raw-dict -> None) producing malformed MHS JSON."""
    def act(i, **kw):
        return lambda r: r["actuators"][i].update(kw)

    def drop_act_key(key):
        return lambda r: r["actuators"][0].pop(key)

    def sensor(i, **kw):
        return lambda r: r["sensors"][i].update(kw)

    def top(**kw):
        return lambda r: r.update(kw)

    def safety(**kw):
        return lambda r: r["safety"].update(kw)

    return [
        # missing / invalid actuator limits
        ("missing low", drop_act_key("low")),
        ("missing high", drop_act_key("high")),
        ("null low", act(0, low=None)),
        ("int high", act(0, high=10)),
        ("low > high", act(0, low=11.0)),
        ("bad rate limit", act(0, rate_limit=0.0)),
        ("negative latency", act(0, latency_s=-0.1)),
        # unknown units / kinds
        ("actuator units lbf", act(0, units="lbf")),
        ("actuator units mismatch kind", act(0, units="N*m")),
        ("sensor units ft", sensor(0, units="ft")),
        ("sensor units of other kind", sensor(2, units="m")),
        ("unknown actuator kind", act(0, kind="thruster")),
        ("unknown sensor kind", sensor(0, kind="lidar")),
        ("unknown body class", top(body_class="blimp")),
        # inconsistent layouts
        ("action layout missing", top(action_layout=["force_x"])),
        ("action layout unknown", top(action_layout=["force_x", "force_z"])),
        ("action layout duplicate", top(action_layout=["force_x", "force_x"])),
        ("observation layout missing", top(observation_layout=["pos_x", "pos_y", "vel_x"])),
        ("observation layout unknown", top(observation_layout=["pos_x", "pos_y", "vel_x", "vel_z"])),
        ("observation layout duplicate", top(observation_layout=["pos_x", "pos_x", "vel_x", "vel_y"])),
        ("duplicate actuator", lambda r: r["actuators"].append(dict(r["actuators"][0]))),
        ("safe action length", safety(safe_action=[0.0])),
        ("safe action outside limits", safety(safe_action=[0.0, 11.0])),
        ("workspace length", safety(workspace_low=[-2.0])),
        ("workspace low >= high", safety(workspace_low=[2.0, -2.0])),
        ("unobservable workspace axis", safety(workspace_axes=["x", "z"])),
        ("velocity sensor wrong frame", sensor(2, frame="base")),
        ("sensor frame unknown", sensor(0, frame="moon")),
        ("actuator frame unknown", act(1, frame="moon")),
        ("mass bounds inverted", safety(mass_lower_bound_kg=2.0)),
        ("mass not positive", safety(mass_kg=0.0)),
        ("speed limit length", safety(speed_limits=[1.0])),
        ("bad estop reset", safety(estop_reset="anyone")),
        ("bad braking", safety(braking="hope")),
        ("watchdog shorter than period", lambda r: r["control"].update(watchdog_timeout_s=0.001)),
        ("gaussian without std", sensor(0, noise={"kind": "gaussian", "std": None})),
        ("version", top(mhs_version="9.9.9")),
        ("unknown field", top(colour="red")),
        ("missing section", lambda r: r.pop("safety")),
        ("non-finite", act(0, low=float("-inf"))),
    ]


@pytest.mark.parametrize("name,mutate", _mutations(), ids=[n for n, _ in _mutations()])
def test_malformed_mhs_raises_contract_error(name, mutate):
    raw = _raw(Puck2D().mhs())
    mutate(raw)
    with pytest.raises(ContractError):
        MHS.from_json(json.dumps(raw))


def test_body_mass_outside_bounds_and_forged_objects_rejected():
    m = Puck2D().mhs()
    with pytest.raises(ContractError):
        dataclasses.replace(m, body=dataclasses.replace(m.body, mass_kg=3.0))
    forged = Puck2D().mhs()
    object.__setattr__(forged, "action_layout", ("force_x",))
    with pytest.raises(ContractError):
        forged.validate()
    with pytest.raises(ContractError):
        MHS.from_json(b"{}")
    with pytest.raises(ContractError):
        MHS.from_json("not json")


def test_kernel_rejects_an_mhs_it_cannot_model(tmp_path):
    # Valid MHS, but velocity/steering actuators are outside the kernel's point-mass model.
    with pytest.raises(ContractError):
        SafetyConfig.from_mhs(wheeled_mhs())
    m = Puck2D().mhs()
    swapped = dataclasses.replace(m, action_layout=("force_y", "force_x"))
    with pytest.raises(ContractError):
        SafetyConfig.from_mhs(swapped)  # action component i must act on workspace axis i
    no_brake = dataclasses.replace(m, safety=dataclasses.replace(m.safety, estop_action="power_off"))
    with pytest.raises(ContractError):
        SafetyConfig.from_mhs(no_brake)
    with pytest.raises(TypeError):
        SafetyConfig.from_mhs(_raw(m))


# -- AC2: kernel from MHS == hand-configured kernel -----------------------------------------


ENV_CONFIGS = [
    {},
    {"dt": 0.05, "mass": 2.0},
    {"force_low": (-3.0, -8.0), "force_high": (6.0, 2.0), "mass": 0.7},
]


@pytest.mark.parametrize("env_kw", ENV_CONFIGS)
@pytest.mark.parametrize("limit_mode", ["clamp", "reject"])
def test_kernel_config_from_puck2d_mhs_equals_hand_config(env_kw, limit_mode):
    env = Puck2D(Puck2DConfig(**env_kw))
    hc = HarnessConfig(limit_mode=limit_mode, workspace_low=(-1.5, -1.0), workspace_high=(1.0, 2.5))
    assert SafetyConfig.from_mhs(hc.mhs(env), limit_mode=limit_mode) == hc.safety_config(env.config)


def _cmd(action, ts, i):
    return Envelope(f"m{i}", ts, "awareness", "safety", i, "c", ActionProposal(
        f"p{i}", "t", tuple(float(a) for a in action), 1.0, Uncertainty()))


@pytest.mark.parametrize("env_kw", ENV_CONFIGS)
@pytest.mark.parametrize("limit_mode", ["clamp", "reject"])
def test_kernel_from_mhs_behaves_identically(tmp_path, env_kw, limit_mode):
    env = Puck2D(Puck2DConfig(**env_kw))
    hc = HarnessConfig(limit_mode=limit_mode, workspace_low=(-1.0, -1.0), workspace_high=(1.0, 1.0))
    traces = []
    for name in ("hand", "mhs"):
        with TelemetryLog(tmp_path / f"{name}.jsonl", clock=lambda: T0) as log:
            if name == "hand":
                k = SafetyKernel(hc.safety_config(env.config), log, KEY, clock=lambda: T0)
            else:
                k = SafetyKernel.from_mhs(hc.mhs(env), log, KEY, clock=lambda: T0, limit_mode=limit_mode)
            rng = np.random.default_rng(11)
            out = []
            for i in range(1500):
                pos = tuple(rng.uniform(-1.1, 1.1, 2))
                vel = tuple(rng.uniform(-4.0, 4.0, 2))
                r = k.check(_cmd(rng.uniform(-20.0, 20.0, 2), T0 - rng.uniform(0, 0.06), i),
                            position=pos, velocity=vel, now=T0)
                out.append((r.decision, r.actuator_command))
                obs = Observation("puck2d", ("pos_x", "pos_y", "vel_x", "vel_y"), pos + vel, Uncertainty())
                assert k.observed_kinematics(obs) == (pos, vel)
            out.append(k.tick(now=T0 + 5.0, velocity=(1.0, -2.0)).actuator_command)
            out.append(k.no_command(now=T0 + 5.0, velocity=(0.01, 0.0)).actuator_command)
            out.append(k.emergency_stop("t", now=T0 + 5.0, velocity=(-3.0, 0.0)).actuator_command)
            traces.append(out)
    assert traces[0] == traces[1]
    verdicts = {d.verdict for d, _ in traces[0][:-3]}
    assert {"approve", "reject"} <= verdicts


def test_kernel_takes_the_declared_actuator_latency_from_the_mhs(tmp_path):
    assert SafetyConfig.from_mhs(Puck2D().mhs()).actuator_latency_s == (0.0, 0.0)
    env = Puck2D(Puck2DConfig(actuator_tau=1.0))
    assert SafetyConfig.from_mhs(env.mhs()).actuator_latency_s == (1.0, 1.0)
    with TelemetryLog(tmp_path / "k.jsonl") as log:
        lagged = SafetyKernel.from_mhs(env.mhs(), log, KEY, clock=lambda: T0)
        prompt = SafetyKernel.from_mhs(Puck2D().mhs(), log, KEY, clock=lambda: T0)
        # Coasting at 0.5 m/s, 5 cm from x=2: stoppable at once, not after a 1 s latency.
        r = lagged.check(_cmd((0.0, 0.0), T0, 1), position=(1.95, 0.0), velocity=(0.5, 0.0), now=T0)
        assert r.decision.verdict == "reject" and "stopping[x]" in r.decision.violated_constraints
        r = prompt.check(_cmd((0.0, 0.0), T0, 2), position=(1.95, 0.0), velocity=(0.5, 0.0), now=T0)
        assert r.decision.verdict == "approve"
        with pytest.raises(ValueError):
            SafetyConfig(("x",), (-1.0,), (1.0,), (-1.0,), (1.0,), 0.05, 0.1, actuator_latency_s=(-0.1,))


def test_lagged_puck_approved_commands_stay_stoppable(tmp_path):
    # Empirical check of the latency model against Puck2D's first-order lag: from a
    # random state and random in-flight force, an approved command followed by the
    # braking safe action keeps the puck inside the workspace.
    tau, ws = 0.06, 1.0
    env = Puck2D(Puck2DConfig(actuator_tau=tau, goal=(9.0, 9.0), max_steps=10_000))
    hc = HarnessConfig(workspace_low=(-ws, -ws), workspace_high=(ws, ws))
    rng = np.random.default_rng(5)
    approved = 0
    with TelemetryLog(tmp_path / "k.jsonl", clock=lambda: T0) as log:
        k = SafetyKernel.from_mhs(hc.mhs(env), log, KEY, clock=lambda: T0)
        for i in range(400):
            pos = rng.uniform(-ws, ws, 2)
            vel = rng.uniform(-1.0, 1.0, 2)
            r = k.check(_cmd(rng.uniform(-10.0, 10.0, 2), T0, i),
                        position=tuple(pos), velocity=tuple(vel), now=T0)
            if r.decision.verdict != "approve":
                continue
            approved += 1
            env.reset(0)
            env.pos, env.vel = pos.copy(), vel.copy()
            env.force = rng.uniform(-10.0, 10.0, 2)  # unknown in-flight force
            command = r.actuator_command
            for _ in range(300):
                env.step(command)
                assert np.all(np.abs(env.pos) <= ws + 1e-9), (i, env.pos)
                command = k.tick(now=T0 + 1.0, velocity=tuple(env.vel)).actuator_command
    assert approved > 20


def test_kernel_reads_channels_declared_by_the_mhs(tmp_path):
    # No per-body code: renamed sensors are found through the MHS layout.
    m = Puck2D().mhs()
    renamed = {"pos_x": "px", "pos_y": "py", "vel_x": "vx", "vel_y": "vy"}
    m2 = dataclasses.replace(
        m,
        sensors=tuple(dataclasses.replace(s, name=renamed[s.name]) for s in m.sensors),
        observation_layout=tuple(renamed[c] for c in m.observation_layout),
    )
    with TelemetryLog(tmp_path / "k.jsonl") as log:
        k = SafetyKernel.from_mhs(m2, log, KEY, clock=lambda: T0)
        obs = Observation("s", ("vy", "px", "vx", "py"), (4.0, 1.0, 3.0, 2.0), Uncertainty())
        assert k.observed_kinematics(obs) == ((1.0, 2.0), (3.0, 4.0))
        old = Observation("s", ("pos_x", "pos_y", "vel_x", "vel_y"), (0.0,) * 4, Uncertainty())
        pos, vel = k.observed_kinematics(old)
        assert all(math.isnan(v) for v in pos + vel)
        dup = Observation("s", ("px", "px", "py", "vx", "vy"), (0.0, 1.0, 0.0, 0.0, 0.0), Uncertainty())
        assert all(math.isnan(v) for v in k.observed_kinematics(dup)[0])
        assert all(math.isnan(v) for v in k.observed_kinematics("garbage")[0])


# -- AC3: brain code is body-agnostic ---------------------------------------------------------


def puck2d_constants() -> set[str]:
    m = Puck2D().mhs()
    names = set(m.observation_layout) | set(m.action_layout) | {s.name for s in m.sensors}
    return names | {m.body_name, "Puck2D", "Puck2DConfig", "PUCK2D_VERSION", "CHANNELS"}


def body_constants_in(packages, src: Path = SRC) -> list[str]:
    """Puck2D layout names / identifiers in string literals, names or attributes."""
    forbidden = puck2d_constants()
    found = []
    for pkg in packages:
        for path in sorted((src / pkg).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    hits = [c for c in forbidden if c in node.value]
                elif isinstance(node, ast.Name):
                    hits = [node.id] if node.id in forbidden else []
                elif isinstance(node, ast.Attribute):
                    hits = [node.attr] if node.attr in forbidden else []
                elif isinstance(node, ast.alias):
                    hits = [n for n in (node.name, node.asname) if n in forbidden]
                else:
                    hits = []
                found += [f"{path.relative_to(src)}:{getattr(node, 'lineno', 0)}: {h}" for h in hits]
    return found


def test_no_puck2d_constants_in_brain_packages():
    for pkg in BRAIN_PACKAGES:
        assert (SRC / pkg / "__init__.py").is_file(), pkg
    assert body_constants_in(BRAIN_PACKAGES) == []


@pytest.mark.parametrize("source", [
    "CH = ('pos_x', 'pos_y')",
    "i = names.index('vel_y')",
    "f'{p}' == 'force_x'",
    "from simulation.puck2d import CHANNELS",
    "n = len(CHANNELS)",
    "import simulation.puck2d as p\nx = p.PUCK2D_VERSION",
])
def test_body_constant_detector_is_not_vacuous(tmp_path, source):
    (tmp_path / "system1").mkdir()
    (tmp_path / "system1" / "policy.py").write_text(source + "\n", encoding="utf-8")
    assert body_constants_in(["system1"], tmp_path)


class LayoutPolicy:
    """Body-agnostic System 1: everything about the body comes from the bound MHS."""

    version = "layout-policy-0"

    def __init__(self):
        self.mhs = None
        self.bound = 0

    def bind_mhs(self, mhs):
        self.mhs = mhs
        self.bound += 1

    def propose(self, state, prediction, rng):
        # Push half the limit along the first action component, using only MHS layouts.
        acts = self.mhs.action_actuators
        u = [0.0] * self.mhs.action_size
        u[0] = 0.5 * acts[0].high
        return ActionProposal("lp", self.version, tuple(u), 1.0, Uncertainty())


@pytest.mark.parametrize("env_kw", [{}, {"force_high": (4.0, 4.0)}])
def test_runner_hands_the_mhs_to_brain_modules(tmp_path, env_kw):
    env = Puck2D(Puck2DConfig(**env_kw))
    policy = LayoutPolicy()
    with TelemetryLog(tmp_path / "r.jsonl") as log:
        kernel = SafetyKernel.from_mhs(env.mhs(), log, KEY, clock=lambda: env.time)
        runner = CycleRunner(kernel, log, np.random.default_rng(0),
                             Modules(environment=env, system1=policy), clock=lambda: env.time)
        assert policy.bound == 1 and policy.mhs == env.mhs() == runner.mhs
        (res,) = runner.run(1)
    assert res.actuated_command == (0.5 * env.config.force_high[0], 0.0)


def test_runner_rejects_an_mhs_that_does_not_match_the_kernel(tmp_path):
    with TelemetryLog(tmp_path / "r.jsonl") as log:
        cfg = SafetyConfig(("x",), (-1.0,), (1.0,), (-1.0,), (1.0,), 0.05, 0.1)
        kernel = SafetyKernel(cfg, log, KEY, clock=lambda: 0.0)
        with pytest.raises(ValueError):
            CycleRunner(kernel, log, np.random.default_rng(0), Modules(environment=Puck2D()))
        with pytest.raises(TypeError):
            CycleRunner(kernel, log, np.random.default_rng(0), mhs={"body_name": "x"})
