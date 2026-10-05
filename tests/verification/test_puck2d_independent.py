"""Independent behavioral checks for the Puck2D simulator."""

import numpy as np

from simulation.puck2d import Impulse, MassChange, Puck2D, Puck2DConfig


def test_seeded_reset_and_step_replay_are_identical():
    config = Puck2DConfig(
        start_pos_std=0.2,
        pos_noise_std=0.03,
        vel_noise_std=0.04,
        impulses=(Impulse(2, (0.5, -0.25)),),
        mass_changes=(MassChange(4, 2.0),),
    )
    actions = [(0.5, -0.2)] * 8
    trajectories = []
    for _ in range(2):
        env = Puck2D(config, seed=123)
        rows = [env.reset(123)[0].tobytes()]
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append((obs.tobytes(), reward, terminated, truncated,
                         tuple(info["pos"]), tuple(info["vel"]),
                         tuple(d["type"] for d in info["active_disturbances"])))
        trajectories.append(rows)
    assert trajectories[0] == trajectories[1]


def test_impulse_and_mass_schedule_match_transition_index_and_ground_truth():
    env = Puck2D(Puck2DConfig(
        mass=1.0,
        force_low=(-10.0, -10.0),
        force_high=(10.0, 10.0),
        impulses=(Impulse(1, (2.0, 0.0)),),
        mass_changes=(MassChange(2, 4.0),),
        goal=(100.0, 100.0),
        max_steps=8,
    ), seed=8)

    expected = [0.0, 2.0, 2.0]
    masses = [1.0, 1.0, 4.0]
    for index in range(3):
        _, _, _, _, info = env.step((0.0, 0.0))
        assert info["step"] == index + 1
        assert info["mass"] == masses[index]
        np.testing.assert_allclose(info["vel"], (expected[index], 0.0))
        types = [entry["type"] for entry in info["active_disturbances"]]
        assert ("impulse" in types) == (index == 1)
        assert ("mass_change" in types) == (index == 2)
        np.testing.assert_array_equal(env.pos, info["pos"])
        np.testing.assert_array_equal(env.vel, info["vel"])
