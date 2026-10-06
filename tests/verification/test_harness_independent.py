"""Independent checks for evaluation-set freezing and runner integration."""

import pytest

from simulation.harness import EvalSet, EvalSetSpec, build_eval_set


def test_eval_set_nested_environment_parameters_are_detached_and_read_only():
    supplied = {"max_steps": 20, "force_high": [5.0, 6.0]}
    eval_set = build_eval_set(
        13,
        EvalSetSpec(n_tasks=1, max_obstacles=0, max_impulses=0),
        env_params=supplied,
    )

    supplied["force_high"][0] = 99.0
    assert eval_set.env_params["force_high"] == (5.0, 6.0)
    with pytest.raises(TypeError):
        eval_set.env_params["force_high"][0] = 99.0

    restored = EvalSet.from_json(eval_set.to_json())
    assert restored == eval_set
    assert restored.to_json() == eval_set.to_json()
