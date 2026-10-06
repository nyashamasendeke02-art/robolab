"""Independent checks for evaluation-set immutability and episode seeding."""

import pytest

from simulation.harness import EvalSet, EvalSetSpec, build_eval_set


def test_eval_set_env_params_are_immutable_after_construction():
    """A frozen evaluation set must not permit silent physics changes in memory."""
    eval_set = build_eval_set(
        17,
        EvalSetSpec(n_tasks=1, max_obstacles=0, max_impulses=0),
        env_params={"friction": 0.05, "max_steps": 20},
    )

    with pytest.raises(TypeError):
        eval_set.env_params["friction"] = 0.9

    restored = EvalSet.from_json(eval_set.to_json())
    assert restored.env_params["friction"] == 0.05


def test_eval_set_json_round_trip_retains_task_inputs():
    eval_set = build_eval_set(29, EvalSetSpec(n_tasks=2, max_obstacles=1))
    restored = EvalSet.from_json(eval_set.to_json())

    assert restored == eval_set
    assert restored.to_dict() == eval_set.to_dict()
