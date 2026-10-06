"""Independent checks for the frozen evaluation-set contract."""

from simulation.evalset import build_evaluation_set


def test_evaluation_set_is_stable_after_construction():
    evaluation_set = build_evaluation_set(91, 2)
    fingerprint = evaluation_set.fingerprint()
    evaluation_set.params["n_tasks"] = 99
    assert evaluation_set.fingerprint() == fingerprint


def test_evaluation_set_round_trips_without_changing_tasks():
    evaluation_set = build_evaluation_set(92, 2)
    restored = type(evaluation_set).from_json(evaluation_set.to_json())
    assert restored.tasks == evaluation_set.tasks
    assert restored.fingerprint() == evaluation_set.fingerprint()
