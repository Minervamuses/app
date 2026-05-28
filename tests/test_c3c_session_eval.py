"""Tests for C3c skill-validation session integration evaluator."""

from agent.evaluation.claims.c3c_session import C3SessionEvaluator
from agent.evaluation.datasets import load_claim_dataset


def _session_cases(split="dev"):
    return [
        case for case in load_claim_dataset("c3", split)
        if case.raw.get("task") == "session"
    ]


def test_c3c_dataset_loads_normal_and_extended_cases():
    cases = _session_cases()

    assert [case.id for case in cases] == [
        "c3c-normal-validator-retry",
        "c3c-extended-final-validation-retry",
    ]


def test_c3_session_evaluator_observes_normal_and_extended_retries(tmp_path):
    result = C3SessionEvaluator().evaluate(_session_cases())

    assert result.name == "C3Session"
    assert result.total == 2
    assert result.scores == {
        "retry_accuracy": 1.0,
        "final_clean_accuracy": 1.0,
    }
    by_id = {detail["id"]: detail for detail in result.details}
    assert by_id["c3c-normal-validator-retry"]["prediction"]["validation_attempts"] == 1
    assert by_id["c3c-normal-validator-retry"]["prediction"]["model_invoke_count"] == 2
    assert by_id["c3c-extended-final-validation-retry"]["prediction"]["validation_attempts"] == 1
    assert by_id["c3c-extended-final-validation-retry"]["prediction"]["model_invoke_count"] == 1


def test_c3_session_evaluator_handles_no_retry_test_case():
    result = C3SessionEvaluator().evaluate(_session_cases("test"))

    assert result.total == 1
    assert result.details[0]["prediction"]["retry_observed"] is False
    assert result.details[0]["scores"]["passed"] is True
