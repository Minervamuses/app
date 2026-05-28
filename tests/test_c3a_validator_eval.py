"""Tests for the C3a deterministic skill-validator evaluator."""

from agent.evaluation.claims.c3a_validator import (
    C3ValidatorEvaluator,
    score_validator_predictions,
)
from agent.evaluation.datasets import load_claim_dataset


def test_c3a_dataset_loads_validator_cases():
    cases = [
        case for case in load_claim_dataset("c3", "dev")
        if case.raw.get("task") == "validator"
    ]

    assert [case.id for case in cases] == [
        "c3a-academic-uncited-percent",
        "c3a-academic-cited-percent",
        "c3a-unregistered-skill",
    ]


def test_score_validator_predictions_counts_binary_outcomes():
    violation = "needs citation"

    assert score_validator_predictions([violation], [violation]) == {
        "exact_match": True,
        "true_positive": True,
        "false_positive": False,
        "false_negative": False,
        "true_negative": False,
    }
    assert score_validator_predictions([violation], [])["false_positive"] is True
    assert score_validator_predictions([], [violation])["false_negative"] is True
    assert score_validator_predictions([], [])["true_negative"] is True


def test_c3_validator_evaluator_scores_deterministically():
    cases = [
        case for case in load_claim_dataset("c3", "dev")
        if case.raw.get("task") == "validator"
    ]

    result = C3ValidatorEvaluator().evaluate(cases)

    assert result.name == "C3Validator"
    assert result.total == 3
    assert result.scores == {
        "violation_precision": 1.0,
        "violation_recall": 1.0,
        "violation_f1": 1.0,
        "exact_match": 1.0,
        "false_positive_rate": 0.0,
    }
    assert all(detail["scores"]["exact_match"] for detail in result.details)
