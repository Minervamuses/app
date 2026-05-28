"""Tests for C4 deterministic checklist evaluator."""

from agent.evaluation.claims.c4_endtoend import C4ChecklistEvaluator, score_c4_checklist
from agent.evaluation.datasets import load_claim_dataset


def test_c4_dataset_loads_checklist_cases():
    cases = load_claim_dataset("c4", "dev")

    assert [case.id for case in cases] == [
        "c4-local-file-summary",
        "c4-history-codename-answer",
    ]


def test_score_c4_checklist_checks_tools_and_answer_text():
    scores = score_c4_checklist(
        "Phase 1 and Phase 2 cover C1/C2.",
        [{"name": "read_file", "args": {}}],
        {
            "required_tools": ["read_file"],
            "forbidden_tools": ["full-web-search"],
            "answer_contains": ["Phase 1", "Phase 2"],
            "answer_regex": ["C1/C2"],
        },
    )

    assert scores["passed"] is True
    assert score_c4_checklist(
        "No local details.",
        [{"name": "full-web-search", "args": {}}],
        {
            "required_tools": ["read_file"],
            "forbidden_tools": ["full-web-search"],
            "answer_contains": ["Phase 1"],
        },
    )["passed"] is False


def test_c4_evaluator_scores_injected_turn_runner():
    cases = load_claim_dataset("c4", "dev")

    def turn_runner(messages):
        if messages[0].startswith("Read EVALUATOR_PLAN"):
            return (
                "Phase 1 builds the base; Phase 2 evaluates C1; Phase 3 evaluates C2; C3 and C4 follow.",
                [{"name": "read_file", "args": {"path": "EVALUATOR_PLAN.md"}}],
            )
        return (
            "The deployment codename was Blue Lantern.",
            [{"name": "recall_history", "args": {"query": "deployment codename"}}],
        )

    result = C4ChecklistEvaluator(turn_runner=turn_runner).evaluate(cases)

    assert result.name == "C4Checklist"
    assert result.total == 2
    assert result.scores == {
        "task_success_rate": 1.0,
        "required_tools_accuracy": 1.0,
        "answer_requirements_accuracy": 1.0,
    }
    assert all(detail["scores"]["passed"] for detail in result.details)
