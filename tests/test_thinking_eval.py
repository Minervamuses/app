"""Tests for thinking reviewer evaluation cases and scoring."""

from langchain_core.messages import AIMessage

from agent.evaluation.thinking import ThinkingReviewerEvaluator


class _QueuedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def invoke(self, _messages):
        return AIMessage(content=self.outputs.pop(0))


def _report_json(severity, needs_user_input):
    import json

    return json.dumps({
        "decision": "block" if severity == "blocker" else "revise",
        "findings": [
            {
                "severity": severity,
                "dimension": "citation integrity",
                "location": "draft",
                "problem": "high-risk scholarly claim",
                "evidence_from_draft": "unsupported citation or claim",
                "revision_instruction": "Ask the user for a source.",
                "needs_user_input": needs_user_input,
            }
        ],
        "summary_for_reviser": "reviewed",
    })


def test_thinking_reviewer_generate_has_major_and_integrity_cases(tmp_path):
    evaluator = ThinkingReviewerEvaluator(model=_QueuedModel([]))
    cases = evaluator.generate(output_path=str(tmp_path / "thinking_cases.json"))

    assert [case["id"] for case in cases] == [
        "major_claim_evidence_alignment",
        "academic_integrity_missing_source",
    ]
    assert (tmp_path / "thinking_cases.json").exists()


def test_thinking_reviewer_evaluate_scores_expected_findings():
    evaluator = ThinkingReviewerEvaluator(model=_QueuedModel([
        _report_json("major", False),
        _report_json("blocker", True),
    ]))

    result = evaluator.evaluate(evaluator.generate())

    assert result.total == 2
    assert result.scores == {
        "reviewer_detection_accuracy": 1.0,
        "severity_accuracy": 1.0,
        "user_input_accuracy": 1.0,
    }
    assert all(detail["scores"]["passed"] for detail in result.details)
