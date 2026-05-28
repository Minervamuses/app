"""Tests for the C3b reviewer classifier evaluator."""

import json

import pytest
from langchain_core.messages import AIMessage

from agent.evaluation.claims.c3b_reviewer import C3ReviewerEvaluator, score_review_report
from agent.evaluation.datasets import load_claim_dataset
from agent.thinking import ReviewReport


class _QueuedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def invoke(self, _messages):
        return AIMessage(content=self.outputs.pop(0))


def _report(decision, *, severity=None, failure_mode=None, needs_user_input=False):
    findings = []
    if severity:
        findings.append({
            "severity": severity,
            "dimension": "routing",
            "location": "draft",
            "problem": "problem",
            "evidence_from_draft": "evidence",
            "revision_instruction": "instruction",
            "needs_user_input": needs_user_input,
            "failure_mode": failure_mode,
        })
    return json.dumps({
        "decision": decision,
        "findings": findings,
        "summary_for_reviser": "summary",
    })


def test_c3b_dataset_loads_reviewer_cases():
    cases = [
        case for case in load_claim_dataset("c3", "dev")
        if case.raw.get("task") == "reviewer"
    ]

    assert [case.id for case in cases] == [
        "c3b-retrieval-not-attempted",
        "c3b-tool-unavailable",
        "c3b-clean-draft-pass",
    ]


def test_score_review_report_derives_route_and_classifier_checks():
    report = ReviewReport.model_validate_json(
        _report(
            "revise",
            severity="major",
            failure_mode="retrieval_not_attempted",
            needs_user_input=False,
        )
    )
    gold = {
        "decision": "revise",
        "min_severity": "major",
        "failure_mode": "retrieval_not_attempted",
        "route": "revise",
        "needs_user_input": False,
    }

    scores = score_review_report(report, gold)

    assert scores["prediction"]["route"] == "revise"
    assert scores["checks"] == {
        "decision_ok": True,
        "route_ok": True,
        "failure_modes_ok": True,
        "needs_user_input_ok": True,
        "severity_ok": True,
    }


def test_c3_reviewer_evaluator_reports_macro_metrics():
    model = _QueuedModel([
        _report(
            "revise",
            severity="major",
            failure_mode="retrieval_not_attempted",
            needs_user_input=False,
        ),
        _report(
            "block",
            severity="blocker",
            failure_mode="tool_unavailable",
            needs_user_input=True,
        ),
        _report("pass"),
    ])
    cases = [
        case for case in load_claim_dataset("c3", "dev")
        if case.raw.get("task") == "reviewer"
    ]

    result = C3ReviewerEvaluator(model=model).evaluate(cases)

    assert result.name == "C3Reviewer"
    assert result.total == 3
    assert result.scores["decision_macro_f1"] == 1.0
    assert result.scores["route_macro_f1"] == 1.0
    assert result.scores["failure_mode_macro_f1"] == 1.0
    assert result.scores["needs_user_input_macro_f1"] == 1.0
    assert result.scores["severity_macro_f1"] == 1.0
    assert result.scores["parse_success_rate"] == 1.0


def test_c3_reviewer_evaluator_counts_parse_failures_as_misses():
    model = _QueuedModel(["not json"])
    case = next(
        case for case in load_claim_dataset("c3", "dev")
        if case.id == "c3b-retrieval-not-attempted"
    )

    result = C3ReviewerEvaluator(model=model).evaluate([case])

    assert result.scores["parse_success_rate"] == 0.0
    assert result.details[0]["error"].startswith("ThinkingOutputError")
    assert result.scores["decision_macro_f1"] == 0.0
