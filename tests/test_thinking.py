"""Tests for extended thinking workflow helpers."""

import pytest
from langchain_core.messages import AIMessage

from agent.thinking import (
    ReviewFinding,
    ReviewReport,
    TaskSpec,
    ThinkingOutputError,
    append_assumption_note,
    compile_task_spec,
    parse_structured_output,
    route_review_report,
    route_task_spec,
)


def _task_spec(**overrides):
    data = {
        "task": "revise abstract",
        "task_type": "academic_revision",
        "target_output": "polished abstract",
        "confidence": "high",
        "decision": "proceed",
        "known_from_user": ["draft supplied"],
        "known_from_context": [],
        "allowed_assumptions": [],
        "forbidden_assumptions": ["do not add data"],
        "missing_info": [],
        "constraints": ["preserve evidence"],
        "success_criteria": ["no fabricated citations"],
        "writer_instruction": "Draft the answer.",
        "reviewer_instruction": "Check claim-evidence alignment.",
    }
    data.update(overrides)
    return TaskSpec.model_validate(data)


def _finding(**overrides):
    data = {
        "severity": "major",
        "dimension": "claim-evidence alignment",
        "location": "paragraph 1",
        "problem": "claim outruns evidence",
        "evidence_from_draft": "unsupported claim",
        "revision_instruction": "Soften the claim.",
        "needs_user_input": False,
    }
    data.update(overrides)
    return ReviewFinding.model_validate(data)


def _report(*findings, decision="revise"):
    return ReviewReport(
        decision=decision,
        findings=list(findings),
        summary_for_reviser="fix major issues",
    )


def test_parse_structured_output_accepts_json_fence():
    parsed = parse_structured_output(
        TaskSpec,
        f"```json\n{_task_spec().model_dump_json()}\n```",
    )

    assert parsed.task == "revise abstract"


def test_parse_structured_output_rejects_invalid_json():
    with pytest.raises(ThinkingOutputError, match="invalid JSON"):
        parse_structured_output(TaskSpec, "not json")


def test_parse_structured_output_rejects_missing_required_fields():
    with pytest.raises(ThinkingOutputError, match="invalid TaskSpec"):
        parse_structured_output(TaskSpec, '{"task": "too little"}')


@pytest.mark.parametrize(
    ("decision", "route"),
    [
        ("proceed", "write"),
        ("proceed_with_assumptions", "write"),
        ("need_clarification", "clarify"),
        ("block", "block"),
    ],
)
def test_route_task_spec_maps_all_decisions(decision, route):
    assert route_task_spec(_task_spec(decision=decision)) == route


def test_route_review_report_passes_minor_and_notes_without_rewrite():
    report = _report(
        _finding(severity="minor"),
        _finding(severity="note"),
        decision="revise",
    )

    assert route_review_report(report, attempts=0) == "pass"


def test_route_review_report_sends_major_to_reviser_before_cap():
    assert route_review_report(_report(_finding()), attempts=1) == "revise"


def test_route_review_report_stops_at_attempt_cap():
    assert route_review_report(_report(_finding()), attempts=2) == "stop"


def test_route_review_report_blocks_reviser_for_user_input():
    report = _report(_finding(needs_user_input=True))

    assert route_review_report(report, attempts=0) == "ask_user"


def test_route_review_report_blocks_reviser_for_blocker():
    report = _report(_finding(severity="blocker"), decision="block")

    assert route_review_report(report, attempts=0) == "ask_user"


def test_append_assumption_note_only_for_assumption_route():
    spec = _task_spec(
        decision="proceed_with_assumptions",
        allowed_assumptions=["use journal-neutral tone"],
    )

    assert "採用的假設" in append_assumption_note("Answer", spec)


def test_compile_task_spec_invokes_model_and_parses_json():
    class FakeModel:
        def invoke(self, messages):
            assert "TaskSpec schema fields" in messages[-1].content
            return AIMessage(content=_task_spec().model_dump_json())

    parsed = compile_task_spec(FakeModel(), user_input="revise this")

    assert parsed.decision == "proceed"
