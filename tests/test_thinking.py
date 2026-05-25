"""Tests for extended thinking workflow helpers."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.thinking import (
    REVISER_FORMAT_WARNING,
    Clarify,
    ReviewFinding,
    ReviewReport,
    Rewrite,
    ThinkingOutputError,
    append_tool_trace,
    extract_draft_for_user,
    parse_reviser_output,
    parse_structured_output,
    render_route_message,
    review_draft,
    rewrite_messages,
    rewrite_prompt,
    route_review_report,
    summarize_tool_trace,
    trim_head,
    trim_tail,
)


class _QueuedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.outputs.pop(0))


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


def _report_json(decision="pass", findings=None, summary="ok"):
    return json.dumps({
        "decision": decision,
        "findings": findings or [],
        "summary_for_reviser": summary,
    })


def test_parse_structured_output_accepts_json_fence():
    parsed = parse_structured_output(
        ReviewReport,
        f"```json\n{_report(decision='pass').model_dump_json()}\n```",
    )

    assert parsed.decision == "pass"


def test_parse_structured_output_rejects_invalid_json():
    with pytest.raises(ThinkingOutputError, match="invalid JSON"):
        parse_structured_output(ReviewReport, "not json")


def test_parse_structured_output_rejects_missing_required_fields():
    with pytest.raises(ThinkingOutputError, match="invalid ReviewReport"):
        parse_structured_output(ReviewReport, '{"decision": "pass"}')


@pytest.mark.parametrize(
    ("text", "trimmed"),
    [
        ("abcdef", "abcdef"),
        ("0123456789abcdefghijklmnopqrst", "... [truncated]\npqrst"),
    ],
)
def test_trim_tail_preserves_recent_context(text, trimmed):
    limit = len(text) + 1 if len(text) <= 6 else len(trimmed)

    assert trim_tail(text, limit) == trimmed


def test_trim_head_preserves_skill_header():
    assert (
        trim_head("0123456789abcdefghijklmnopqrst", len("0123\n... [truncated]"))
        == "0123\n... [truncated]"
    )


def test_rewrite_prompt_returns_rewritten_prompt_and_includes_context():
    model = _QueuedModel(["Rewrite this as a precise task."])

    result = rewrite_prompt(
        model,
        skill_text="prompt-master skill",
        user_input="raw request",
        visible_context="recent context",
        skill_context="active skill context",
    )

    assert isinstance(result, Rewrite)
    assert result.prompt == "Rewrite this as a precise task."
    prompt_text = "\n".join(message.content for message in model.calls[0])
    assert "prompt-master skill" in prompt_text
    assert "raw request" in prompt_text
    assert "recent context" in prompt_text
    assert "active skill context" in prompt_text
    assert "[Tool availability]" in prompt_text
    assert "你不得新增" in prompt_text


def test_rewrite_prompt_includes_runtime_tool_availability():
    model = _QueuedModel(["Rewrite this as a precise task."])
    tool_block = (
        "[Tool availability]\n"
        "active_skill: paper\n"
        "tool_policy_active: true\n"
        "available_tools: alpha_search\n"
        "denied_tools: shell_runner"
    )

    rewrite_prompt(
        model,
        skill_text="prompt-master skill",
        user_input="raw request",
        tool_availability=tool_block,
    )

    prompt_text = "\n".join(message.content for message in model.calls[0])
    assert tool_block in prompt_text
    assert "alpha_search" in prompt_text
    assert "shell_runner" in prompt_text


def test_rewrite_messages_do_not_embed_stale_tool_names():
    rewrite_messages(
        skill_text="prompt-master skill",
        user_input="raw request",
        visible_context="",
        skill_context="",
    )
    source = (Path(__file__).resolve().parents[1] / "agent" / "thinking.py").read_text(
        encoding="utf-8"
    )

    for name in (
        "rag_explore",
        "rag_search",
        "recall_history",
        "read_file",
        "bash",
        "web_search",
        "github",
    ):
        assert name not in source


def test_rewrite_prompt_detects_clarify_sentinel():
    model = _QueuedModel(["<<CLARIFY>>\n- Which journal?"])

    result = rewrite_prompt(model, skill_text="skill", user_input="revise")

    assert isinstance(result, Clarify)
    assert result.text == "- Which journal?"


def test_review_draft_invokes_model_with_evidence_and_rebuttal():
    model = _QueuedModel([_report_json("pass")])
    tool_block = (
        "[Tool availability]\n"
        "tool_policy_active: true\n"
        "available_tools: alpha_search"
    )

    report = review_draft(
        model,
        raw_user_input="raw",
        rewritten_prompt="rewritten",
        draft="draft",
        skill_context="skill ctx",
        evidence_trace_summary="[Writer] tool trace",
        previous_rebuttal="reasonable objection",
        tool_availability=tool_block,
    )

    assert report.decision == "pass"
    prompt_text = model.calls[0][-1].content
    assert "raw" in prompt_text
    assert "rewritten" in prompt_text
    assert tool_block in prompt_text
    assert "[Writer] tool trace" in prompt_text
    assert "reasonable objection" in prompt_text


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


def test_route_review_report_pass_overrides_attempt_cap():
    report = _report(decision="pass")

    assert route_review_report(report, attempts=2) == "pass"


def test_render_route_message_adds_warning_to_draft_routes():
    rendered = render_route_message(
        "pass",
        "Clean draft",
        _report(decision="pass"),
        format_warning="warning",
    )

    assert rendered == "warning\n\nClean draft"


def test_summarize_tool_trace_matches_tool_messages_and_truncates_result():
    trace = summarize_tool_trace(
        [{"id": "call-1", "name": "read_file", "args": {"path": "x.md"}}],
        [ToolMessage(content="abcdefghijklmnopqrstuvwxyz", tool_call_id="call-1")],
        source_label="[Writer]",
        per_result_chars=len("abc\n... [truncated]"),
    )

    assert "=== [Writer] ===" in trace
    assert "read_file" in trace
    assert '"path": "x.md"' in trace
    assert "abc\n" in trace
    assert "... [truncated]" in trace


def test_append_tool_trace_keeps_recent_evidence_under_cap():
    combined = append_tool_trace(
        "older evidence " * 20,
        [],
        [],
        source_label="[Reviser round 1]",
        total_chars_cap=80,
    )

    assert combined.startswith("... [older evidence truncated]")
    assert "[Reviser round 1]" in combined


def test_parse_reviser_output_splits_draft_and_rebuttal():
    parsed = parse_reviser_output(
        "DRAFT:\nClean answer\n\nREBUTTAL:\nI disagree with finding 1."
    )

    assert parsed.draft == "Clean answer"
    assert parsed.rebuttal == "I disagree with finding 1."


def test_parse_reviser_output_accepts_draft_only_marker():
    parsed = parse_reviser_output("DRAFT: Clean answer")

    assert parsed.draft == "Clean answer"
    assert parsed.rebuttal == ""


def test_parse_reviser_output_repairs_missing_markers_once():
    repair = _QueuedModel(["DRAFT:\nClean answer\n\nREBUTTAL:\n(none)"])

    parsed = parse_reviser_output("Clean answer\nInternal note", repair_model=repair)

    assert parsed.draft == "Clean answer"
    assert parsed.rebuttal == "(none)"
    assert len(repair.calls) == 1


def test_parse_reviser_output_heuristically_strips_internal_tail():
    repair = _QueuedModel(["still unmarked"])

    parsed = parse_reviser_output(
        "Clean answer paragraph with enough content to keep.\n\nREBUTTAL:\n(none)",
        repair_model=repair,
    )

    assert parsed.draft == "Clean answer paragraph with enough content to keep."
    assert "(none)" in parsed.rebuttal
    assert parsed.format_warning == ""


def test_parse_reviser_output_final_fallback_warns_when_unsafe_to_strip():
    repair = _QueuedModel(["still unmarked"])

    parsed = parse_reviser_output("Clean answer without markers", repair_model=repair)

    assert parsed.draft == "Clean answer without markers"
    assert parsed.format_warning == REVISER_FORMAT_WARNING


def test_extract_draft_for_user_uses_marker_when_present():
    assert extract_draft_for_user("DRAFT:\nVisible\n\nREBUTTAL:\nHidden") == "Visible"
