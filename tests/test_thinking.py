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
    render_review_stop_message,
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


def test_review_finding_failure_mode_is_optional_and_parsed():
    parsed = parse_structured_output(
        ReviewReport,
        json.dumps({
            "decision": "revise",
            "findings": [
                {
                    "severity": "major",
                    "dimension": "instruction following",
                    "location": "whole draft",
                    "problem": "retrieval was skipped",
                    "evidence_from_draft": "asks user to restate context",
                    "revision_instruction": "Use the available history tool first.",
                    "needs_user_input": True,
                    "failure_mode": "retrieval_not_attempted",
                }
            ],
            "summary_for_reviser": "retrieve first",
        }),
    )

    assert parsed.findings[0].failure_mode == "retrieval_not_attempted"


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


def test_rewrite_messages_fallback_renders_base_tool_availability():
    from agent.tools.inventory import base_tool_names

    messages = rewrite_messages(
        skill_text="prompt-master skill",
        user_input="raw request",
        visible_context="",
        skill_context="",
    )
    prompt_text = "\n".join(str(message.content) for message in messages)

    assert "[Tool availability]" in prompt_text
    for name in base_tool_names():
        assert name in prompt_text


def test_review_messages_fallback_renders_base_tool_availability():
    from agent.thinking import review_messages
    from agent.tools.inventory import base_tool_names

    messages = review_messages(
        raw_user_input="raw",
        rewritten_prompt="rewritten",
        draft="draft",
        skill_context="",
        evidence_trace_summary="",
        previous_rebuttal="",
    )
    prompt_text = "\n".join(str(message.content) for message in messages)

    assert "[Tool availability]" in prompt_text
    for name in base_tool_names():
        assert name in prompt_text


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


def test_review_prompt_includes_retrieval_failure_routing_contract():
    model = _QueuedModel([_report_json("pass")])

    review_draft(
        model,
        raw_user_input="請看我之前的紀錄",
        rewritten_prompt="Use prior chat history.",
        draft="I need your full research background.",
        evidence_trace_summary="=== [Writer] ===\nTool calls: none",
        tool_availability=(
            "[Tool availability]\n"
            "tool_policy_active: true\n"
            "available_tools: recall_history\n"
            "denied_tools: (none)"
        ),
    )

    prompt_text = model.calls[0][-1].content
    assert "Finding routing contract" in prompt_text
    assert "severity=major" in prompt_text
    assert "needs_user_input=false" in prompt_text
    assert "decision=revise" in prompt_text
    assert "failure_mode" in prompt_text
    assert "retrieval_not_attempted" in prompt_text
    assert "result is empty" in prompt_text
    assert "tool policy/settings problem" in prompt_text
    assert "Never allow fabricated scholarly content" in prompt_text


def test_history_retrieval_gap_routes_to_revise_not_ask_user():
    finding = _finding(
        severity="major",
        problem="Writer did not call the available history tool before asking the user.",
        revision_instruction=(
            "Call recall_history with a January research progress query before asking "
            "the user for all background again."
        ),
        needs_user_input=False,
    )
    model = _QueuedModel([
        _report_json("revise", [finding.model_dump()], "Use history retrieval first.")
    ])

    report = review_draft(
        model,
        raw_user_input="我一月上半做了什麼？你自行看一下紀錄。",
        rewritten_prompt="Answer from prior chat history.",
        draft="請提供完整研究背景。",
        evidence_trace_summary="=== [Writer] ===\nTool calls: none",
        tool_availability=(
            "[Tool availability]\n"
            "tool_policy_active: true\n"
            "available_tools: recall_history\n"
            "denied_tools: (none)"
        ),
    )

    assert report.decision == "revise"
    assert report.findings[0].severity == "major"
    assert report.findings[0].needs_user_input is False
    assert route_review_report(report, attempts=0) == "revise"


def test_retrieval_failure_mode_overrides_bad_user_input_flag():
    report = _report(
        _finding(
            severity="major",
            needs_user_input=True,
            failure_mode="retrieval_not_attempted",
        ),
        decision="block",
    )

    assert route_review_report(report, attempts=0) == "revise"


def test_retrieval_empty_failure_mode_does_not_ask_user():
    report = _report(
        _finding(
            severity="minor",
            needs_user_input=True,
            failure_mode="retrieval_empty",
        ),
        decision="block",
    )

    assert route_review_report(report, attempts=0) == "revise"


def test_tool_unavailable_failure_mode_still_asks_user():
    report = _report(
        _finding(
            severity="major",
            needs_user_input=False,
            failure_mode="tool_unavailable",
        ),
        decision="revise",
    )

    assert route_review_report(report, attempts=0) == "ask_user"


def test_review_stop_message_sanitizes_internal_reviser_instruction():
    report = _report(
        _finding(
            severity="blocker",
            needs_user_input=True,
            revision_instruction=(
                "reviser 應回到 academic-paper-writing 技能的 Intake checklist"
            ),
        ),
        decision="block",
    )

    rendered = render_review_stop_message(report)

    assert "reviser" not in rendered
    assert "Intake checklist" not in rendered
    assert "需要更多資訊" in rendered


def test_review_stop_message_preserves_user_readable_question():
    report = _report(
        _finding(
            severity="blocker",
            needs_user_input=True,
            revision_instruction="請提供研究資料所在的檔案或資料夾名稱。",
        ),
        decision="block",
    )

    rendered = render_review_stop_message(report)

    assert "請提供研究資料所在的檔案或資料夾名稱。" in rendered


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


# --- Fusion aggregator + evidence -----------------------------------------

from agent.thinking import (  # noqa: E402
    FusionAggregateResult,
    FusionCandidate,
    FusionCandidateTrace,
    FusionTurnMetadata,
    aggregate_candidates,
    aggregate_messages,
    build_fusion_evidence_summary,
    parse_aggregate_result,
)


def _candidate(candidate_id, model_id, answer, *, status="success", error="", summary="Tool calls: none"):
    return FusionCandidate(
        candidate_id=candidate_id,
        model_id=model_id,
        status=status,
        answer=answer,
        tool_trace_summary=summary,
        error=error,
    )


def _aggregate_json(draft="fused", selected=None, dropped=None, summary="merged", removed=None):
    return json.dumps({
        "draft": draft,
        "selected_candidate_ids": selected if selected is not None else ["candidate-1"],
        "dropped_candidate_ids": dropped or [],
        "summary_for_reviewer": summary,
        "removed_or_uncertain_points": removed or [],
    })


def test_parse_aggregate_result_accepts_valid_json():
    result = parse_aggregate_result(
        _aggregate_json(draft="fused draft", selected=["candidate-1"], dropped=["candidate-2"]),
        successful_candidate_ids=["candidate-1", "candidate-2"],
    )

    assert isinstance(result, FusionAggregateResult)
    assert result.draft == "fused draft"
    assert result.selected_candidate_ids == ["candidate-1"]
    assert result.dropped_candidate_ids == ["candidate-2"]
    assert result.summary_for_reviewer == "merged"


def test_parse_aggregate_result_rejects_invalid_json():
    with pytest.raises(ThinkingOutputError, match="invalid JSON from aggregator"):
        parse_aggregate_result("not json", successful_candidate_ids=["candidate-1"])


def test_parse_aggregate_result_rejects_blank_draft():
    with pytest.raises(ThinkingOutputError, match="blank draft"):
        parse_aggregate_result(
            _aggregate_json(draft="   ", selected=[]),
            successful_candidate_ids=["candidate-1"],
        )


def test_parse_aggregate_result_rejects_unknown_candidate_id():
    with pytest.raises(ThinkingOutputError, match="unknown candidate ids"):
        parse_aggregate_result(
            _aggregate_json(selected=["candidate-9"]),
            successful_candidate_ids=["candidate-1"],
        )


def test_parse_aggregate_result_rejects_selected_dropped_overlap():
    with pytest.raises(ThinkingOutputError, match="selected and dropped"):
        parse_aggregate_result(
            _aggregate_json(selected=["candidate-1"], dropped=["candidate-1"]),
            successful_candidate_ids=["candidate-1"],
        )


def test_aggregate_messages_include_inputs_and_candidate_identity():
    candidates = [
        _candidate("candidate-1", "model-a", "Answer one"),
        _candidate("candidate-2", "model-b", "Answer two"),
    ]
    messages = aggregate_messages(
        raw_user_input="raw question",
        rewritten_prompt="rewritten task",
        successful_candidates=candidates,
        skill_context="skill ctx",
        tool_availability="[Tool availability]\navailable_tools: read_file",
    )
    prompt_text = "\n".join(str(m.content) for m in messages)

    assert "raw question" in prompt_text
    assert "rewritten task" in prompt_text
    assert "Answer one" in prompt_text
    assert "Answer two" in prompt_text
    assert "candidate-1" in prompt_text
    assert "candidate-2" in prompt_text
    assert "model-a" in prompt_text
    assert "model-b" in prompt_text
    assert "[Tool availability]" in prompt_text
    assert "skill ctx" in prompt_text


def test_aggregate_messages_do_not_describe_call_as_tool_evidence():
    messages = aggregate_messages(
        raw_user_input="raw",
        rewritten_prompt="rewritten",
        successful_candidates=[_candidate("candidate-1", "model-a", "answer")],
    )
    system_text = str(messages[0].content)

    assert "not calling a tool" in system_text
    assert "must not invent a tool" in system_text


def test_aggregate_candidates_invokes_model_and_parses():
    model = _QueuedModel([_aggregate_json(draft="merged draft", selected=["candidate-1"])])
    result = aggregate_candidates(
        model,
        raw_user_input="raw",
        rewritten_prompt="rewritten",
        successful_candidates=[_candidate("candidate-1", "model-a", "answer")],
    )

    assert result.draft == "merged draft"
    assert result.selected_candidate_ids == ["candidate-1"]


def _trace(candidate_id, model_id, *, call_id, args, result, answer, status="success"):
    return FusionCandidateTrace(
        candidate_id=candidate_id,
        model_id=model_id,
        status=status,
        new_messages=[ToolMessage(content=result, tool_call_id=call_id)],
        tool_calls=[{"id": call_id, "name": "read_file", "args": args}],
        trace_events=[],
        answer_excerpt=answer,
    )


def test_build_fusion_evidence_summary_includes_tier_and_ids():
    candidates = [
        _candidate("candidate-1", "model-a", "Answer one"),
        _candidate("candidate-2", "model-b", "", status="failed", error="boom"),
    ]
    aggregate = FusionAggregateResult(
        draft="fused",
        selected_candidate_ids=["candidate-1"],
        dropped_candidate_ids=[],
        reliability_tier="partial_panel",
        summary_for_reviewer="kept candidate-1",
        removed_or_uncertain_points=["dropped weak claim"],
    )
    metadata = FusionTurnMetadata(
        selected_ids=["candidate-1"],
        dropped_ids=[],
        omitted_successful_ids=[],
        reliability_tier="partial_panel",
        aggregator_error="",
    )

    summary = build_fusion_evidence_summary(
        candidates=candidates,
        candidate_traces=[],
        aggregate_result=aggregate,
        metadata=metadata,
    )

    assert "reliability_tier: partial_panel" in summary
    assert "selected_candidate_ids: candidate-1" in summary
    assert "kept candidate-1" in summary
    assert "dropped weak claim" in summary
    assert "candidate-1 (model: model-a) status=success" in summary
    assert "candidate-2 (model: model-b) status=failed" in summary
    assert "error: boom" in summary


def test_fusion_evidence_summary_does_not_cross_pair_colliding_call_ids():
    traces = [
        _trace("candidate-1", "model-a", call_id="call-1", args={"path": "a.md"},
               result="RESULT-A", answer="answer A"),
        _trace("candidate-2", "model-b", call_id="call-1", args={"path": "b.md"},
               result="RESULT-B", answer="answer B"),
    ]
    candidates = [
        _candidate("candidate-1", "model-a", "answer A"),
        _candidate("candidate-2", "model-b", "answer B"),
    ]
    metadata = FusionTurnMetadata(reliability_tier="full_panel")

    summary = build_fusion_evidence_summary(
        candidates=candidates,
        candidate_traces=traces,
        aggregate_result=FusionAggregateResult(draft="fused"),
        metadata=metadata,
    )

    seg1 = summary.split("candidate-2 (model: model-b)")[0]
    seg2 = summary.split("candidate-2 (model: model-b)")[1]
    assert "RESULT-A" in seg1 and "RESULT-B" not in seg1
    assert "RESULT-B" in seg2 and "RESULT-A" not in seg2
