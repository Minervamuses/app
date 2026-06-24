"""Structured helpers for the optional extended thinking workflow."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from agent.skills.runtime import render_tool_availability_block


FusionCandidateStatus = Literal["success", "failed", "timeout", "empty"]
FusionReliabilityTier = Literal[
    "full_panel",
    "partial_panel",
    "single_candidate",
    "fallback",
]

ReviewSeverity = Literal["blocker", "major", "minor", "note"]
ReviewDecision = Literal["pass", "revise", "block"]
ReviewRoute = Literal["pass", "revise", "ask_user", "stop"]
ReviewFailureMode = Literal[
    "retrieval_not_attempted",
    "retrieval_empty",
    "tool_unavailable",
    "user_input_missing",
    "fabrication_risk",
]

MAX_REVIEW_ATTEMPTS = 2
REVISER_FORMAT_WARNING = (
    "（注意：本次回應的 reviser 輸出格式異常，可能混入內部審稿討論，請斟酌使用。）"
)

logger = logging.getLogger(__name__)


class ThinkingOutputError(ValueError):
    """Raised when an extended-thinking LLM step returns invalid structured output."""


class Clarify(BaseModel):
    """Prompt rewrite result asking the user for missing information."""

    text: str


class Rewrite(BaseModel):
    """Prompt rewrite result containing a clarified agent prompt."""

    prompt: str


RewriteResult = Clarify | Rewrite


class ReviewFinding(BaseModel):
    severity: ReviewSeverity
    dimension: str
    location: str
    problem: str
    evidence_from_draft: str
    revision_instruction: str
    needs_user_input: bool
    failure_mode: ReviewFailureMode | None = None


class ReviewReport(BaseModel):
    decision: ReviewDecision
    findings: list[ReviewFinding] = Field(default_factory=list)
    summary_for_reviser: str


class RevisedDraft(BaseModel):
    draft: str
    rebuttal: str = ""
    format_warning: str = ""


@dataclass(frozen=True)
class FusionCandidate:
    """One proposer graph result in the extended-thinking fusion panel."""

    candidate_id: str
    model_id: str
    status: FusionCandidateStatus
    answer: str
    tool_trace_summary: str
    error: str = ""
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class FusionCandidateTrace:
    """Per-candidate segmented trace; kept separate so tool_call_ids never cross."""

    candidate_id: str
    model_id: str
    status: FusionCandidateStatus
    new_messages: list = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    trace_events: list[dict] = field(default_factory=list)
    tool_trace_summary: str = ""
    answer_excerpt: str = ""


@dataclass
class FusionAggregateResult:
    """Session-level aggregate result over the successful candidate panel."""

    draft: str
    selected_candidate_ids: list[str] = field(default_factory=list)
    dropped_candidate_ids: list[str] = field(default_factory=list)
    reliability_tier: FusionReliabilityTier | str = ""
    summary_for_reviewer: str = ""
    removed_or_uncertain_points: list[str] = field(default_factory=list)
    aggregator_error: str = ""


@dataclass
class FusionTurnMetadata:
    """Compact, JSON-serializable fusion metadata for one extended turn."""

    candidate_statuses: dict[str, str] = field(default_factory=dict)
    model_ids: dict[str, str] = field(default_factory=dict)
    selected_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    omitted_successful_ids: list[str] = field(default_factory=list)
    reliability_tier: str = ""
    aggregator_error: str = ""
    side_effect_policy: bool = False
    quorum: int = 0
    resolved_proposer_models: list[str] = field(default_factory=list)
    resolved_aggregator_model: str = ""

    def to_dict(self) -> dict:
        """Return a plain dict suitable for turn logs and trace events."""
        return asdict(self)


class _AggregatorResponse(BaseModel):
    """Raw JSON contract returned by the aggregator LLM."""

    draft: str
    selected_candidate_ids: list[str] = Field(default_factory=list)
    dropped_candidate_ids: list[str] = Field(default_factory=list)
    summary_for_reviewer: str = ""
    removed_or_uncertain_points: list[str] = Field(default_factory=list)


_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_SECTION_MARKER_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?(DRAFT|REBUTTAL)[ \t]*:"
    r"(?:\*\*)?[ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_CLARIFY_SENTINEL = "<<CLARIFY>>"
_TRUNCATED = "... [truncated]"
_OLDER_EVIDENCE_TRUNCATED = "... [older evidence truncated]"
_T = TypeVar("_T", bound=BaseModel)
_RECOVERABLE_FAILURE_MODES = frozenset({
    "retrieval_not_attempted",
    "retrieval_empty",
})
_USER_BLOCKING_FAILURE_MODES = frozenset({
    "tool_unavailable",
    "user_input_missing",
})
_INTERNAL_REVISION_RE = re.compile(
    r"\b(?:reviser|writer|reviewer|finding|revision_instruction|"
    r"summary_for_reviser)\b|(?:reviser|writer|reviewer)\s*(?:應|should|must)|"
    r"\bdraft\s+(?:should|must)\b|內部|審稿意見",
    re.IGNORECASE,
)
_RETRIEVAL_REVIEW_RULES = """
Finding routing contract, highest priority:
- Before writing any finding, choose whether the issue is recoverable by another
  writer/reviser pass or genuinely needs the user.
- Set failure_mode on each finding when one applies:
  retrieval_not_attempted, retrieval_empty, tool_unavailable,
  user_input_missing, or fabrication_risk.
- Use needs_user_input=true only when another writer/reviser pass cannot fix the
  issue with the currently available tools.
- If the user asks for earlier conversation content and the relevant
  history-retrieval tool is listed under available_tools, but the evidence trace
  has no matching tool call, emit decision=revise with one finding shaped as
  severity=major and needs_user_input=false. The revision_instruction is for the
  writer/reviser and must name the available tool and query to try.
- If the relevant history-retrieval tool was called and the result is empty,
  do not ask the user to restate all research content. Use severity=minor or
  severity=note with needs_user_input=false, and allow an honest draft that says
  the search found insufficient records. A narrow follow-up question is allowed.
- If the relevant history-retrieval tool appears under denied_tools or
  unavailable_base_tools, emit severity=blocker with needs_user_input=true and
  decision=block; the revision_instruction must be user-readable and explain
  that this is a tool policy/settings problem.
- If the user truly has not provided necessary information and the available
  tools cannot recover it, needs_user_input=true is allowed, but the
  revision_instruction must be a concrete user-facing question.
- If the draft introduces research results, data, methods, citations, quotes,
  page numbers, or claims not supported by the input or evidence trace, block
  or revise it. Never allow fabricated scholarly content.
""".strip()


def parse_structured_output(model_type: type[_T], text: str) -> _T:
    """Parse one JSON object into the requested Pydantic model."""
    raw = text.strip()
    fenced = _JSON_FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group("body").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ThinkingOutputError(f"invalid JSON from extended thinking step: {exc}") from exc
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ThinkingOutputError(f"invalid {model_type.__name__}: {exc}") from exc


def trim_tail(text: str, max_chars: int) -> str:
    """Head-truncate text while preserving its tail."""
    if not text:
        return ""
    if max_chars <= 0:
        return _TRUNCATED
    if len(text) <= max_chars:
        return text
    marker = f"{_TRUNCATED}\n"
    keep = max(max_chars - len(marker), 0)
    if keep <= 0:
        return _TRUNCATED
    return marker + text[-keep:]


def trim_head(text: str, max_chars: int) -> str:
    """Tail-truncate text while preserving its head."""
    if not text:
        return ""
    if max_chars <= 0:
        return _TRUNCATED
    if len(text) <= max_chars:
        return text
    marker = f"\n{_TRUNCATED}"
    keep = max(max_chars - len(marker), 0)
    if keep <= 0:
        return _TRUNCATED
    return text[:keep] + marker


def route_review_report(
    report: ReviewReport,
    *,
    attempts: int,
    max_attempts: int = MAX_REVIEW_ATTEMPTS,
) -> ReviewRoute:
    """Route review findings with blocker/user-input checks before revision."""
    recoverable_findings = [
        finding
        for finding in report.findings
        if finding.failure_mode in _RECOVERABLE_FAILURE_MODES
    ]
    blocking_findings = [
        finding
        for finding in report.findings
        if finding.failure_mode in _USER_BLOCKING_FAILURE_MODES
    ]

    if blocking_findings:
        return "ask_user"
    if any(
        finding.needs_user_input
        for finding in report.findings
        if finding.failure_mode not in _RECOVERABLE_FAILURE_MODES
    ):
        return "ask_user"
    if report.decision == "block" and recoverable_findings:
        if attempts >= max_attempts:
            return "stop"
        return "revise"
    if report.decision == "block" or any(
        finding.severity == "blocker"
        and finding.failure_mode not in _RECOVERABLE_FAILURE_MODES
        for finding in report.findings
    ):
        return "ask_user"
    if report.decision == "pass":
        return "pass"
    if attempts >= max_attempts:
        return "stop"
    if any(
        finding.failure_mode == "retrieval_not_attempted"
        for finding in report.findings
    ):
        return "revise"
    if any(finding.severity == "major" for finding in report.findings):
        return "revise"
    return "pass"


def render_review_stop_message(report: ReviewReport) -> str:
    """Render a user-facing stop message for blocker or missing-input findings."""
    findings = [
        finding
        for finding in report.findings
        if finding.needs_user_input or finding.severity == "blocker"
    ]
    if not findings:
        return "目前仍有無法安全自動修正的問題，需要使用者確認。"
    lines = ["目前仍有無法安全自動修正的問題，需要使用者確認："]
    lines.extend(f"- {_user_facing_review_instruction(finding)}" for finding in findings)
    return "\n".join(lines)


def _user_facing_review_instruction(finding: ReviewFinding) -> str:
    instruction = (finding.revision_instruction or "").strip()
    if instruction and not _looks_internal_review_instruction(instruction):
        return instruction

    if finding.failure_mode == "tool_unavailable":
        return (
            "目前需要的工具被 active skill policy 或工具設定排除；請切換 skill、"
            "調整工具設定，或提供可由目前工具讀取的資料位置。"
        )
    if finding.failure_mode == "fabrication_risk":
        return (
            "目前草稿包含缺乏 evidence 支撐的研究內容；請提供來源，"
            "或允許我移除那些 unsupported claims。"
        )
    return "需要更多資訊才能安全完成這個任務；請補充缺少的資料或材料位置。"


def _looks_internal_review_instruction(text: str) -> bool:
    return bool(_INTERNAL_REVISION_RE.search(text))


def render_route_message(
    route: ReviewRoute,
    draft: str,
    report: ReviewReport,
    *,
    format_warning: str = "",
) -> str:
    """Render the final user-visible message for a reviewer route."""
    if route == "ask_user":
        return render_review_stop_message(report)
    if route == "stop":
        answer = (
            draft.rstrip()
            + "\n\n仍需確認處：\n"
            + (report.summary_for_reviser or "Reviewer 仍指出未完全修正的問題。")
        )
        return _prepend_warning(answer, format_warning)
    return _prepend_warning(draft, format_warning)


def rewrite_messages(
    *,
    skill_text: str,
    user_input: str,
    visible_context: str,
    skill_context: str,
    tool_availability: str = "",
) -> list:
    """Build prompt-master rewrite messages."""
    availability = tool_availability.strip() or render_tool_availability_block()
    wrapper = f"""

[內部 extended-thinking wrapper]

你是內部 pipeline 的一環。target tool 是一個 LangGraph research agent。
以下工具可用性區塊是該 agent 本 turn 的實際工具狀態，必須視為唯一事實來源：

{availability}

請把使用者的 prompt 改寫成給該 agent 看的自然語言指令。
若某工具或工具 family 不在 available_tools 內，或出現在 denied_tools /
unavailable_base_tools 內，不要假設 target agent 可以使用它。

硬性禁令：你不得新增以下「原始輸入、visible context 與 active skill context」
三者都未提供的內容：
- citation、DOI、page number、quote
- 數據、樣本數、dataset 名稱、統計結果
- 研究方法細節、實驗條件、研究發現
- 對使用者意圖的擴張詮釋

若必要事實缺失，不要自行補齊，請向使用者詢問。

若需要使用者補充資訊：
第一行寫 <<CLARIFY>>，然後列出最多 3 個澄清問題。

若資訊足夠：
直接輸出改寫後的 prompt，不要前綴、不要解釋、不要 code fence。

語言策略：改寫後的 prompt 與澄清問題使用與「Original user input」相同的語言。
如果使用者輸入是中文，輸出使用繁體中文（絕對不要使用簡體），保留技術專有名詞
（RAG、GPT、DOI、LangGraph、MCP 等）原文，不要翻譯。
""".strip()
    return [
        SystemMessage(content=f"{skill_text.rstrip()}\n\n{wrapper}"),
        HumanMessage(content=(
            f"Original user input:\n{user_input}\n\n"
            "Visible context (recent turns, tail-truncated):\n"
            f"{visible_context or '(none)'}\n\n"
            "Active skill context (head-truncated):\n"
            f"{skill_context or '(none)'}"
        )),
    ]


def review_messages(
    *,
    raw_user_input: str,
    rewritten_prompt: str,
    draft: str,
    skill_context: str,
    evidence_trace_summary: str,
    previous_rebuttal: str,
    tool_availability: str = "",
) -> list:
    """Build reviewer messages for a structured ReviewReport JSON response."""
    availability = tool_availability.strip() or render_tool_availability_block()
    return [
        SystemMessage(content=(
            "You are an independent reviewer for extended thinking mode. "
            "Review the draft against the raw user input, rewritten prompt, "
            "active skill context, tool availability, evidence trace, and previous rebuttal. "
            "Return only valid JSON matching ReviewReport. Do not rewrite the draft.\n\n"
            "語言策略：JSON 內所有自然語言欄位（problem、evidence_from_draft、"
            "revision_instruction、summary_for_reviser）使用與「Raw user input」"
            "相同的語言。如果 raw input 是中文，所有自然語言欄位都使用繁體中文"
            "（絕對不要使用簡體），保留技術專有名詞（RAG、GPT、DOI、JSON 等）原文。"
        )),
        HumanMessage(content=(
            "ReviewReport schema:\n"
            "{\n"
            '  "decision": "pass|revise|block",\n'
            '  "findings": [\n'
            "    {\n"
            '      "severity": "blocker|major|minor|note",\n'
            '      "dimension": "instruction following|background logic|method logic|'
            'claim-evidence alignment|citation integrity|section coherence|other",\n'
            '      "location": "where the issue appears",\n'
            '      "problem": "what is wrong",\n'
            '      "evidence_from_draft": "quote or paraphrase from the draft",\n'
            '      "revision_instruction": "specific fix or user question",\n'
            '      "needs_user_input": true,\n'
            '      "failure_mode": "retrieval_not_attempted|retrieval_empty|'
            'tool_unavailable|user_input_missing|fabrication_risk|null"\n'
            "    }\n"
            "  ],\n"
            '  "summary_for_reviser": "concise actionable summary"\n'
            "}\n\n"
            f"{_RETRIEVAL_REVIEW_RULES}\n\n"
            f"Raw user input:\n{raw_user_input}\n\n"
            f"Rewritten prompt:\n{rewritten_prompt}\n\n"
            f"Active skill context:\n{skill_context or '(none)'}\n\n"
            f"Tool availability:\n{availability}\n\n"
            f"Evidence trace summary:\n{evidence_trace_summary or '(none)'}\n\n"
            f"Previous rebuttal:\n{previous_rebuttal or '(none)'}\n\n"
            f"Draft:\n{draft}"
        )),
    ]


def invoke_text(model, messages: list) -> str:
    """Invoke a LangChain chat model and normalize text content."""
    response = model.invoke(messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(_content_part_to_text(part) for part in content)
    return str(content or "").strip()


def rewrite_prompt(
    model,
    *,
    skill_text: str,
    user_input: str,
    visible_context: str = "",
    skill_context: str = "",
    tool_availability: str = "",
) -> RewriteResult:
    """Run prompt-master rewrite and parse clarify vs rewritten prompt."""
    text = invoke_text(
        model,
        rewrite_messages(
            skill_text=skill_text,
            user_input=user_input,
            visible_context=visible_context,
            skill_context=skill_context,
            tool_availability=tool_availability,
        ),
    )
    stripped = text.lstrip()
    if stripped.startswith(_CLARIFY_SENTINEL):
        return Clarify(text=stripped[len(_CLARIFY_SENTINEL):].strip())
    return Rewrite(prompt=text.strip())


def review_draft(
    model,
    *,
    raw_user_input: str,
    rewritten_prompt: str,
    draft: str,
    skill_context: str = "",
    evidence_trace_summary: str = "",
    previous_rebuttal: str = "",
    tool_availability: str = "",
) -> ReviewReport:
    """Run the reviewer LLM step and parse a ReviewReport."""
    text = invoke_text(
        model,
        review_messages(
            raw_user_input=raw_user_input,
            rewritten_prompt=rewritten_prompt,
            draft=draft,
            skill_context=skill_context,
            evidence_trace_summary=evidence_trace_summary,
            previous_rebuttal=previous_rebuttal,
            tool_availability=tool_availability,
        ),
    )
    return parse_structured_output(ReviewReport, text)


def aggregate_messages(
    *,
    raw_user_input: str,
    rewritten_prompt: str,
    successful_candidates: Sequence[FusionCandidate],
    skill_context: str = "",
    tool_availability: str = "",
) -> list:
    """Build aggregator messages that fuse successful candidate answers.

    The aggregator is a plain LLM synthesis step, NOT a tool call. Its prompt
    must never describe itself as tool evidence so downstream traces keep the
    aggregator out of ``tool_calls``.
    """
    availability = tool_availability.strip() or render_tool_availability_block()
    candidate_blocks = []
    for candidate in successful_candidates:
        candidate_blocks.append(
            f"--- {candidate.candidate_id} (model: {candidate.model_id}) ---\n"
            f"Answer:\n{candidate.answer}\n\n"
            f"Tool trace summary:\n{candidate.tool_trace_summary or '(none)'}"
        )
    candidates_text = "\n\n".join(candidate_blocks) or "(none)"
    valid_ids = ", ".join(candidate.candidate_id for candidate in successful_candidates)
    return [
        SystemMessage(content=(
            "You are the aggregator for extended thinking mode. Several proposer "
            "agents independently answered the same rewritten prompt. Fuse their "
            "candidate answers into one best full-text draft. You are performing a "
            "synthesis step, not calling a tool, and you must not invent a tool "
            "call. Prefer claims that multiple candidates agree on; drop claims "
            "only one candidate makes that look unsupported by its tool trace. Do "
            "not introduce citations, data, methods, or findings that no candidate "
            "and neither the raw input nor the skill context provides. Return only "
            "valid JSON matching the schema. Use only the supplied candidate ids.\n\n"
            "語言策略：JSON 內所有自然語言欄位（draft、summary_for_reviewer、"
            "removed_or_uncertain_points）使用與「Raw user input」相同的語言。"
            "如果 raw input 是中文，使用繁體中文（絕對不要使用簡體），保留技術專有"
            "名詞（RAG、GPT、DOI、JSON 等）原文。"
        )),
        HumanMessage(content=(
            "Aggregate result schema:\n"
            "{\n"
            '  "draft": "the fused full-text answer for the user",\n'
            '  "selected_candidate_ids": ["candidate ids whose content you kept"],\n'
            '  "dropped_candidate_ids": ["candidate ids you rejected"],\n'
            '  "summary_for_reviewer": "what you merged, agreed, or rejected",\n'
            '  "removed_or_uncertain_points": ["claims you removed or flagged uncertain"]\n'
            "}\n\n"
            f"Valid candidate ids: {valid_ids or '(none)'}\n"
            "selected_candidate_ids and dropped_candidate_ids must each be a subset "
            "of the valid candidate ids and must not overlap.\n\n"
            f"Raw user input:\n{raw_user_input}\n\n"
            f"Rewritten prompt:\n{rewritten_prompt}\n\n"
            f"Active skill context:\n{skill_context or '(none)'}\n\n"
            f"Tool availability:\n{availability}\n\n"
            f"Candidate answers:\n{candidates_text}"
        )),
    ]


def parse_aggregate_result(
    text: str,
    *,
    successful_candidate_ids: Sequence[str],
) -> FusionAggregateResult:
    """Parse and validate the aggregator JSON into a FusionAggregateResult.

    Raises :class:`ThinkingOutputError` on invalid JSON, schema violations, a
    blank draft, an unknown candidate id, or selected/dropped overlap. The
    ``reliability_tier`` is left unset; the session control flow owns it.
    """
    raw = text.strip()
    fenced = _JSON_FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group("body").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ThinkingOutputError(f"invalid JSON from aggregator: {exc}") from exc
    if not isinstance(payload, dict):
        raise ThinkingOutputError("aggregator output is not a JSON object")
    try:
        parsed = _AggregatorResponse.model_validate(payload)
    except ValidationError as exc:
        raise ThinkingOutputError(f"invalid aggregator schema: {exc}") from exc

    if not parsed.draft.strip():
        raise ThinkingOutputError("aggregator returned a blank draft")

    valid = set(successful_candidate_ids)
    selected = list(parsed.selected_candidate_ids)
    dropped = list(parsed.dropped_candidate_ids)
    unknown = [cid for cid in (*selected, *dropped) if cid not in valid]
    if unknown:
        raise ThinkingOutputError(
            f"aggregator referenced unknown candidate ids: {', '.join(unknown)}"
        )
    overlap = set(selected) & set(dropped)
    if overlap:
        raise ThinkingOutputError(
            f"aggregator selected and dropped the same candidate ids: "
            f"{', '.join(sorted(overlap))}"
        )

    return FusionAggregateResult(
        draft=parsed.draft,
        selected_candidate_ids=selected,
        dropped_candidate_ids=dropped,
        reliability_tier="",
        summary_for_reviewer=parsed.summary_for_reviewer,
        removed_or_uncertain_points=list(parsed.removed_or_uncertain_points),
        aggregator_error="",
    )


def aggregate_candidates(
    model,
    *,
    raw_user_input: str,
    rewritten_prompt: str,
    successful_candidates: Sequence[FusionCandidate],
    skill_context: str = "",
    tool_availability: str = "",
) -> FusionAggregateResult:
    """Run the aggregator LLM over successful candidates and parse the result.

    Only successful candidates may be passed; failed, timed-out, or empty
    candidates belong in fusion metadata and reviewer evidence, never in the
    aggregator's candidate input.
    """
    text = invoke_text(
        model,
        aggregate_messages(
            raw_user_input=raw_user_input,
            rewritten_prompt=rewritten_prompt,
            successful_candidates=successful_candidates,
            skill_context=skill_context,
            tool_availability=tool_availability,
        ),
    )
    return parse_aggregate_result(
        text,
        successful_candidate_ids=[c.candidate_id for c in successful_candidates],
    )


def build_fusion_evidence_summary(
    *,
    candidates: Sequence[FusionCandidate],
    candidate_traces: Sequence[FusionCandidateTrace],
    aggregate_result: FusionAggregateResult,
    metadata: FusionTurnMetadata,
    per_result_chars: int = 500,
) -> str:
    """Build the reviewer evidence summary from the fusion result objects.

    Each candidate's tool trace is summarized from that candidate's own
    segmented trace, so the same ``tool_call_id`` appearing in two candidates is
    never cross-matched to the wrong ToolMessage.
    """
    trace_by_id = {trace.candidate_id: trace for trace in candidate_traces}
    lines = [
        "=== Fusion candidate panel ===",
        f"reliability_tier: {metadata.reliability_tier or '(none)'}",
        f"selected_candidate_ids: {', '.join(metadata.selected_ids) or '(none)'}",
        f"dropped_candidate_ids: {', '.join(metadata.dropped_ids) or '(none)'}",
        f"omitted_successful_ids: {', '.join(metadata.omitted_successful_ids) or '(none)'}",
    ]
    if metadata.aggregator_error:
        lines.append(f"aggregator_error: {metadata.aggregator_error}")
    if aggregate_result.summary_for_reviewer:
        lines.append(
            f"aggregator_summary_for_reviewer: {aggregate_result.summary_for_reviewer}"
        )
    if aggregate_result.removed_or_uncertain_points:
        lines.append("removed_or_uncertain_points:")
        lines.extend(
            f"  - {point}" for point in aggregate_result.removed_or_uncertain_points
        )
    for candidate in candidates:
        trace = trace_by_id.get(candidate.candidate_id)
        if trace is not None:
            tool_summary = summarize_tool_trace(
                trace.tool_calls,
                trace.new_messages,
                source_label=f"[{candidate.candidate_id} {candidate.model_id}]",
                per_result_chars=per_result_chars,
            )
        else:
            tool_summary = candidate.tool_trace_summary or "Tool calls: none"
        excerpt = (
            trim_head(candidate.answer, per_result_chars)
            if candidate.answer
            else "(no answer)"
        )
        lines.extend([
            f"--- {candidate.candidate_id} (model: {candidate.model_id}) "
            f"status={candidate.status} ---",
            f"answer_excerpt: {excerpt}",
            tool_summary,
        ])
        if candidate.error:
            lines.append(f"error: {candidate.error}")
    return "\n".join(lines)


def summarize_tool_trace(
    tool_calls: list[dict],
    new_messages: list,
    *,
    source_label: str,
    per_result_chars: int = 500,
) -> str:
    """Summarize graph tool calls and matching ToolMessage result excerpts."""
    lines = [f"=== {source_label} ==="]
    if not tool_calls:
        lines.append("Tool calls: none")
        return "\n".join(lines)

    tool_messages = _tool_messages_by_call_id(new_messages)
    seen: set[tuple[str, str, str]] = set()
    for call in tool_calls:
        name = str(call.get("name", "unknown"))
        args_text = json.dumps(call.get("args", {}), ensure_ascii=False, sort_keys=True)
        result_text = _tool_result_text(tool_messages.get(str(call.get("id")), []))
        result_excerpt = trim_head(result_text, per_result_chars) if result_text else "(no result)"
        key = (name, args_text, result_excerpt)
        if key in seen:
            continue
        seen.add(key)
        lines.extend([
            f"- tool: {name}",
            f"  args: {args_text}",
            "  result_excerpt: |",
            *_indent_block(result_excerpt, "    "),
        ])
    return "\n".join(lines)


def append_tool_trace(
    existing: str,
    tool_calls: list[dict],
    new_messages: list,
    *,
    source_label: str,
    per_result_chars: int = 500,
    total_chars_cap: int = 4000,
) -> str:
    """Append one trace segment and keep the newest evidence within a char cap."""
    new_segment = summarize_tool_trace(
        tool_calls,
        new_messages,
        source_label=source_label,
        per_result_chars=per_result_chars,
    )
    combined = "\n\n".join(part for part in (existing.strip(), new_segment.strip()) if part)
    if total_chars_cap <= 0 or len(combined) <= total_chars_cap:
        return combined
    marker = f"{_OLDER_EVIDENCE_TRUNCATED}\n"
    keep = max(total_chars_cap - len(marker), 0)
    if keep <= 0:
        return _OLDER_EVIDENCE_TRUNCATED
    return marker + combined[-keep:]


def parse_reviser_output(text: str, *, repair_model=None) -> RevisedDraft:
    """Parse DRAFT/REBUTTAL output with repair and conservative fallbacks."""
    parsed = _parse_marked_reviser_output(text)
    if parsed is not None:
        return parsed

    if repair_model is not None:
        try:
            repaired = invoke_text(
                repair_model,
                [
                    SystemMessage(content=(
                        "Split the following text strictly into two sections marked "
                        "DRAFT: and REBUTTAL:. Preserve the user's visible draft content. "
                        "Move internal disagreement or reviewer discussion into REBUTTAL. "
                        "Return only the two marked sections. "
                        "Preserve the original content's language verbatim—do not translate. "
                        "Keep the marker names themselves in English (DRAFT, REBUTTAL)."
                    )),
                    HumanMessage(content=text),
                ],
            )
            parsed = _parse_marked_reviser_output(repaired)
            if parsed is not None:
                return parsed
            logger.warning("reviser output marker repair failed")
        except Exception as exc:  # pragma: no cover - logging-only safety path
            logger.warning("reviser output marker repair raised: %s", exc)

    stripped = _heuristic_strip_tail(text)
    if stripped is not None:
        return stripped

    logger.error("reviser output marker parsing failed; using whole text as draft")
    return RevisedDraft(
        draft=text.strip(),
        rebuttal="",
        format_warning=REVISER_FORMAT_WARNING,
    )


def extract_draft_for_user(text: str) -> str:
    """Return the DRAFT section when markers exist, otherwise the whole text."""
    parsed = _parse_marked_reviser_output(text)
    if parsed is None:
        return text.strip()
    return parsed.draft


def _parse_marked_reviser_output(text: str) -> RevisedDraft | None:
    matches = list(_SECTION_MARKER_RE.finditer(text))
    draft_match = next(
        (match for match in matches if match.group(1).casefold() == "draft"),
        None,
    )
    if draft_match is None:
        return None

    rebuttal_match = next(
        (
            match
            for match in matches
            if match.start() > draft_match.start()
            and match.group(1).casefold() == "rebuttal"
        ),
        None,
    )
    draft_end = rebuttal_match.start() if rebuttal_match else len(text)
    draft = _section_text(text, draft_match, draft_end).strip()
    if not draft:
        return None
    if rebuttal_match is None:
        return RevisedDraft(draft=draft, rebuttal="")
    rebuttal = _section_text(text, rebuttal_match, len(text)).strip()
    return RevisedDraft(draft=draft, rebuttal=rebuttal)


def _section_text(text: str, match: re.Match[str], end: int) -> str:
    inline = match.group(2).strip()
    body = text[match.end():end].lstrip("\r\n")
    if inline and body:
        return f"{inline}\n{body}"
    return inline or body


def _heuristic_strip_tail(text: str) -> RevisedDraft | None:
    raw = text.strip()
    if not raw:
        return RevisedDraft(draft="", rebuttal="", format_warning=REVISER_FORMAT_WARNING)
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n|(?=^#{1,6}\s+)", raw, flags=re.MULTILINE)
        if part.strip()
    ]
    if len(paragraphs) <= 1:
        return None

    internal_keywords = (
        "REBUTTAL",
        "rebuttal",
        "駁斥",
        "我不同意",
        "I disagree",
        "Reviewer feedback",
        "Internal note",
        "(none)",
    )
    stripped_parts: list[str] = []
    while paragraphs and any(keyword in paragraphs[-1] for keyword in internal_keywords):
        stripped_parts.insert(0, paragraphs.pop())
    if not stripped_parts:
        return None

    draft = "\n\n".join(paragraphs).strip()
    rebuttal = "\n\n".join(stripped_parts).strip()
    stripped_chars = len(rebuttal)
    if not draft or stripped_chars > len(raw) * 0.5:
        logger.error("reviser heuristic fallback stripped too much internal text")
        return None
    return RevisedDraft(draft=draft, rebuttal=rebuttal)


def _tool_messages_by_call_id(new_messages: list) -> dict[str, list[ToolMessage]]:
    tool_messages: dict[str, list[ToolMessage]] = {}
    for message in new_messages:
        if not isinstance(message, ToolMessage):
            continue
        call_id = getattr(message, "tool_call_id", None)
        if call_id:
            tool_messages.setdefault(str(call_id), []).append(message)
    return tool_messages


def _tool_result_text(messages: list[ToolMessage]) -> str:
    return "\n".join(
        _message_content_to_text(getattr(message, "content", ""))
        for message in messages
    ).strip()


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(_content_part_to_text(part) for part in content)
    return str(content or "")


def _content_part_to_text(part: Any) -> str:
    if isinstance(part, dict):
        text = part.get("text")
        if text is not None:
            return str(text)
    return str(part)


def _indent_block(text: str, prefix: str) -> list[str]:
    return [f"{prefix}{line}" for line in text.splitlines() or [""]]


def _prepend_warning(answer: str, warning: str) -> str:
    if not warning:
        return answer
    return f"{warning}\n\n{answer}"
