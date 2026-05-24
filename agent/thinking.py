"""Structured helpers for the optional extended thinking workflow."""

from __future__ import annotations

import json
import re
from typing import Literal, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError


TaskDecision = Literal["proceed", "proceed_with_assumptions", "need_clarification", "block"]
ReviewSeverity = Literal["blocker", "major", "minor", "note"]
ReviewDecision = Literal["pass", "revise", "block"]
TaskRoute = Literal["write", "clarify", "block"]
ReviewRoute = Literal["pass", "revise", "ask_user", "stop"]

MAX_REVIEW_ATTEMPTS = 2


class ThinkingOutputError(ValueError):
    """Raised when an extended-thinking LLM step returns invalid structured output."""


class TaskSpec(BaseModel):
    task: str
    task_type: str
    target_output: str

    confidence: Literal["high", "medium", "low"]
    decision: TaskDecision

    known_from_user: list[str]
    known_from_context: list[str]
    allowed_assumptions: list[str]
    forbidden_assumptions: list[str]
    missing_info: list[str]

    constraints: list[str]
    success_criteria: list[str]

    writer_instruction: str
    reviewer_instruction: str


class ReviewFinding(BaseModel):
    severity: ReviewSeverity
    dimension: str
    location: str
    problem: str
    evidence_from_draft: str
    revision_instruction: str
    needs_user_input: bool


class ReviewReport(BaseModel):
    decision: ReviewDecision
    findings: list[ReviewFinding] = Field(default_factory=list)
    summary_for_reviser: str


_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_T = TypeVar("_T", bound=BaseModel)


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


def route_task_spec(spec: TaskSpec) -> TaskRoute:
    """Route the compiler decision without forcing an unsafe executable task."""
    if spec.decision in {"proceed", "proceed_with_assumptions"}:
        return "write"
    if spec.decision == "need_clarification":
        return "clarify"
    return "block"


def route_review_report(
    report: ReviewReport,
    *,
    attempts: int,
    max_attempts: int = MAX_REVIEW_ATTEMPTS,
) -> ReviewRoute:
    """Route review findings with blocker/user-input checks before revision."""
    if any(finding.needs_user_input for finding in report.findings):
        return "ask_user"
    if report.decision == "block" or any(
        finding.severity == "blocker" for finding in report.findings
    ):
        return "ask_user"
    if report.decision == "pass":
        return "pass"
    if attempts >= max_attempts:
        return "stop"
    if any(finding.severity == "major" for finding in report.findings):
        return "revise"
    return "pass"


def render_task_spec_stop_message(spec: TaskSpec) -> str:
    """Render the compiler clarification/block response."""
    if spec.decision == "need_clarification":
        heading = "需要補充資訊才能安全完成："
    else:
        heading = "目前不能安全完成這個任務："
    missing = spec.missing_info or spec.forbidden_assumptions or spec.constraints
    if not missing:
        return heading
    bullets = "\n".join(f"- {item}" for item in missing)
    return f"{heading}\n{bullets}"


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
    lines.extend(f"- {finding.revision_instruction}" for finding in findings)
    return "\n".join(lines)


def append_assumption_note(answer: str, spec: TaskSpec) -> str:
    """Append a concise assumption note for proceed_with_assumptions."""
    if spec.decision != "proceed_with_assumptions" or not spec.allowed_assumptions:
        return answer
    lines = ["", "採用的假設："]
    lines.extend(f"- {item}" for item in spec.allowed_assumptions)
    return answer.rstrip() + "\n".join(lines)


def task_spec_messages(
    *,
    user_input: str,
    visible_context: str,
    skill_context: str,
) -> list:
    """Build compiler messages for a structured TaskSpec JSON response."""
    return [
        SystemMessage(content=(
            "You compile the user's request into a strict TaskSpec JSON object. "
            "Return only valid JSON matching the requested schema. Do not solve the task."
        )),
        HumanMessage(content=(
            "TaskSpec schema fields:\n"
            "- task, task_type, target_output\n"
            "- confidence: high|medium|low\n"
            "- decision: proceed|proceed_with_assumptions|need_clarification|block\n"
            "- known_from_user, known_from_context, allowed_assumptions, "
            "forbidden_assumptions, missing_info, constraints, success_criteria\n"
            "- writer_instruction, reviewer_instruction\n\n"
            f"User input:\n{user_input}\n\n"
            f"Visible context:\n{visible_context or '(none)'}\n\n"
            f"Active skill context:\n{skill_context or '(none)'}"
        )),
    ]


def review_messages(
    *,
    user_input: str,
    task_spec: TaskSpec,
    draft: str,
    skill_context: str,
) -> list:
    """Build reviewer messages for a structured ReviewReport JSON response."""
    return [
        SystemMessage(content=(
            "You review the draft against the TaskSpec. Return only valid JSON "
            "matching ReviewReport. Do not rewrite the draft."
        )),
        HumanMessage(content=(
            f"User input:\n{user_input}\n\n"
            f"TaskSpec JSON:\n{task_spec.model_dump_json(indent=2)}\n\n"
            f"Active skill context:\n{skill_context or '(none)'}\n\n"
            f"Draft:\n{draft}"
        )),
    ]


def reviser_messages(
    *,
    user_input: str,
    task_spec: TaskSpec,
    draft: str,
    review_report: ReviewReport,
) -> list:
    """Build reviser messages constrained to safe major findings only."""
    return [
        SystemMessage(content=(
            "Revise the draft only for major findings that do not require user input. "
            "Do not add citations, data, methods, findings, or assumptions forbidden "
            "by the TaskSpec. Return only the revised answer text."
        )),
        HumanMessage(content=(
            f"User input:\n{user_input}\n\n"
            f"TaskSpec JSON:\n{task_spec.model_dump_json(indent=2)}\n\n"
            f"Original draft:\n{draft}\n\n"
            f"ReviewReport JSON:\n{review_report.model_dump_json(indent=2)}"
        )),
    ]


def invoke_text(model, messages: list) -> str:
    """Invoke a LangChain chat model and normalize text content."""
    response = model.invoke(messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content or "").strip()


def compile_task_spec(
    model,
    *,
    user_input: str,
    visible_context: str = "",
    skill_context: str = "",
) -> TaskSpec:
    """Run the compiler LLM step and parse a TaskSpec."""
    text = invoke_text(
        model,
        task_spec_messages(
            user_input=user_input,
            visible_context=visible_context,
            skill_context=skill_context,
        ),
    )
    return parse_structured_output(TaskSpec, text)


def review_draft(
    model,
    *,
    user_input: str,
    task_spec: TaskSpec,
    draft: str,
    skill_context: str = "",
) -> ReviewReport:
    """Run the reviewer LLM step and parse a ReviewReport."""
    text = invoke_text(
        model,
        review_messages(
            user_input=user_input,
            task_spec=task_spec,
            draft=draft,
            skill_context=skill_context,
        ),
    )
    return parse_structured_output(ReviewReport, text)


def revise_draft(
    model,
    *,
    user_input: str,
    task_spec: TaskSpec,
    draft: str,
    review_report: ReviewReport,
) -> str:
    """Run the reviser LLM step."""
    return invoke_text(
        model,
        reviser_messages(
            user_input=user_input,
            task_spec=task_spec,
            draft=draft,
            review_report=review_report,
        ),
    )
