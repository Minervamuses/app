"""Evaluation cases for the extended thinking reviewer step."""

from __future__ import annotations

import json
from pathlib import Path

from agent.config import AgentConfig
from agent.evaluation.base import BaseEvaluator, EvalResult
from agent.llm.thinking import ExtendedModeNotConfigured, get_chat_model_for_role
from agent.thinking import (
    ReviewReport,
    ThinkingOutputError,
    review_draft,
)


_SEVERITY_RANK = {
    "note": 0,
    "minor": 1,
    "major": 2,
    "blocker": 3,
}


class ThinkingReviewerEvaluator(BaseEvaluator):
    """Evaluate whether the extended reviewer catches high-risk draft issues."""

    def __init__(self, config: AgentConfig | None = None, *, model=None):
        self.config = config or AgentConfig()
        self.model = model or get_chat_model_for_role(self.config, role="reviewer")

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Return a small built-in reviewer eval set."""
        cases = [
            {
                "id": "major_claim_evidence_alignment",
                "category": "major_issue_detection",
                "user_input": "Revise this abstract without adding new study results.",
                "rewritten_prompt": _base_rewritten_prompt(),
                "draft": (
                    "This study definitively proves the intervention improves all "
                    "clinical outcomes, although the supplied notes only mention "
                    "a preliminary feasibility signal."
                ),
                "expected_min_severity": "major",
                "expected_needs_user_input": False,
                "rationale": "Reviewer should flag overclaiming against supplied evidence.",
            },
            {
                "id": "academic_integrity_missing_source",
                "category": "academic_integrity",
                "user_input": "Polish this paragraph for an academic paper.",
                "rewritten_prompt": _base_rewritten_prompt(
                    extra_constraints=[
                        "Do not invent citations.",
                        "Do not invent statistics.",
                    ]
                ),
                "draft": (
                    "Prior work shows a 72% improvement in adherence "
                    "(Chen, 2025), but no source or dataset was supplied."
                ),
                "expected_min_severity": "blocker",
                "expected_needs_user_input": True,
                "rationale": "Reviewer should stop on fabricated citation/statistic risk.",
            },
        ]
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        return cases

    def evaluate(self, cases: list[dict]) -> EvalResult:
        details: list[dict] = []
        severity_hits = 0
        user_input_hits = 0
        passed = 0

        for case in cases:
            try:
                report = review_draft(
                    self.model,
                    raw_user_input=case["user_input"],
                    rewritten_prompt=case["rewritten_prompt"],
                    draft=case["draft"],
                    skill_context=case.get("skill_context", ""),
                    evidence_trace_summary=case.get("evidence_trace_summary", ""),
                    previous_rebuttal=case.get("previous_rebuttal", ""),
                )
                scores = self._score_report(case, report)
                error = None
            except (KeyError, ValueError, ExtendedModeNotConfigured, ThinkingOutputError) as exc:
                report = None
                scores = {
                    "severity_ok": False,
                    "user_input_ok": False,
                    "passed": False,
                }
                error = f"{type(exc).__name__}: {exc}"

            severity_hits += int(scores["severity_ok"])
            user_input_hits += int(scores["user_input_ok"])
            passed += int(scores["passed"])
            details.append({
                "id": case.get("id"),
                "category": case.get("category"),
                "scores": scores,
                "report": report.model_dump() if report is not None else None,
                "error": error,
            })

        total = len(cases)
        denom = total or 1
        return EvalResult(
            name="ThinkingReviewer",
            total=total,
            scores={
                "reviewer_detection_accuracy": passed / denom,
                "severity_accuracy": severity_hits / denom,
                "user_input_accuracy": user_input_hits / denom,
            },
            details=details,
            metadata={"cases": total},
        )

    @staticmethod
    def _score_report(case: dict, report: ReviewReport) -> dict[str, bool]:
        expected_min = case.get("expected_min_severity", "major")
        expected_needs_user_input = case.get("expected_needs_user_input")
        highest = max(
            (_SEVERITY_RANK.get(finding.severity, -1) for finding in report.findings),
            default=-1,
        )
        severity_ok = highest >= _SEVERITY_RANK[expected_min]
        if expected_needs_user_input is None:
            user_input_ok = True
        else:
            user_input_ok = any(
                finding.needs_user_input == expected_needs_user_input
                for finding in report.findings
            )
        return {
            "severity_ok": severity_ok,
            "user_input_ok": user_input_ok,
            "passed": severity_ok and user_input_ok,
        }


def _base_rewritten_prompt(*, extra_constraints: list[str] | None = None) -> str:
    constraints = [
        "Preserve evidence boundaries.",
        "Do not invent citations.",
        "Do not invent data.",
        "Do not invent research findings.",
        *(extra_constraints or []),
    ]
    rendered = "\n".join(f"- {constraint}" for constraint in constraints)
    return (
        "Revise the supplied academic prose for clarity and style while preserving "
        "the user's evidence boundaries.\n\n"
        f"Constraints:\n{rendered}\n\n"
        "Reviewer focus: claim-evidence alignment and citation integrity."
    )
