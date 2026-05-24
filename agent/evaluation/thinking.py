"""Evaluation cases for the extended thinking reviewer step."""

from __future__ import annotations

import json
from pathlib import Path

from agent.config import AgentConfig
from agent.evaluation.base import BaseEvaluator, EvalResult
from agent.llm.openrouter import get_chat_model
from agent.thinking import (
    ReviewReport,
    TaskSpec,
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
        self.model = model or get_chat_model(self.config)

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Return a small built-in reviewer eval set."""
        cases = [
            {
                "id": "major_claim_evidence_alignment",
                "category": "major_issue_detection",
                "user_input": "Revise this abstract without adding new study results.",
                "task_spec": _base_task_spec(),
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
                "task_spec": _base_task_spec(
                    forbidden_assumptions=[
                        "do not invent citations",
                        "do not invent statistics",
                    ],
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
                task_spec = TaskSpec.model_validate(case["task_spec"])
                report = review_draft(
                    self.model,
                    user_input=case["user_input"],
                    task_spec=task_spec,
                    draft=case["draft"],
                    skill_context=case.get("skill_context", ""),
                )
                scores = self._score_report(case, report)
                error = None
            except (KeyError, ValueError, ThinkingOutputError) as exc:
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


def _base_task_spec(**overrides) -> dict:
    data = {
        "task": "revise academic prose",
        "task_type": "academic_revision",
        "target_output": "revised academic text",
        "confidence": "high",
        "decision": "proceed",
        "known_from_user": ["draft supplied"],
        "known_from_context": [],
        "allowed_assumptions": [],
        "forbidden_assumptions": [
            "do not invent citations",
            "do not invent data",
            "do not invent research findings",
        ],
        "missing_info": [],
        "constraints": ["preserve evidence boundaries"],
        "success_criteria": [
            "claims align with supplied evidence",
            "no fabricated scholarly content",
        ],
        "writer_instruction": "Revise without adding facts.",
        "reviewer_instruction": "Check claim-evidence alignment and citation integrity.",
    }
    data.update(overrides)
    return data
