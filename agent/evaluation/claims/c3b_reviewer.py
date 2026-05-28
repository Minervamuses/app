"""C3b reviewer-as-classifier evaluator with deterministic scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from agent.config import AgentConfig
from agent.evaluation.base import BaseEvaluator, EvalResult
from agent.evaluation.datasets import EvalCase, load_claim_dataset
from agent.llm.thinking import ExtendedModeNotConfigured, get_chat_model_for_role
from agent.thinking import (
    ReviewReport,
    ThinkingOutputError,
    review_draft,
    route_review_report,
)

_DECISION_LABELS = ("pass", "revise", "block")
_ROUTE_LABELS = ("pass", "revise", "ask_user", "stop")
_FAILURE_MODE_LABELS = (
    "retrieval_not_attempted",
    "retrieval_empty",
    "tool_unavailable",
    "user_input_missing",
    "fabrication_risk",
)
_SEVERITY_RANK = {
    "note": 0,
    "minor": 1,
    "major": 2,
    "blocker": 3,
}
_CounterKey = Literal["tp", "fp", "fn", "tn"]


class C3ReviewerEvaluator(BaseEvaluator):
    """Evaluate the extended-thinking reviewer as a classifier."""

    def __init__(self, config: AgentConfig | None = None, *, model=None):
        self.config = config or AgentConfig()
        self.model = model or get_chat_model_for_role(self.config, role="reviewer")

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Load the frozen C3 reviewer dev dataset."""
        del n
        cases = [
            case.raw
            for case in load_claim_dataset("c3", "dev")
            if case.raw.get("task") == "reviewer"
        ]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(cases, file_obj, ensure_ascii=False, indent=2)
        return cases

    def evaluate(self, cases: list[EvalCase | dict]) -> EvalResult:
        details = []
        counters = _new_counters()
        parse_errors = 0
        for raw_case in cases:
            case = _raw_case(raw_case)
            try:
                report = review_draft(
                    self.model,
                    raw_user_input=case["inputs"]["raw_user_input"],
                    rewritten_prompt=case["inputs"].get("rewritten_prompt", ""),
                    draft=case["inputs"]["draft"],
                    skill_context=case["inputs"].get("skill_context", ""),
                    evidence_trace_summary=case["inputs"].get("evidence_trace_summary", ""),
                    previous_rebuttal=case["inputs"].get("previous_rebuttal", ""),
                    tool_availability=case["inputs"].get("tool_availability", ""),
                )
                scores = score_review_report(
                    report,
                    case["gold"],
                    attempts=int(case["inputs"].get("attempts", 0)),
                )
                _merge_counters(counters, scores["counters"])
                error = None
            except (KeyError, ValueError, ExtendedModeNotConfigured, ThinkingOutputError) as exc:
                report = None
                scores = _error_scores(case["gold"])
                _merge_counters(counters, scores["counters"])
                parse_errors += 1
                error = f"{type(exc).__name__}: {exc}"

            details.append({
                "id": case["id"],
                "scores": scores["checks"],
                "prediction": scores["prediction"],
                "gold": case["gold"],
                "report": report.model_dump() if report is not None else None,
                "error": error,
            })

        aggregates = _aggregate_scores(counters)
        aggregates["parse_success_rate"] = (
            (len(cases) - parse_errors) / len(cases) if cases else 0.0
        )
        return EvalResult(
            name="C3Reviewer",
            total=len(cases),
            scores=aggregates,
            details=details,
            metadata={"claim": "c3", "subclaim": "reviewer"},
        )


def score_review_report(
    report: ReviewReport,
    gold: dict,
    *,
    attempts: int = 0,
) -> dict:
    """Score one reviewer report against fixed classifier labels."""
    prediction = _prediction_from_report(report, attempts=attempts)
    counters = _new_counters()

    _score_multiclass(
        counters["decision"],
        labels=_DECISION_LABELS,
        predicted=prediction["decision"],
        gold=gold["decision"],
    )
    _score_multiclass(
        counters["route"],
        labels=_ROUTE_LABELS,
        predicted=prediction["route"],
        gold=gold["route"],
    )
    _score_multilabel(
        counters["failure_mode"],
        labels=_FAILURE_MODE_LABELS,
        predicted=set(prediction["failure_modes"]),
        gold=_gold_failure_modes(gold),
    )
    _score_binary(
        counters["needs_user_input"],
        "needs_user_input",
        predicted=bool(prediction["needs_user_input"]),
        gold=bool(gold.get("needs_user_input", False)),
    )
    _score_binary(
        counters["severity"],
        "severity_at_or_above_min",
        predicted=_severity_positive(prediction["max_severity"], gold.get("min_severity")),
        gold=gold.get("min_severity") is not None,
    )

    return {
        "prediction": prediction,
        "checks": {
            "decision_ok": prediction["decision"] == gold["decision"],
            "route_ok": prediction["route"] == gold["route"],
            "failure_modes_ok": set(prediction["failure_modes"]) == _gold_failure_modes(gold),
            "needs_user_input_ok": (
                bool(prediction["needs_user_input"])
                == bool(gold.get("needs_user_input", False))
            ),
            "severity_ok": _severity_positive(
                prediction["max_severity"],
                gold.get("min_severity"),
            )
            == (gold.get("min_severity") is not None),
        },
        "counters": counters,
    }


def _prediction_from_report(report: ReviewReport, *, attempts: int) -> dict:
    failure_modes = sorted({
        finding.failure_mode
        for finding in report.findings
        if finding.failure_mode is not None
    })
    severities = [
        finding.severity for finding in report.findings
        if finding.severity in _SEVERITY_RANK
    ]
    max_severity = max(severities, key=lambda item: _SEVERITY_RANK[item], default=None)
    return {
        "decision": report.decision,
        "route": route_review_report(report, attempts=attempts),
        "failure_modes": failure_modes,
        "needs_user_input": any(finding.needs_user_input for finding in report.findings),
        "max_severity": max_severity,
    }


def _error_scores(gold: dict) -> dict:
    prediction = {
        "decision": None,
        "route": None,
        "failure_modes": [],
        "needs_user_input": False,
        "max_severity": None,
    }
    counters = _new_counters()
    _score_multiclass(counters["decision"], labels=_DECISION_LABELS, predicted=None, gold=gold["decision"])
    _score_multiclass(counters["route"], labels=_ROUTE_LABELS, predicted=None, gold=gold["route"])
    _score_multilabel(
        counters["failure_mode"],
        labels=_FAILURE_MODE_LABELS,
        predicted=set(),
        gold=_gold_failure_modes(gold),
    )
    _score_binary(
        counters["needs_user_input"],
        "needs_user_input",
        predicted=False,
        gold=bool(gold.get("needs_user_input", False)),
    )
    _score_binary(
        counters["severity"],
        "severity_at_or_above_min",
        predicted=False,
        gold=gold.get("min_severity") is not None,
    )
    return {
        "prediction": prediction,
        "checks": {
            "decision_ok": False,
            "route_ok": False,
            "failure_modes_ok": not _gold_failure_modes(gold),
            "needs_user_input_ok": not bool(gold.get("needs_user_input", False)),
            "severity_ok": gold.get("min_severity") is None,
        },
        "counters": counters,
    }


def _new_counters() -> dict[str, dict[str, dict[_CounterKey, int]]]:
    return {
        "decision": _label_counters(_DECISION_LABELS),
        "route": _label_counters(_ROUTE_LABELS),
        "failure_mode": _label_counters(_FAILURE_MODE_LABELS),
        "needs_user_input": _label_counters(("needs_user_input",)),
        "severity": _label_counters(("severity_at_or_above_min",)),
    }


def _label_counters(labels: tuple[str, ...]) -> dict[str, dict[_CounterKey, int]]:
    return {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in labels}


def _score_multiclass(counters, *, labels, predicted, gold) -> None:
    for label in labels:
        _score_binary(counters, label, predicted=predicted == label, gold=gold == label)


def _score_multilabel(counters, *, labels, predicted: set[str], gold: set[str]) -> None:
    for label in labels:
        _score_binary(counters, label, predicted=label in predicted, gold=label in gold)


def _score_binary(counters, label: str, *, predicted: bool, gold: bool) -> None:
    if predicted and gold:
        counters[label]["tp"] += 1
    elif predicted and not gold:
        counters[label]["fp"] += 1
    elif not predicted and gold:
        counters[label]["fn"] += 1
    else:
        counters[label]["tn"] += 1


def _merge_counters(base, incoming) -> None:
    for group, labels in incoming.items():
        for label, counts in labels.items():
            for key, value in counts.items():
                base[group][label][key] += value


def _aggregate_scores(counters) -> dict[str, float]:
    scores = {}
    for group, labels in counters.items():
        macro = _macro_prf(labels)
        scores[f"{group}_macro_precision"] = macro["precision"]
        scores[f"{group}_macro_recall"] = macro["recall"]
        scores[f"{group}_macro_f1"] = macro["f1"]
    return scores


def _macro_prf(label_counts) -> dict[str, float]:
    active = [
        _prf(counts)
        for counts in label_counts.values()
        if counts["tp"] + counts["fp"] + counts["fn"] > 0
    ]
    if not active:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": sum(item["precision"] for item in active) / len(active),
        "recall": sum(item["recall"] for item in active) / len(active),
        "f1": sum(item["f1"] for item in active) / len(active),
    }


def _prf(counts) -> dict[str, float]:
    precision = (
        counts["tp"] / (counts["tp"] + counts["fp"])
        if counts["tp"] + counts["fp"]
        else 0.0
    )
    recall = (
        counts["tp"] / (counts["tp"] + counts["fn"])
        if counts["tp"] + counts["fn"]
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _gold_failure_modes(gold: dict) -> set[str]:
    modes = gold.get("failure_modes")
    if modes is None and gold.get("failure_mode") is not None:
        modes = [gold["failure_mode"]]
    return set(modes or [])


def _severity_positive(predicted: str | None, min_severity: str | None) -> bool:
    if predicted is None or min_severity is None:
        return False
    return _SEVERITY_RANK[predicted] >= _SEVERITY_RANK[min_severity]


def _raw_case(case: EvalCase | dict) -> dict:
    return case.raw if isinstance(case, EvalCase) else dict(case)
