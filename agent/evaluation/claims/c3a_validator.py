"""C3a deterministic skill-validator evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from agent.config import AgentConfig
from agent.evaluation.base import BaseEvaluator, EvalResult
from agent.evaluation.datasets import EvalCase, load_claim_dataset
from agent.skills.validator import validate_skill_output


class C3ValidatorEvaluator(BaseEvaluator):
    """Evaluate deterministic skill validator predictions on labeled outputs."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Load the frozen C3 validator dev dataset."""
        del n
        cases = [
            case.raw
            for case in load_claim_dataset("c3", "dev")
            if case.raw.get("task") == "validator"
        ]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(cases, file_obj, ensure_ascii=False, indent=2)
        return cases

    def evaluate(self, cases: list[EvalCase | dict]) -> EvalResult:
        """Run `validate_skill_output` and aggregate deterministic metrics."""
        details = []
        tp = fp = fn = tn = exact = 0
        for raw_case in cases:
            case = _raw_case(raw_case)
            active_skill = case["inputs"].get("active_skill")
            text = case["inputs"]["text"]
            expected = list(case["gold"].get("expected_violations", []))
            predicted = validate_skill_output(active_skill=active_skill, text=text)
            scores = score_validator_predictions(predicted, expected)

            tp += int(scores["true_positive"])
            fp += int(scores["false_positive"])
            fn += int(scores["false_negative"])
            tn += int(scores["true_negative"])
            exact += int(scores["exact_match"])
            details.append({
                "id": case["id"],
                "active_skill": active_skill,
                "prediction": predicted,
                "gold": expected,
                "scores": scores,
            })

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        total = len(cases)
        return EvalResult(
            name="C3Validator",
            total=total,
            scores={
                "violation_precision": precision,
                "violation_recall": recall,
                "violation_f1": f1,
                "exact_match": exact / total if total else 0.0,
                "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            },
            details=details,
            metadata={"claim": "c3", "subclaim": "validator"},
        )


def score_validator_predictions(predicted: list[str], expected: list[str]) -> dict[str, bool]:
    """Score validator output as exact labels plus any-violation binary class."""
    predicted_set = set(predicted)
    expected_set = set(expected)
    pred_positive = bool(predicted_set)
    gold_positive = bool(expected_set)
    return {
        "exact_match": predicted_set == expected_set,
        "true_positive": pred_positive and gold_positive,
        "false_positive": pred_positive and not gold_positive,
        "false_negative": not pred_positive and gold_positive,
        "true_negative": not pred_positive and not gold_positive,
    }


def _raw_case(case: EvalCase | dict) -> dict:
    return case.raw if isinstance(case, EvalCase) else dict(case)
