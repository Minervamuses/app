"""C4 end-to-end deterministic checklist evaluator."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Callable

from langgraph.errors import GraphRecursionError

from agent.config import AgentConfig
from agent.evaluation.base import BaseEvaluator, EvalResult
from agent.evaluation.datasets import EvalCase, load_claim_dataset
from agent.session import ChatSession

TurnRunner = Callable[[list[str]], tuple[str, list[dict]]]


class C4ChecklistEvaluator(BaseEvaluator):
    """Evaluate task completion with an independent deterministic checklist."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        extra_tools: list | None = None,
        turn_runner: TurnRunner | None = None,
    ):
        self.config = config or AgentConfig()
        self.extra_tools = list(extra_tools or [])
        self.turn_runner = turn_runner

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Load the frozen C4 dev dataset."""
        del n
        cases = [case.raw for case in load_claim_dataset("c4", "dev")]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(cases, file_obj, ensure_ascii=False, indent=2)
        return cases

    def evaluate(self, cases: list[EvalCase | dict]) -> EvalResult:
        details = []
        checklist_hits = 0
        tool_hits = 0
        answer_hits = 0
        for raw_case in cases:
            case = _raw_case(raw_case)
            answer, tool_calls, error = self._run_case(case)
            scores = score_c4_checklist(answer, tool_calls, case["gold"]["checklist"])
            checklist_hits += int(scores["passed"])
            tool_hits += int(scores["required_tools_ok"])
            answer_hits += int(scores["answer_requirements_ok"])
            detail = {
                "id": case["id"],
                "answer": answer,
                "actual_tools": [call["name"] for call in tool_calls],
                "scores": scores,
                "gold": case["gold"],
            }
            if error:
                detail["error"] = error
            details.append(detail)

        total = len(cases)
        return EvalResult(
            name="C4Checklist",
            total=total,
            scores={
                "task_success_rate": checklist_hits / total if total else 0.0,
                "required_tools_accuracy": tool_hits / total if total else 0.0,
                "answer_requirements_accuracy": answer_hits / total if total else 0.0,
            },
            details=details,
            metadata={"claim": "c4"},
        )

    def _run_case(self, case: dict) -> tuple[str, list[dict], str | None]:
        messages = list(case["inputs"].get("messages") or [case["inputs"]["prompt"]])
        if self.turn_runner is not None:
            answer, calls = self.turn_runner(messages)
            return answer, calls, None

        session = ChatSession(
            self.config,
            recursion_limit=32,
            extra_tools=self.extra_tools,
        )
        tool_calls: list[dict] = []
        answer = ""
        try:
            for message in messages:
                answer, turn_calls = asyncio.run(session.turn_with_trace(message))
                tool_calls.extend(turn_calls)
        except GraphRecursionError:
            return "", [], "GraphRecursionError"
        except Exception as exc:
            return "", [], f"{type(exc).__name__}: {exc}"
        return answer, tool_calls, None


def score_c4_checklist(answer: str, tool_calls: list[dict], checklist: dict) -> dict[str, bool]:
    """Score one C4 output with deterministic tool and answer checks."""
    actual_tools = [call["name"] for call in tool_calls]
    required_tools = list(checklist.get("required_tools", []))
    forbidden_tools = list(checklist.get("forbidden_tools", []))
    required_tools_ok = all(tool in actual_tools for tool in required_tools)
    forbidden_tools_ok = not any(tool in actual_tools for tool in forbidden_tools)

    answer_contains = list(checklist.get("answer_contains", []))
    answer_regex = list(checklist.get("answer_regex", []))
    answer_contains_ok = all(fragment in answer for fragment in answer_contains)
    answer_regex_ok = all(re.search(pattern, answer, re.IGNORECASE) for pattern in answer_regex)
    answer_requirements_ok = answer_contains_ok and answer_regex_ok
    return {
        "required_tools_ok": required_tools_ok,
        "forbidden_tools_ok": forbidden_tools_ok,
        "answer_contains_ok": answer_contains_ok,
        "answer_regex_ok": answer_regex_ok,
        "answer_requirements_ok": answer_requirements_ok,
        "passed": required_tools_ok and forbidden_tools_ok and answer_requirements_ok,
    }


def _raw_case(case: EvalCase | dict) -> dict:
    return case.raw if isinstance(case, EvalCase) else dict(case)
