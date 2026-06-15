"""C1 tool-routing claim evaluator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langgraph.errors import GraphRecursionError

from agent.config import AgentConfig
from agent.evaluation.base import BaseEvaluator, EvalResult, tool_inventory
from agent.evaluation.datasets import EvalCase, load_claim_dataset
from agent.evaluation.metrics.tool_routing import missing_required_tools, score_tool_expectations
from agent.history_rag import ChatHistoryStore
from agent.session import ChatSession

TurnRunner = Callable[[list[str], list[dict] | None], list[dict]]

_METRIC_NAMES = {
    "first_tool": "first_tool_accuracy",
    "count_ok": "tool_count_accuracy",
    "no_tool": "no_tool_accuracy",
    "tools_covered": "tools_coverage",
    "tools_any": "tools_any_accuracy",
    "forbidden_ok": "forbidden_tool_accuracy",
    "tool_family": "tool_family_accuracy",
    "filters_used": "filter_accuracy",
    "answer_ok": "answer_accuracy",
}


class MissingRequiredToolsError(RuntimeError):
    """Raised when an official C1 run cannot evaluate all required cases."""


class _StaticHistoryStore:
    """Tiny deterministic store for C1 cases that need hidden history."""

    def __init__(self, entries: list[dict]):
        self.documents = [
            Document(
                page_content=entry["text"],
                metadata={
                    "role": entry.get("role", "user"),
                    "turn_id": entry.get("turn_id", idx + 1),
                    "timestamp": entry.get("timestamp", "2026-04-27T00:00:00+00:00"),
                },
            )
            for idx, entry in enumerate(entries)
        ]
        self.adds: list[dict] = []

    def search(self, query: str, k: int = 5, role: str | None = None):
        docs = [
            doc
            for doc in self.documents
            if role is None or doc.metadata.get("role") == role
        ]
        return docs[:k]

    def add_turn(self, turn, *, session_id: str, turn_id: int, timestamp: str) -> None:
        self.adds.append({
            "turn": turn,
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": timestamp,
        })


class C1RoutingEvaluator(BaseEvaluator):
    """Evaluate claim C1 with frozen JSONL cases and deterministic scoring."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        extra_tools: list | None = None,
        history_store: ChatHistoryStore | None = None,
        allow_skips: bool = False,
        turn_runner: TurnRunner | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ):
        self.config = config or AgentConfig()
        self.extra_tools = list(extra_tools or [])
        self.history_store = history_store
        self.allow_skips = allow_skips
        self.turn_runner = turn_runner
        self.progress_cb = progress_cb
        self.available_tools = tool_inventory(self.extra_tools)

    def _emit(self, message: str) -> None:
        """Forward a progress line to the configured callback, if any.

        Progress makes slow model/RAG calls distinguishable from a hang: a live
        run streams case/turn/tool activity instead of going silent.
        """
        if self.progress_cb is not None:
            self.progress_cb(message)

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Load the frozen C1 dev dataset.

        C1 cases are not generated at runtime. The `n` argument is ignored and
        is present only to satisfy the shared evaluator interface.
        """
        del n
        cases = [case.raw for case in load_claim_dataset("c1", "dev")]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(cases, file_obj, ensure_ascii=False, indent=2)
        return cases

    def evaluate(self, cases: list[EvalCase | dict]) -> EvalResult:
        """Run C1 cases through the agent and score the observed tool trace."""
        metric_correct = {key: 0 for key in _METRIC_NAMES}
        metric_total = {key: 0 for key in _METRIC_NAMES}
        routing_correct = 0
        evaluated = 0
        skipped = 0
        details = []
        available = set(self.available_tools)

        for raw_case in cases:
            case = c1_case_to_behavior_case(raw_case)
            self._emit(f"[c1] case start: {case.get('id')}")
            missing = missing_required_tools(case, available)
            if missing:
                if not self.allow_skips:
                    raise MissingRequiredToolsError(
                        f"C1 case {case.get('id')!r} requires unavailable tools: "
                        f"{sorted(missing)}"
                    )
                skipped += 1
                self._emit(f"[c1] case skipped: {case.get('id')} ({sorted(missing)})")
                details.append(_skipped_detail(case, missing))
                continue

            tool_calls, error, step_log, answer = self._run_trace(case)
            actual_tools = [tc["name"] for tc in tool_calls]
            actual_args = [tc["args"] for tc in tool_calls]
            scores = score_c1_trace(raw_case, actual_tools, actual_args, answer=answer)
            case_passed = bool(scores) and all(scores.values())
            routing_correct += case_passed
            evaluated += 1

            for key, correct in scores.items():
                metric_total[key] += 1
                metric_correct[key] += bool(correct)

            case_detail = {
                "id": case.get("id"),
                "category": case.get("category"),
                "question": _first_message(case),
                "actual_tools": actual_tools,
                # Phase-0 instrumentation: keep args + per-step trace so we can
                # tell (a) parallel-call overshoot from (b) uncounted tool
                # messages, and read the actual queries/results behind a runaway.
                "actual_args": actual_args,
                "actual_count": len(actual_tools),
                "max_emitted_per_step": max(
                    (entry.get("n_emitted", 0) for entry in step_log), default=0
                ),
                "scores": scores,
                "passed": case_passed,
                "step_log": step_log,
            }
            # Answer-regex cases (e.g. graceful give-up) are scored on the final
            # text, so keep that text in the detail for auditing the verdict.
            if case.get("expected_answer_regex"):
                case_detail["final_answer"] = answer
            if error:
                case_detail["error"] = error
            self._emit(
                f"[c1] case done: {case.get('id')} "
                f"({len(actual_tools)} tool(s), passed={case_passed})"
            )
            details.append(case_detail)

        scores = {"routing_accuracy": routing_correct / evaluated if evaluated else 0}
        for key, metric_name in _METRIC_NAMES.items():
            if metric_total[key]:
                scores[metric_name] = metric_correct[key] / metric_total[key]

        return EvalResult(
            name="C1Routing",
            total=len(cases),
            scores=scores,
            details=details,
            metadata={
                "claim": "c1",
                "available_tools": self.available_tools,
                "eligible": len(cases) - skipped,
                "evaluated": evaluated,
                "skipped": skipped,
                "allow_skips": self.allow_skips,
                "baseline_eligible": skipped == 0,
            },
        )

    def _run_trace(
        self, case: dict
    ) -> tuple[list[dict], str | None, list[dict], str]:
        case_id = case.get("id")
        messages = list(case.get("messages") or [case["question"]])
        setup_history = case.get("setup_history")
        if self.turn_runner is not None:
            result = self.turn_runner(messages, setup_history)
            # The runner may return just the tool-call list (legacy) or a
            # (tool_calls, final_answer) pair when answer-regex scoring matters.
            if isinstance(result, tuple):
                runner_calls, runner_answer = result
            else:
                runner_calls, runner_answer = result, ""
            return list(runner_calls), None, [], runner_answer

        step_log: list[dict] = []

        def _progress(node_name, new_msgs):
            """Record per-graph-step activity for Phase-0 runaway diagnosis."""
            for msg in new_msgs:
                entry: dict = {"node": node_name, "msg_type": type(msg).__name__}
                emitted = getattr(msg, "tool_calls", None)
                if emitted:
                    names = [
                        call.get("name") if isinstance(call, dict)
                        else getattr(call, "name", None)
                        for call in emitted
                    ]
                    entry["emitted_tool_calls"] = names
                    entry["n_emitted"] = len(emitted)
                    for name in names:
                        self._emit(f"[c1] case {case_id} tool call: {name}")
                if type(msg).__name__ == "ToolMessage":
                    entry["tool_name"] = getattr(msg, "name", None)
                    entry["result_preview"] = str(getattr(msg, "content", ""))[:300]
                    self._emit(
                        f"[c1] case {case_id} tool result: "
                        f"{getattr(msg, 'name', None)}"
                    )
                step_log.append(entry)

        session = ChatSession(
            self.config,
            recursion_limit=32,
            extra_tools=self.extra_tools,
            history_store=(
                _StaticHistoryStore(setup_history)
                if setup_history
                else self.history_store
            ),
            progress_cb=_progress,
        )
        tool_calls: list[dict] = []
        answer = ""
        try:
            for index, message in enumerate(messages, start=1):
                self._emit(f"[c1] case {case_id} turn {index}/{len(messages)} start")
                answer, turn_calls = asyncio.run(session.turn_with_trace(message))
                tool_calls.extend(turn_calls)
        except GraphRecursionError:
            return [], "GraphRecursionError", step_log, ""
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}", step_log, ""
        return tool_calls, None, step_log, answer


def score_c1_trace(
    case: EvalCase | dict,
    actual_tools: list[str],
    actual_args: list[dict],
    answer: str = "",
) -> dict[str, bool]:
    """Score one C1 dataset case against a tool trace.

    ``answer`` is only consulted for cases carrying ``expected_answer_regex``
    (e.g. graceful-give-up cases); other cases score unchanged.
    """
    return score_tool_expectations(
        c1_case_to_behavior_case(case), actual_tools, actual_args, answer=answer
    )


def c1_case_to_behavior_case(case: EvalCase | dict) -> dict:
    """Convert a frozen C1 JSONL case into the legacy behavior scorer shape."""
    raw = case.raw if isinstance(case, EvalCase) else dict(case)
    inputs = dict(raw.get("inputs") or {})
    gold = dict(raw.get("gold") or {})
    messages = list(inputs.get("messages") or [])
    behavior_case = {
        "id": raw.get("id"),
        "category": raw.get("category"),
        **gold,
    }
    if messages:
        behavior_case["messages"] = messages
        if len(messages) == 1:
            behavior_case["question"] = messages[0]
    elif "question" in inputs:
        behavior_case["question"] = inputs["question"]
    if "setup_history" in inputs:
        behavior_case["setup_history"] = inputs["setup_history"]
    return behavior_case


def _skipped_detail(case: dict, missing: set[str]) -> dict:
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": _first_message(case),
        "skipped": True,
        "skip_reason": f"required tools not loaded: {sorted(missing)}",
    }


def _first_message(case: dict) -> str:
    messages = case.get("messages") or []
    if messages:
        return messages[0]
    return case.get("question", "")
