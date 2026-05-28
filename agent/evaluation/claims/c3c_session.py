"""C3c normal/extended skill-validation retry integration evaluator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.config import AgentConfig
from agent.evaluation.base import BaseEvaluator, EvalResult
from agent.evaluation.datasets import EvalCase, load_claim_dataset
from agent.graph import build_graph
from agent.session import ChatSession
from agent.skills.validator import validate_skill_output


class C3SessionEvaluator(BaseEvaluator):
    """Evaluate skill validation retry behavior in normal and extended paths."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Load the frozen C3 session-integration dev dataset."""
        del n
        cases = [
            case.raw
            for case in load_claim_dataset("c3", "dev")
            if case.raw.get("task") == "session"
        ]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(cases, file_obj, ensure_ascii=False, indent=2)
        return cases

    def evaluate(self, cases: list[EvalCase | dict]) -> EvalResult:
        details = []
        retry_hits = 0
        final_clean_hits = 0
        for raw_case in cases:
            case = _raw_case(raw_case)
            if case["inputs"]["path"] == "normal":
                prediction = _run_normal_case(self.config, case)
            elif case["inputs"]["path"] == "extended":
                prediction = _run_extended_case(self.config, case)
            else:
                raise ValueError(f"unknown C3c path: {case['inputs']['path']!r}")

            retry_ok = prediction["retry_observed"] == bool(case["gold"]["retry_expected"])
            final_clean_ok = (
                prediction["final_violations_empty"]
                == bool(case["gold"]["final_violations_empty"])
            )
            retry_hits += int(retry_ok)
            final_clean_hits += int(final_clean_ok)
            details.append({
                "id": case["id"],
                "path": case["inputs"]["path"],
                "prediction": prediction,
                "gold": case["gold"],
                "scores": {
                    "retry_ok": retry_ok,
                    "final_clean_ok": final_clean_ok,
                    "passed": retry_ok and final_clean_ok,
                },
            })

        total = len(cases)
        return EvalResult(
            name="C3Session",
            total=total,
            scores={
                "retry_accuracy": retry_hits / total if total else 0.0,
                "final_clean_accuracy": final_clean_hits / total if total else 0.0,
            },
            details=details,
            metadata={"claim": "c3", "subclaim": "session"},
        )


class _SequencedModel:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.invoke_count = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.invoke_count += 1
        return AIMessage(content=self.answers.pop(0))


class _FakeGraph:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls: list[dict] = []

    async def astream(self, state, config=None, stream_mode="updates"):
        del config, stream_mode
        self.calls.append(state)
        answer = self.answers.pop(0)
        yield {"agent": {"messages": [AIMessage(content=answer)]}}


class _NullHistoryStore:
    def search(self, query: str, k: int = 5, role: str | None = None):
        del query, k, role
        return []

    def add_turn(self, turn, *, session_id: str, turn_id: int, timestamp: str) -> None:
        del turn, session_id, turn_id, timestamp


def _run_normal_case(config: AgentConfig, case: dict) -> dict:
    model = _SequencedModel(list(case["inputs"]["model_outputs"]))
    with patch("agent.graph.get_chat_model", lambda _cfg: model):
        graph = build_graph(config, history_store=_NullHistoryStore())
        result = graph.invoke({
            "messages": [HumanMessage(content=case["inputs"]["user_input"])],
            "active_skill": case["inputs"]["active_skill"],
            "task_mode": case["inputs"].get("task_mode", "revision"),
            "allowed_tools": case["inputs"].get("allowed_tools", ["read_file"]),
            "denied_tools": case["inputs"].get("denied_tools", ["bash"]),
        })
    final_answer = str(result["messages"][-1].content)
    final_violations = validate_skill_output(
        active_skill=case["inputs"]["active_skill"],
        text=final_answer,
    )
    attempts = int(result.get("validation_attempts") or 0)
    return {
        "final_answer": final_answer,
        "validation_attempts": attempts,
        "model_invoke_count": model.invoke_count,
        "retry_observed": attempts > 0,
        "final_violations_empty": not final_violations,
    }


def _run_extended_case(config: AgentConfig, case: dict) -> dict:
    graph = _FakeGraph(list(case["inputs"]["model_outputs"]))
    with patch(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None, history_store=None, **kwargs: graph,
    ):
        session = ChatSession(config, history_store=_NullHistoryStore())
    session.active_skill_runtime = SimpleNamespace(
        name=case["inputs"]["active_skill"],
        root=Path("."),
        instructions="# Eval skill",
        pinned_references={},
        task_mode=case["inputs"].get("task_mode", "revision"),
        allowed_tools=frozenset(case["inputs"].get("allowed_tools", ["read_file"])),
        denied_tools=frozenset(case["inputs"].get("denied_tools", ["bash"])),
        tool_policy_active=True,
        context_block=lambda: "[Active skill]\nname: eval",
    )
    result = asyncio.run(session._apply_final_skill_validation(
        user_input=case["inputs"]["user_input"],
        answer=case["inputs"]["initial_answer"],
        new_messages=[],
        tool_calls=[],
        trace_events=[],
    ))
    final_violations = validate_skill_output(
        active_skill=case["inputs"]["active_skill"],
        text=result.answer,
    )
    return {
        "final_answer": result.answer,
        "validation_attempts": len(graph.calls),
        "model_invoke_count": len(graph.calls),
        "retry_observed": bool(graph.calls),
        "final_violations_empty": not final_violations,
    }


def _raw_case(case: EvalCase | dict) -> dict:
    return case.raw if isinstance(case, EvalCase) else dict(case)
