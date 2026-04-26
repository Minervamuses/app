"""Behavior evaluator — tests agent tool selection strategy.

Runs predefined scenario questions through the LangGraph agent and checks
whether the tool call sequence matches expected patterns (first tool choice,
tool count, required tools).
"""

import asyncio
import json
from pathlib import Path

from langgraph.errors import GraphRecursionError

from agent.config import AgentConfig
from agent.session import ChatSession
from agent.history_rag import ChatHistoryStore
from agent.evaluation.base import BaseEvaluator, EvalResult, tool_inventory


class BehaviorEvaluator(BaseEvaluator):
    """Evaluate agent tool-selection behavior against expected patterns."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        extra_tools: list | None = None,
        history_store: ChatHistoryStore | None = None,
    ):
        self.config = config or AgentConfig()
        self.extra_tools = list(extra_tools or [])
        self.history_store = history_store
        self.available_tools = tool_inventory(self.extra_tools)

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Return built-in behavior test cases.

        Behavior cases are hand-crafted scenarios, not LLM-generated.
        The n parameter is ignored. Override by loading your own cases from JSON.

        Args:
            n: Ignored (kept for interface compatibility).
            output_path: Optional path to save cases as JSON.

        Returns:
            List of behavior test case dicts.
        """
        cases = [
            {
                "question": "What's in the knowledge base?",
                "expected_first_tool": "rag_explore",
                "expected_tool_count": {"min": 1, "max": 2},
                "rationale": "Unknown KB structure, should explore first",
            },
            {
                "question": "Show me the available categories and tags.",
                "expected_first_tool": "rag_explore",
                "expected_tool_count": {"min": 1, "max": 1},
                "rationale": "Explicitly asking for KB overview",
            },
            {
                "question": "How does the scoring module work?",
                "expected_first_tool": "rag_search",
                "expected_tool_count": {"min": 1, "max": 4},
                "rationale": "Specific technical question, direct search",
            },
            {
                "question": "What research notes were written in March?",
                "expected_first_tool": "rag_search",
                "expected_filters_include": ["date_from"],
                "expected_tool_count": {"min": 1, "max": 3},
                "rationale": "Clear time + category signal, should filter",
            },
            {
                "messages": [
                    "How does the embedding module work?",
                    "Show me more context around that result.",
                ],
                "expected_tools_include": ["rag_search", "rag_get_context"],
                "expected_tool_count": {"min": 2, "max": 6},
                "rationale": "Multi-turn: first search, then ask for context expansion",
            },
            {
                "question": "Thanks, that's all I needed.",
                "expected_first_tool": None,
                "expected_tool_count": {"min": 0, "max": 0},
                "rationale": "Chitchat, should not call any tool",
            },
            {
                "question": "Hello!",
                "expected_first_tool": None,
                "expected_tool_count": {"min": 0, "max": 0},
                "rationale": "Greeting, no tool needed",
            },
            {
                "question": "Find all Python files related to database models.",
                "expected_first_tool": "rag_search",
                "expected_filters_include": ["file_type"],
                "expected_tool_count": {"min": 1, "max": 3},
                "rationale": "Specific file type + topic, should use file_type filter",
            },
        ]

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)

        return cases

    def evaluate(self, cases: list[dict]) -> EvalResult:
        """Run each case through the agent graph and compare tool behavior.

        Args:
            cases: Behavior test cases (from generate() or loaded from JSON).

        Returns:
            EvalResult with first_tool_accuracy, tool_count_accuracy,
            tools_coverage, and no_tool_accuracy.
        """
        first_tool_correct = 0
        first_tool_total = 0
        count_correct = 0
        coverage_correct = 0
        coverage_total = 0
        no_tool_correct = 0
        no_tool_total = 0
        filter_correct = 0
        filter_total = 0
        details = []

        for case in cases:
            # Support both single-turn ("question") and multi-turn ("messages")
            questions = case.get("messages") or [case["question"]]
            session = ChatSession(
                self.config,
                recursion_limit=32,
                extra_tools=self.extra_tools,
                history_store=self.history_store,
            )
            tool_calls: list[dict] = []

            try:
                for question in questions:
                    _answer, turn_calls = asyncio.run(session.turn_with_trace(question))
                    tool_calls.extend(turn_calls)
            except (GraphRecursionError, Exception):
                tool_calls = []

            actual_tools = [tc["name"] for tc in tool_calls]
            actual_args = [tc["args"] for tc in tool_calls]
            actual_count = len(actual_tools)

            case_detail = {
                "question": case.get("question") or case.get("messages", [""])[0],
                "actual_tools": actual_tools,
                "actual_count": actual_count,
                "scores": {},
            }

            # Check first tool
            expected_first = case.get("expected_first_tool")
            if expected_first is not None:
                first_tool_total += 1
                correct = (actual_tools[0] == expected_first) if actual_tools else False
                first_tool_correct += correct
                case_detail["scores"]["first_tool"] = correct
            elif expected_first is None and "expected_first_tool" in case:
                # Explicitly expected no tool
                no_tool_total += 1
                correct = actual_count == 0
                no_tool_correct += correct
                case_detail["scores"]["no_tool"] = correct

            # Check tool count
            expected_count = case.get("expected_tool_count", {})
            if expected_count:
                in_range = expected_count["min"] <= actual_count <= expected_count["max"]
                count_correct += in_range
                case_detail["scores"]["count_ok"] = in_range

            # Check required tools present
            expected_include = case.get("expected_tools_include", [])
            if expected_include:
                coverage_total += 1
                covered = all(t in actual_tools for t in expected_include)
                coverage_correct += covered
                case_detail["scores"]["tools_covered"] = covered

            # Check expected filter keys in search args
            expected_filters = case.get("expected_filters_include", [])
            if expected_filters:
                search_args = [a for t, a in zip(actual_tools, actual_args) if t == "rag_search"]
                filter_used = any(
                    all(f in args for f in expected_filters)
                    for args in search_args
                ) if search_args else False
                filter_total += 1
                filter_correct += filter_used
                case_detail["scores"]["filters_used"] = filter_used

            details.append(case_detail)

        total = len(cases)

        scores = {
            "first_tool_accuracy": first_tool_correct / first_tool_total if first_tool_total else 0,
            "tool_count_accuracy": count_correct / total if total else 0,
            "no_tool_accuracy": no_tool_correct / no_tool_total if no_tool_total else 0,
        }
        if coverage_total:
            scores["tools_coverage"] = coverage_correct / coverage_total
        if filter_total:
            scores["filter_accuracy"] = filter_correct / filter_total

        return EvalResult(
            name="Behavior",
            total=total,
            scores=scores,
            details=details,
            metadata={"available_tools": self.available_tools},
        )
