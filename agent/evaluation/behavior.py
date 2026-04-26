"""Behavior evaluator — tests agent tool selection strategy.

Runs predefined scenario questions through the LangGraph agent and checks
whether the tool call sequence matches expected patterns (first tool choice,
tool count, required tools).
"""

import asyncio
import json
from pathlib import Path

from langchain_core.documents import Document
from langgraph.errors import GraphRecursionError

from agent.config import AgentConfig
from agent.session import ChatSession
from agent.history_rag import ChatHistoryStore
from agent.evaluation.base import BaseEvaluator, EvalResult, tool_inventory

RAG_TOOL_NAMES = ("rag_explore", "rag_search", "rag_get_context")
HISTORY_TOOL_NAMES = ("recall_history",)
WEB_TOOL_NAMES = (
    "full-web-search",
    "get-web-search-summaries",
    "get-single-web-page-content",
)
ALL_BEHAVIOR_TOOL_NAMES = (*RAG_TOOL_NAMES, *HISTORY_TOOL_NAMES, *WEB_TOOL_NAMES)
TOOL_FAMILIES = {
    "rag": set(RAG_TOOL_NAMES),
    "history": set(HISTORY_TOOL_NAMES),
    "web": set(WEB_TOOL_NAMES),
}
RAG_FORBIDDEN = [*HISTORY_TOOL_NAMES, *WEB_TOOL_NAMES]
HISTORY_FORBIDDEN = [*RAG_TOOL_NAMES, *WEB_TOOL_NAMES]
WEB_FORBIDDEN = [*RAG_TOOL_NAMES, *HISTORY_TOOL_NAMES]


class _StaticHistoryStore:
    """Tiny deterministic store for behavior cases that need hidden history."""

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
                "id": "rag_explore_inventory",
                "category": "rag_explore",
                "question": "What's in the local knowledge base? Show me the available categories and tags.",
                "expected_tool_family": "rag",
                "expected_first_tool": "rag_explore",
                "expected_tools_include": ["rag_explore"],
                "expected_tools_forbidden": RAG_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 2},
                "rationale": "Unknown KB structure should start with rag_explore.",
            },
            {
                "id": "rag_explore_date_range",
                "category": "rag_explore",
                "question": "Give me a high-level overview of what is indexed locally before answering any specific topic.",
                "expected_tool_family": "rag",
                "expected_first_tool": "rag_explore",
                "expected_tools_include": ["rag_explore"],
                "expected_tools_forbidden": RAG_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 2},
                "rationale": "A broad inventory question should use rag_explore, not search or web.",
            },
            {
                "id": "rag_search_scoring",
                "category": "rag_search",
                "question": "How does the scoring module work?",
                "expected_tool_family": "rag",
                "expected_first_tool": "rag_search",
                "expected_tools_include": ["rag_search"],
                "expected_tools_forbidden": RAG_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 4},
                "rationale": "Specific project-internal technical question should use semantic KB search.",
            },
            {
                "id": "rag_search_python_files_filter",
                "category": "rag_search",
                "question": "Find all Python files related to database models in the local knowledge base.",
                "expected_tool_family": "rag",
                "expected_first_tool": "rag_search",
                "expected_tools_include": ["rag_search"],
                "expected_filters_include": ["file_type"],
                "expected_tools_forbidden": RAG_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 3},
                "rationale": "Specific file type plus topic should use rag_search with a file_type filter.",
            },
            {
                "id": "rag_context_embedding_followup",
                "category": "rag_get_context",
                "messages": [
                    "How does the embedding module work?",
                    "Show me more context around the most relevant result.",
                ],
                "expected_tool_family": "rag",
                "expected_first_tool": "rag_search",
                "expected_tools_include": ["rag_search", "rag_get_context"],
                "expected_tools_forbidden": RAG_FORBIDDEN,
                "expected_tool_count": {"min": 2, "max": 6},
                "rationale": "Multi-turn context expansion should search first, then retrieve surrounding chunks.",
            },
            {
                "id": "rag_context_pwm_followup",
                "category": "rag_get_context",
                "messages": [
                    "Find local notes about Step A PWM generation.",
                    "Expand the context around that result so I can see nearby details.",
                ],
                "expected_tool_family": "rag",
                "expected_first_tool": "rag_search",
                "expected_tools_include": ["rag_search", "rag_get_context"],
                "expected_tools_forbidden": RAG_FORBIDDEN,
                "expected_tool_count": {"min": 2, "max": 6},
                "rationale": "A follow-up asking for nearby details should call rag_get_context.",
            },
            {
                "id": "history_codename",
                "category": "recall_history",
                "question": "What deployment codename did I ask you to remember earlier?",
                "setup_history": [
                    {
                        "role": "user",
                        "text": "Remember that the deployment codename is Blue Lantern.",
                    },
                    {
                        "role": "assistant",
                        "text": "I will remember that the deployment codename is Blue Lantern.",
                    },
                ],
                "expected_tool_family": "history",
                "expected_first_tool": "recall_history",
                "expected_tools_include": ["recall_history"],
                "expected_tools_forbidden": HISTORY_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 3},
                "rationale": "The answer is only in persisted chat history, not in the current prompt.",
            },
            {
                "id": "history_chart_color",
                "category": "recall_history",
                "question": "Earlier I told you my preferred chart color. What was it?",
                "setup_history": [
                    {
                        "role": "user",
                        "text": "For future charts, my preferred accent color is teal.",
                    },
                    {
                        "role": "assistant",
                        "text": "Got it. I will use teal as the preferred chart accent color.",
                    },
                ],
                "expected_tool_family": "history",
                "expected_first_tool": "recall_history",
                "expected_tools_include": ["recall_history"],
                "expected_tools_forbidden": HISTORY_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 3},
                "rationale": "The user explicitly references older conversation content.",
            },
            {
                "id": "web_summary_openai_models",
                "category": "get-web-search-summaries",
                "question": "Use web search summaries to check current OpenAI model announcement headlines.",
                "expected_tool_family": "web",
                "expected_first_tool": "get-web-search-summaries",
                "expected_tools_include": ["get-web-search-summaries"],
                "expected_tools_forbidden": WEB_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 3},
                "rationale": "The user asks for current external search summaries.",
            },
            {
                "id": "web_summary_python_release",
                "category": "get-web-search-summaries",
                "question": "Get web search summaries for the latest Python 3.13 release news.",
                "expected_tool_family": "web",
                "expected_first_tool": "get-web-search-summaries",
                "expected_tools_include": ["get-web-search-summaries"],
                "expected_tools_forbidden": WEB_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 3},
                "rationale": "A current-news summary request should use the lightweight web-summary tool.",
            },
            {
                "id": "web_full_langgraph_docs",
                "category": "full-web-search",
                "question": "Search the web and read the full result pages to summarize current LangGraph tool-calling guidance.",
                "expected_tool_family": "web",
                "expected_first_tool": "full-web-search",
                "expected_tools_include": ["full-web-search"],
                "expected_tools_forbidden": WEB_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 4},
                "rationale": "The user asks to read full pages from current external sources.",
            },
            {
                "id": "web_full_chromadb_docs",
                "category": "full-web-search",
                "question": "Use full web search to compare current pages about ChromaDB persistent client setup.",
                "expected_tool_family": "web",
                "expected_first_tool": "full-web-search",
                "expected_tools_include": ["full-web-search"],
                "expected_tools_forbidden": WEB_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 4},
                "rationale": "Full-page comparison should use full-web-search.",
            },
            {
                "id": "web_single_example",
                "category": "get-single-web-page-content",
                "question": "Read https://example.com and summarize what the page says.",
                "expected_tool_family": "web",
                "expected_first_tool": "get-single-web-page-content",
                "expected_tools_include": ["get-single-web-page-content"],
                "expected_tools_forbidden": WEB_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 2},
                "rationale": "A direct URL should be fetched with the single-page tool.",
            },
            {
                "id": "web_single_iana",
                "category": "get-single-web-page-content",
                "question": "Open https://www.iana.org/domains/reserved and tell me what that page is about.",
                "expected_tool_family": "web",
                "expected_first_tool": "get-single-web-page-content",
                "expected_tools_include": ["get-single-web-page-content"],
                "expected_tools_forbidden": WEB_FORBIDDEN,
                "expected_tool_count": {"min": 1, "max": 2},
                "rationale": "A single known URL should use get-single-web-page-content.",
            },
            {
                "id": "no_tool_thanks",
                "category": "no_tool",
                "question": "Thanks, that's all I needed.",
                "expected_first_tool": None,
                "expected_tools_forbidden": list(ALL_BEHAVIOR_TOOL_NAMES),
                "expected_tool_count": {"min": 0, "max": 0},
                "rationale": "Chitchat should not call any tool.",
            },
            {
                "id": "no_tool_greeting",
                "category": "no_tool",
                "question": "Hello!",
                "expected_first_tool": None,
                "expected_tools_forbidden": list(ALL_BEHAVIOR_TOOL_NAMES),
                "expected_tool_count": {"min": 0, "max": 0},
                "rationale": "Greeting should not call any tool.",
            },
        ]

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)

        return cases

    def _history_store_for_case(self, case: dict):
        if case.get("setup_history"):
            return _StaticHistoryStore(case["setup_history"])
        return self.history_store

    def _score_tool_expectations(
        self,
        case: dict,
        actual_tools: list[str],
        actual_args: list[dict],
    ) -> dict[str, bool]:
        """Score one behavior case against its expected tool trace."""
        scores: dict[str, bool] = {}
        actual_count = len(actual_tools)

        expected_first = case.get("expected_first_tool")
        allowed_first = list(case.get("expected_first_tool_in", []))
        if expected_first is not None:
            allowed_first.append(expected_first)

        if allowed_first:
            scores["first_tool"] = (actual_tools[0] in allowed_first) if actual_tools else False
        elif "expected_first_tool" in case:
            # Explicit None means "no tool should be called".
            scores["no_tool"] = actual_count == 0

        expected_count = case.get("expected_tool_count", {})
        if expected_count:
            scores["count_ok"] = expected_count["min"] <= actual_count <= expected_count["max"]

        expected_include = case.get("expected_tools_include", [])
        if expected_include:
            scores["tools_covered"] = all(t in actual_tools for t in expected_include)

        expected_any = case.get("expected_tools_any_of", [])
        if expected_any:
            scores["tools_any"] = any(t in actual_tools for t in expected_any)

        expected_forbidden = case.get("expected_tools_forbidden", [])
        if expected_forbidden:
            scores["forbidden_ok"] = not any(t in actual_tools for t in expected_forbidden)

        expected_family = case.get("expected_tool_family")
        if expected_family:
            family_tools = TOOL_FAMILIES.get(expected_family, set())
            scores["tool_family"] = any(t in family_tools for t in actual_tools)

        expected_filters = case.get("expected_filters_include", [])
        if expected_filters:
            search_args = [a for t, a in zip(actual_tools, actual_args) if t == "rag_search"]
            scores["filters_used"] = (
                any(all(f in args for f in expected_filters) for args in search_args)
                if search_args else False
            )

        return scores

    def evaluate(self, cases: list[dict]) -> EvalResult:
        """Run each case through the agent graph and compare tool behavior.

        Args:
            cases: Behavior test cases (from generate() or loaded from JSON).

        Returns:
            EvalResult with routing metrics and per-case tool traces.
        """
        metric_names = {
            "first_tool": "first_tool_accuracy",
            "count_ok": "tool_count_accuracy",
            "no_tool": "no_tool_accuracy",
            "tools_covered": "tools_coverage",
            "tools_any": "tools_any_accuracy",
            "forbidden_ok": "forbidden_tool_accuracy",
            "tool_family": "tool_family_accuracy",
            "filters_used": "filter_accuracy",
        }
        metric_correct = {key: 0 for key in metric_names}
        metric_total = {key: 0 for key in metric_names}
        routing_correct = 0
        details = []

        for case in cases:
            # Support both single-turn ("question") and multi-turn ("messages")
            questions = case.get("messages") or [case["question"]]
            history_store = self._history_store_for_case(case)
            session = ChatSession(
                self.config,
                recursion_limit=32,
                extra_tools=self.extra_tools,
                history_store=history_store,
            )
            tool_calls: list[dict] = []
            error = None

            try:
                for question in questions:
                    _answer, turn_calls = asyncio.run(session.turn_with_trace(question))
                    tool_calls.extend(turn_calls)
            except GraphRecursionError:
                error = "GraphRecursionError"
                tool_calls = []
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                tool_calls = []

            actual_tools = [tc["name"] for tc in tool_calls]
            actual_args = [tc["args"] for tc in tool_calls]
            actual_count = len(actual_tools)
            scores = self._score_tool_expectations(case, actual_tools, actual_args)
            case_passed = bool(scores) and all(scores.values())
            routing_correct += case_passed

            case_detail = {
                "id": case.get("id"),
                "category": case.get("category"),
                "question": case.get("question") or case.get("messages", [""])[0],
                "actual_tools": actual_tools,
                "actual_count": actual_count,
                "scores": scores,
            }
            if error:
                case_detail["error"] = error

            for key, correct in scores.items():
                metric_total[key] += 1
                metric_correct[key] += bool(correct)

            details.append(case_detail)

        total = len(cases)

        scores = {"routing_accuracy": routing_correct / total if total else 0}
        for key, metric_name in metric_names.items():
            if metric_total[key]:
                scores[metric_name] = metric_correct[key] / metric_total[key]

        return EvalResult(
            name="Behavior",
            total=total,
            scores=scores,
            details=details,
            metadata={"available_tools": self.available_tools},
        )
