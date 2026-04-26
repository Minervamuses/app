"""Tests for evaluation runtime wiring."""

import json

from langchain_core.tools import tool

from agent.cli import eval as eval_cli
from agent.config import AgentConfig
from agent.evaluation.base import EvalResult, tool_inventory
from agent.evaluation.behavior import BehaviorEvaluator


@tool("web_fetch")
def fake_web_fetch(url: str) -> str:
    """Fake MCP tool."""
    return url


def test_tool_inventory_includes_base_and_extra_tools():
    assert tool_inventory([fake_web_fetch]) == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
        "web_fetch",
    ]


def test_behavior_evaluator_records_available_tools(tmp_path):
    evaluator = BehaviorEvaluator(
        AgentConfig(persist_dir=str(tmp_path)),
        extra_tools=[fake_web_fetch],
    )

    assert evaluator.available_tools == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
        "web_fetch",
    ]


def test_eval_cli_passes_extra_tools_and_saves_metadata(monkeypatch, tmp_path):
    seen: dict = {}

    class FakeBehaviorEvaluator:
        def __init__(self, _config, *, extra_tools=None):
            seen["extra_tools"] = extra_tools

        def generate(self, n=0, output_path=None):
            return [{"question": "hi"}]

        def evaluate(self, cases):
            return EvalResult(
                name="Behavior",
                total=len(cases),
                scores={"tool_count_accuracy": 1.0},
                details=[],
                metadata={"available_tools": [tool.name for tool in seen["extra_tools"]]},
            )

    monkeypatch.setattr(eval_cli, "BehaviorEvaluator", FakeBehaviorEvaluator)

    result = eval_cli._run_suite(
        suite="behavior",
        config=AgentConfig(persist_dir=str(tmp_path)),
        generate_n=None,
        cases_path=None,
        output_dir=str(tmp_path),
        extra_tools=[fake_web_fetch],
    )

    assert result.metadata == {"available_tools": ["web_fetch"]}
    assert seen["extra_tools"] == [fake_web_fetch]

    result_files = list(tmp_path.glob("behavior_results_*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["metadata"] == {"available_tools": ["web_fetch"]}
