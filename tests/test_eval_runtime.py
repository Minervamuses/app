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
        "read_file",
        "bash",
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
        "read_file",
        "bash",
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


def test_eval_cli_runs_thinking_suite(monkeypatch, tmp_path):
    class FakeThinkingReviewerEvaluator:
        def __init__(self, _config):
            pass

        def generate(self, n=0, output_path=None):
            return [{"id": "thinking-case"}]

        def evaluate(self, cases):
            return EvalResult(
                name="ThinkingReviewer",
                total=len(cases),
                scores={"reviewer_detection_accuracy": 1.0},
                details=[],
                metadata={"cases": len(cases)},
            )

    monkeypatch.setattr(
        eval_cli,
        "ThinkingReviewerEvaluator",
        FakeThinkingReviewerEvaluator,
    )

    result = eval_cli._run_suite(
        suite="thinking",
        config=AgentConfig(persist_dir=str(tmp_path)),
        generate_n=None,
        cases_path=None,
        output_dir=str(tmp_path),
        extra_tools=[],
    )

    assert result.name == "ThinkingReviewer"
    assert result.metadata == {"cases": 1}


def test_eval_cli_runs_c1_claim_and_appends_ledger(monkeypatch, tmp_path):
    seen: dict = {}

    class FakeC1RoutingEvaluator:
        def __init__(self, _config, *, extra_tools=None, allow_skips=False):
            seen["extra_tools"] = extra_tools
            seen["allow_skips"] = allow_skips

        def evaluate(self, cases):
            return EvalResult(
                name="C1Routing",
                total=len(cases),
                scores={"routing_accuracy": 1.0},
                details=[{"id": "case-1", "passed": True}],
                metadata={"claim": "c1"},
            )

    monkeypatch.setattr(eval_cli, "C1RoutingEvaluator", FakeC1RoutingEvaluator)

    result = eval_cli._run_claim(
        claim="c1",
        config=AgentConfig(persist_dir=str(tmp_path)),
        split="dev",
        output_dir=str(tmp_path / "eval"),
        extra_tools=[fake_web_fetch],
        allow_skips=True,
    )

    assert result.name == "C1Routing"
    assert seen == {"extra_tools": [fake_web_fetch], "allow_skips": True}

    ledger_path = tmp_path / "eval" / "runs" / "c1.jsonl"
    payload = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["claim"] == "c1"
    assert payload["metadata"]["dataset_id"] == "c1/dev"
    assert payload["metadata"]["claim"] == "c1"
