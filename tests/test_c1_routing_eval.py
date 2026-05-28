"""Tests for the C1 routing claim evaluator."""

import pytest
from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.evaluation.behavior import BehaviorEvaluator
from agent.evaluation.claims.c1_routing import (
    C1RoutingEvaluator,
    MissingRequiredToolsError,
    c1_case_to_behavior_case,
    score_c1_trace,
)
from agent.evaluation.datasets import load_claim_dataset


@tool("full-web-search")
def fake_full_web_search(query: str) -> str:
    """Fake full web search."""
    return query


def test_c1_datasets_load_from_repo_root():
    dev = load_claim_dataset("c1", "dev")
    test = load_claim_dataset("c1", "test")

    assert len(dev) == 8
    assert len(test) == 8
    assert {case.split for case in dev} == {"dev"}
    assert {case.split for case in test} == {"test"}


def test_c1_dataset_cases_match_legacy_behavior_case_ids():
    frozen_ids = {
        case.id
        for split in ("dev", "test")
        for case in load_claim_dataset("c1", split)
    }
    legacy_ids = {case["id"] for case in BehaviorEvaluator().generate()}

    assert frozen_ids == legacy_ids


def test_score_c1_trace_matches_legacy_scorer_for_frozen_case():
    frozen = load_claim_dataset("c1", "dev")[1]
    behavior_case = c1_case_to_behavior_case(frozen)
    actual_tools = ["rag_explore", "rag_search"]
    actual_args = [{}, {"query": "scoring"}]
    legacy = BehaviorEvaluator()._score_tool_expectations(
        behavior_case,
        actual_tools,
        actual_args,
    )

    assert score_c1_trace(frozen, actual_tools, actual_args) == legacy


def test_c1_evaluator_fails_fast_on_missing_required_tools_by_default(tmp_path):
    web_case = load_claim_dataset("c1", "dev")[4]
    evaluator = C1RoutingEvaluator(AgentConfig(persist_dir=str(tmp_path)))

    with pytest.raises(MissingRequiredToolsError, match="unavailable tools"):
        evaluator.evaluate([web_case])


def test_c1_evaluator_can_skip_missing_required_tools_for_dev_runs(tmp_path):
    web_case = load_claim_dataset("c1", "dev")[4]
    evaluator = C1RoutingEvaluator(
        AgentConfig(persist_dir=str(tmp_path)),
        allow_skips=True,
    )

    result = evaluator.evaluate([web_case])

    assert result.metadata["eligible"] == 0
    assert result.metadata["evaluated"] == 0
    assert result.metadata["skipped"] == 1
    assert result.metadata["baseline_eligible"] is False
    assert result.details[0]["skipped"] is True


def test_c1_evaluator_scores_trace_from_injected_runner(tmp_path):
    case = load_claim_dataset("c1", "dev")[1]

    def turn_runner(messages, setup_history):
        assert messages == ["How does the scoring module work?"]
        assert setup_history is None
        return [
            {"name": "rag_explore", "args": {}},
            {"name": "rag_search", "args": {"query": "scoring"}},
        ]

    evaluator = C1RoutingEvaluator(
        AgentConfig(persist_dir=str(tmp_path)),
        turn_runner=turn_runner,
    )

    result = evaluator.evaluate([case])

    assert result.name == "C1Routing"
    assert result.scores["routing_accuracy"] == 1.0
    assert result.metadata["eligible"] == 1
    assert result.metadata["baseline_eligible"] is True
    assert result.details[0]["passed"] is True
