"""Tests for the C1 routing claim evaluator."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.evaluation.behavior import BehaviorEvaluator
from agent.evaluation.claims import c1_routing
from agent.evaluation.claims.c1_routing import (
    C1RoutingEvaluator,
    MissingRequiredToolsError,
    c1_case_to_behavior_case,
    score_c1_trace,
)
from agent.evaluation.datasets import load_claim_dataset


def _give_up_case():
    """Return the reclassified graceful-give-up dev case (embedding module)."""
    case = next(
        c for c in load_claim_dataset("c1", "dev")
        if c.id == "rag_context_embedding_followup"
    )
    assert case.raw["category"] == "rag_graceful_give_up"
    return case


_GIVE_UP_ANSWER = (
    "I searched the indexed knowledge base but it does not contain enough "
    "evidence about the embedding module to answer."
)


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


def test_graceful_give_up_scoring_passes_for_bounded_search_and_not_found():
    case = _give_up_case()

    scores = score_c1_trace(
        case,
        ["rag_search", "rag_search"],
        [{"query": "embedding module"}, {"query": "embedder"}],
        answer=_GIVE_UP_ANSWER,
    )

    # Bounded search + honest not-found answer satisfies every expectation.
    assert scores["first_tool"] is True
    assert scores["count_ok"] is True
    assert scores["forbidden_ok"] is True
    assert scores["tools_covered"] is True
    assert scores["tool_family"] is True
    assert scores["answer_ok"] is True
    assert all(scores.values())


def test_graceful_give_up_scoring_fails_on_rag_get_context():
    case = _give_up_case()

    scores = score_c1_trace(
        case,
        ["rag_search", "rag_get_context"],
        [{"query": "embedding module"}, {"pid": "x", "chunk_id": 1}],
        answer=_GIVE_UP_ANSWER,
    )

    # rag_get_context on irrelevant results is now forbidden for this case.
    assert scores["forbidden_ok"] is False


def test_graceful_give_up_scoring_fails_on_too_many_searches():
    case = _give_up_case()

    scores = score_c1_trace(
        case,
        ["rag_search"] * 4,
        [{"query": "embedding"}] * 4,
        answer=_GIVE_UP_ANSWER,
    )

    # Four searches exceeds the 1-3 bound: this is the runaway we reject.
    assert scores["count_ok"] is False


def test_graceful_give_up_scoring_fails_when_answer_is_not_a_give_up():
    case = _give_up_case()

    scores = score_c1_trace(
        case,
        ["rag_search"],
        [{"query": "embedding"}],
        answer="The embedding module uses bge-m3 via Ollama to produce vectors.",
    )

    # A confident answer with no not-found cue must fail answer scoring.
    assert scores["answer_ok"] is False


def test_c1_evaluator_records_final_answer_for_answer_regex_case(tmp_path):
    case = {
        "id": "give_up_case",
        "claim": "c1",
        "split": "dev",
        "category": "rag_graceful_give_up",
        "inputs": {"messages": ["How does the embedding module work?"]},
        "gold": {
            "expected_first_tool_in": ["rag_search", "rag_explore"],
            "expected_tools_include": ["rag_search"],
            "expected_tools_forbidden": ["rag_get_context"],
            "expected_tool_count": {"min": 1, "max": 3},
            "expected_answer_regex": [
                "(?i)knowledge base",
                "(?i)not contain",
            ],
        },
        "provenance": {"source": "test", "labeler": "test", "date": "2026-06-15"},
    }

    def turn_runner(messages, setup_history):
        assert messages == ["How does the embedding module work?"]
        return [{"name": "rag_search", "args": {"query": "embedding"}}], _GIVE_UP_ANSWER

    evaluator = C1RoutingEvaluator(
        AgentConfig(persist_dir=str(tmp_path)),
        turn_runner=turn_runner,
    )

    result = evaluator.evaluate([case])

    assert result.scores["answer_accuracy"] == 1.0
    assert result.details[0]["final_answer"] == _GIVE_UP_ANSWER
    assert result.details[0]["scores"]["answer_ok"] is True
    assert result.details[0]["passed"] is True


def test_c1_evaluator_emits_case_turn_and_tool_progress(monkeypatch, tmp_path):
    case = load_claim_dataset("c1", "dev")[1]  # rag_search_scoring (single turn)

    class _FakeSession:
        def __init__(self, _config, *, recursion_limit, extra_tools,
                     history_store, progress_cb):
            self._progress_cb = progress_cb

        async def turn_with_trace(self, message):
            ai = AIMessage(
                content="",
                tool_calls=[{"name": "rag_search", "args": {"query": "x"}, "id": "c1"}],
            )
            tool_msg = ToolMessage(
                content="no relevant results",
                name="rag_search",
                tool_call_id="c1",
            )
            self._progress_cb("agent", [ai])
            self._progress_cb("tools", [tool_msg])
            return "answer", [{"name": "rag_search", "args": {"query": "x"}, "id": "c1"}]

    monkeypatch.setattr(c1_routing, "ChatSession", _FakeSession)

    events: list[str] = []
    evaluator = C1RoutingEvaluator(
        AgentConfig(persist_dir=str(tmp_path)),
        progress_cb=events.append,
    )

    evaluator.evaluate([case])

    joined = "\n".join(events)
    assert any("case start: rag_search_scoring" in e for e in events)
    assert "turn 1/1 start" in joined
    assert "tool call: rag_search" in joined
    assert "tool result: rag_search" in joined
    assert any("case done: rag_search_scoring" in e for e in events)
