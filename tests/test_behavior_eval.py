"""Tests for behavior evaluation case schema and scoring."""

from agent.evaluation.base import EvalResult
from agent.evaluation.behavior import BehaviorEvaluator


def test_behavior_generate_has_two_cases_per_tool_plus_no_tool(tmp_path):
    evaluator = BehaviorEvaluator()
    cases = evaluator.generate()

    assert len(cases) == 16
    categories = [case["category"] for case in cases]
    assert categories.count("rag_explore") == 2
    assert categories.count("rag_search") == 2
    assert categories.count("rag_get_context") == 2
    assert categories.count("recall_history") == 2
    assert categories.count("get-web-search-summaries") == 2
    assert categories.count("full-web-search") == 2
    assert categories.count("get-single-web-page-content") == 2
    assert categories.count("no_tool") == 2


def test_score_tool_expectations_accepts_first_tool_options():
    evaluator = BehaviorEvaluator()
    case = {
        "expected_first_tool_in": ["get-web-search-summaries", "full-web-search"],
        "expected_tools_any_of": ["get-web-search-summaries", "full-web-search"],
        "expected_tools_forbidden": ["rag_search"],
        "expected_tool_family": "web",
        "expected_tool_count": {"min": 1, "max": 2},
    }

    scores = evaluator._score_tool_expectations(
        case,
        ["full-web-search"],
        [{}],
    )

    assert scores == {
        "first_tool": True,
        "count_ok": True,
        "tools_any": True,
        "forbidden_ok": True,
        "tool_family": True,
    }


def test_score_tool_expectations_rejects_forbidden_tools():
    evaluator = BehaviorEvaluator()
    case = {
        "expected_tools_include": ["recall_history"],
        "expected_tools_forbidden": ["rag_search"],
        "expected_tool_family": "history",
    }

    scores = evaluator._score_tool_expectations(
        case,
        ["recall_history", "rag_search"],
        [{}, {}],
    )

    assert scores["tools_covered"] is True
    assert scores["forbidden_ok"] is False
    assert scores["tool_family"] is True


def test_score_tool_expectations_merges_first_tool_and_first_tool_in():
    evaluator = BehaviorEvaluator()
    case = {
        "expected_first_tool": "rag_search",
        "expected_first_tool_in": ["rag_explore"],
    }

    scores_search = evaluator._score_tool_expectations(case, ["rag_search"], [{}])
    scores_explore = evaluator._score_tool_expectations(case, ["rag_explore"], [{}])
    scores_other = evaluator._score_tool_expectations(case, ["recall_history"], [{}])

    assert scores_search["first_tool"] is True
    assert scores_explore["first_tool"] is True
    assert scores_other["first_tool"] is False


def test_missing_required_tools_flags_hard_requirements():
    available = {"rag_search", "rag_explore"}
    case = {
        "expected_first_tool": "full-web-search",
        "expected_tools_include": ["full-web-search"],
    }
    assert BehaviorEvaluator._missing_required_tools(case, available) == {
        "full-web-search"
    }


def test_missing_required_tools_passes_when_any_of_options_partially_available():
    available = {"rag_search"}
    case = {"expected_first_tool_in": ["rag_search", "rag_explore"]}
    assert BehaviorEvaluator._missing_required_tools(case, available) == set()


def test_missing_required_tools_flags_when_all_any_of_options_absent():
    available = {"rag_search"}
    case = {"expected_tools_any_of": ["full-web-search", "get-web-search-summaries"]}
    missing = BehaviorEvaluator._missing_required_tools(case, available)
    assert missing == {"full-web-search", "get-web-search-summaries"}


def test_evaluate_skips_cases_whose_tools_are_unavailable(tmp_path):
    from agent.config import AgentConfig

    evaluator = BehaviorEvaluator(AgentConfig(persist_dir=str(tmp_path)))
    cases = [
        {
            "id": "needs_web",
            "category": "full-web-search",
            "question": "fetch a page",
            "expected_first_tool": "full-web-search",
            "expected_tools_include": ["full-web-search"],
        },
    ]

    result = evaluator.evaluate(cases)

    assert isinstance(result, EvalResult)
    assert result.total == 1
    assert result.metadata["evaluated"] == 0
    assert result.metadata["skipped"] == 1
    assert result.scores["routing_accuracy"] == 0
    assert result.details[0]["skipped"] is True
    assert "full-web-search" in result.details[0]["skip_reason"]


def test_score_tool_expectations_checks_search_filter_keys():
    evaluator = BehaviorEvaluator()
    case = {"expected_filters_include": ["file_type"]}

    scores = evaluator._score_tool_expectations(
        case,
        ["rag_search"],
        [{"query": "models", "file_type": "py"}],
    )

    assert scores == {"filters_used": True}
