"""Tests for end-to-end evaluation result details."""

from agent.config import AgentConfig
from agent.evaluation.endtoend import EndToEndEvaluator


class _FakeJudge:
    def invoke(self, *args, **kwargs):
        return '{"score": 3, "rationale": "matches"}'


class _FakeSession:
    captured_history_stores: list = []

    def __init__(self, *args, **kwargs):
        type(self).captured_history_stores.append(kwargs.get("history_store"))

    async def turn_with_trace(self, _question):
        return "answer", [
            {"name": "rag_search", "args": {"query": "x"}},
            {"name": "rag_get_context", "args": {"pid": "p", "chunk_id": 1}},
        ]


def test_e2e_records_tool_trace(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.evaluation.endtoend.ChatSession", _FakeSession)

    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator.config = AgentConfig(persist_dir=str(tmp_path))
    evaluator.extra_tools = []
    evaluator.available_tools = ["rag_explore", "rag_search", "rag_get_context", "recall_history"]
    evaluator._judge_llm = _FakeJudge()

    result = evaluator.evaluate([
        {
            "question": "How does it work?",
            "reference_answer": "It works.",
            "question_type": "direct_search",
        }
    ])

    assert result.details[0]["actual_tools"] == ["rag_search", "rag_get_context"]
    assert result.details[0]["actual_tool_count"] == 2
    assert result.scores["avg_score_raw"] == 3


def test_e2e_injects_noop_history_store(monkeypatch, tmp_path):
    _FakeSession.captured_history_stores = []
    monkeypatch.setattr("agent.evaluation.endtoend.ChatSession", _FakeSession)

    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator.config = AgentConfig(persist_dir=str(tmp_path))
    evaluator.extra_tools = []
    evaluator.available_tools = []
    evaluator._judge_llm = _FakeJudge()

    evaluator.evaluate([
        {"question": "q", "reference_answer": "a", "question_type": "direct_search"},
        {"question": "q2", "reference_answer": "a2", "question_type": "direct_search"},
    ])

    stores = _FakeSession.captured_history_stores
    assert len(stores) == 2
    for store in stores:
        assert store is not None
        # Noop store: search returns nothing, add_turn is a no-op.
        assert store.search("anything") == []
        assert store.add_turn(
            object(), session_id="s", turn_id=1, timestamp="t",
        ) is None
