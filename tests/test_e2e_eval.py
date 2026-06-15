"""Tests for end-to-end evaluation result details."""

from agent.config import AgentConfig
from agent.evaluation.endtoend import EndToEndEvaluator, JUDGE_RESPONSE_FORMAT


class _FakeTextResponse:
    content = '  {"score": 3, "rationale": "matches"}  '


class _FakeJudge:
    def invoke(self, messages):
        self.messages = messages
        return _FakeTextResponse()


class _FakeModel:
    def __init__(self):
        self.bound_kwargs = None

    def bind(self, **kwargs):
        self.bound_kwargs = kwargs
        return self


class _FakeSession:
    captured_history_stores: list = []

    def __init__(self, *args, **kwargs):
        type(self).captured_history_stores.append(kwargs.get("history_store"))

    async def turn_with_trace(self, _question):
        return "answer", [
            {"name": "rag_search", "args": {"query": "x"}},
            {"name": "rag_get_context", "args": {"pid": "p", "chunk_id": 1}},
        ]


def test_e2e_initializes_langchain_eval_models(monkeypatch, tmp_path):
    ollama_calls: list[dict] = []
    openrouter_calls: list[dict] = []

    def fake_ollama_factory(config, **kwargs):
        ollama_calls.append({"config": config, **kwargs})
        return _FakeModel()

    def fake_openrouter_factory(config, **kwargs):
        openrouter_calls.append({"config": config, **kwargs})
        return _FakeModel()

    monkeypatch.setattr(
        "agent.evaluation.endtoend.get_ollama_chat_model",
        fake_ollama_factory,
    )
    monkeypatch.setattr(
        "agent.evaluation.endtoend.get_openrouter_chat_model",
        fake_openrouter_factory,
    )
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        gen_llm_model="gen/model",
        judge_llm_model="judge/model",
        filter_llm_model="filter:model",
    )

    evaluator = EndToEndEvaluator(cfg)

    assert ollama_calls[0]["model_name"] == "filter:model"
    assert ollama_calls[0]["max_tokens"] == 8
    assert ollama_calls[0]["temperature"] == 0.0
    assert openrouter_calls[0]["model_name"] == "gen/model"
    assert openrouter_calls[0]["max_tokens"] == 4096
    assert openrouter_calls[1]["model_name"] == "judge/model"
    assert openrouter_calls[1]["max_tokens"] == 300
    assert evaluator._judge_llm.bound_kwargs == {
        "response_format": JUDGE_RESPONSE_FORMAT,
    }


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
