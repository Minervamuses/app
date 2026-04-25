"""Tests for the recall_history StructuredTool factory."""

import json

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool

from agent.config import AgentConfig
from agent.history_rag.tool import create_history_tool


class _FakeStore:
    def __init__(self, results=None):
        self.results = results or []
        self.calls: list[dict] = []

    def search(self, query, k=5, role=None):
        self.calls.append({"query": query, "k": k, "role": role})
        return self.results


def _make_doc(role: str, text: str, turn_id: int, timestamp: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "role": role,
            "turn_id": turn_id,
            "session_id": "sess-x",
            "timestamp": timestamp,
        },
    )


def test_create_history_tool_returns_structured_tool(tmp_path):
    fake = _FakeStore()
    tool = create_history_tool(AgentConfig(persist_dir=str(tmp_path)), store=fake)

    assert isinstance(tool, StructuredTool)
    assert tool.name == "recall_history"


def test_recall_history_passes_query_and_role_to_store(tmp_path):
    fake = _FakeStore()
    tool = create_history_tool(AgentConfig(persist_dir=str(tmp_path)), store=fake)

    tool.invoke({"query": "earlier discussion", "k": 3, "role": "user"})

    assert fake.calls == [{"query": "earlier discussion", "k": 3, "role": "user"}]


def test_recall_history_omits_role_when_unspecified(tmp_path):
    fake = _FakeStore()
    tool = create_history_tool(AgentConfig(persist_dir=str(tmp_path)), store=fake)

    tool.invoke({"query": "anything"})

    assert fake.calls[0]["role"] is None


def test_recall_history_renders_documents_as_json(tmp_path):
    fake = _FakeStore(
        results=[
            _make_doc("user", "How does X work?", 4, "2026-04-25T10:00"),
            _make_doc("assistant", "X works by ...", 4, "2026-04-25T10:00"),
        ]
    )
    tool = create_history_tool(AgentConfig(persist_dir=str(tmp_path)), store=fake)

    out = tool.invoke({"query": "X"})

    payload = json.loads(out)
    assert payload == [
        {
            "role": "user",
            "text": "How does X work?",
            "turn_id": 4,
            "timestamp": "2026-04-25T10:00",
        },
        {
            "role": "assistant",
            "text": "X works by ...",
            "turn_id": 4,
            "timestamp": "2026-04-25T10:00",
        },
    ]


def test_recall_history_empty_store_returns_empty_json_array(tmp_path):
    fake = _FakeStore(results=[])
    tool = create_history_tool(AgentConfig(persist_dir=str(tmp_path)), store=fake)

    out = tool.invoke({"query": "nothing here"})
    assert json.loads(out) == []


def test_role_filter_only_accepts_user_or_assistant(tmp_path):
    """Pydantic validation should reject other strings."""
    import pytest
    from pydantic import ValidationError

    fake = _FakeStore()
    tool = create_history_tool(AgentConfig(persist_dir=str(tmp_path)), store=fake)

    with pytest.raises(ValidationError):
        tool.invoke({"query": "x", "role": "system"})
