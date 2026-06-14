"""Tests for skill output validation."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.graph import build_graph
from agent.skills.validator import validate_skill_output


@tool("rag_explore")
def _rag_explore() -> str:
    """Explore."""
    return "explore"


@tool("rag_search")
def _rag_search(query: str) -> str:
    """Search."""
    return query


@tool("rag_get_context")
def _rag_get_context(pid: str, chunk_id: int) -> str:
    """Context."""
    return f"{pid}:{chunk_id}"


@tool("recall_history")
def _recall_history(query: str) -> str:
    """Recall."""
    return query


def _patch_graph_tools(monkeypatch, model):
    monkeypatch.setattr("agent.graph.get_chat_model", lambda _cfg: model)
    monkeypatch.setattr(
        "agent.tools.inventory.create_rag_tools",
        lambda _cfg: [_rag_explore, _rag_search, _rag_get_context],
    )
    monkeypatch.setattr(
        "agent.tools.inventory.create_history_tool",
        lambda _cfg, store=None: _recall_history,
    )


class _SequencedModel:
    def __init__(self, answers):
        self.answers = list(answers)
        self.invoke_count = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.invoke_count += 1
        return AIMessage(content=self.answers.pop(0))


def test_validate_skill_output_catches_uncited_percentage():
    violations = validate_skill_output(
        active_skill="academic-paper-writing",
        text="The intervention improved retention by 42%.",
    )

    assert violations == [
        "Quantitative claims with percentages need a supplied source, citation marker, or explicit placeholder."
    ]


def test_validate_skill_output_allows_percentage_with_citation_marker():
    violations = validate_skill_output(
        active_skill="academic-paper-writing",
        text="The intervention improved retention by 42% [source].",
    )

    assert violations == []


def test_validate_skill_output_skips_unregistered_skill():
    violations = validate_skill_output(
        active_skill="other-skill",
        text="The intervention improved retention by 42%.",
    )

    assert violations == []


def test_skill_validator_retries_once_then_accepts_clean_revision(monkeypatch, tmp_path):
    model = _SequencedModel([
        "The intervention improved retention by 42%.",
        "The intervention improved retention by 42% [source].",
    ])
    _patch_graph_tools(monkeypatch, model)
    cfg = AgentConfig(persist_dir=str(tmp_path), skill_max_validation_retries=1)
    graph = build_graph(cfg)

    result = graph.invoke({
        "messages": [HumanMessage(content="revise this abstract")],
        "active_skill": "academic-paper-writing",
        "task_mode": "revision",
        "allowed_tools": ["read_file"],
        "denied_tools": ["bash"],
    })

    assert model.invoke_count == 2
    assert result["messages"][-1].content == "The intervention improved retention by 42% [source]."
    assert result["validation_attempts"] == 1
    assert result["validation_errors"] == []


def test_skill_validator_skips_no_skill_path(monkeypatch, tmp_path):
    model = _SequencedModel(["The intervention improved retention by 42%."])
    _patch_graph_tools(monkeypatch, model)
    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg)

    result = graph.invoke({
        "messages": [HumanMessage(content="revise this abstract")],
    })

    assert model.invoke_count == 1
    assert result["messages"][-1].content == "The intervention improved retention by 42%."
    assert "validation_attempts" not in result
