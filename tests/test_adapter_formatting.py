"""Ensure LangChain adapters preserve the expected tool-output format."""

from unittest.mock import patch

from rag.types import Hit


def test_search_adapter_output_format():
    """Search adapter output should remain byte-identical for one hit."""
    from agent.adapters.langchain.search import create_search_tool
    from agent.config import AgentConfig

    hits = [
        Hit(
            pid="p1",
            chunk_id=3,
            text="hello world",
            file_path="foo/bar.py",
            category="source-code",
            file_type=".py",
            folder="foo",
            date=20260101,
            tags=["source-code"],
        ),
    ]

    with patch("agent.adapters.langchain.search.api_search", return_value=hits):
        tool = create_search_tool(AgentConfig())
        out = tool.invoke({"query": "x"})

    expected = "[1] foo/bar.py (category=source-code) (date=20260101) [pid=p1, chunk_id=3]\nhello world"
    assert out == expected


def test_search_adapter_empty():
    """Search adapter should preserve the empty-results string."""
    from agent.adapters.langchain.search import create_search_tool
    from agent.config import AgentConfig

    with patch("agent.adapters.langchain.search.api_search", return_value=[]):
        tool = create_search_tool(AgentConfig())
        out = tool.invoke({"query": "x"})

    assert out == "No results found."
