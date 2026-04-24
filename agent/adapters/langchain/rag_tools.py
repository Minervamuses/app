"""LangChain bridge for rag's framework-neutral tool contract."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from rag import TOOL_SCHEMAS, dispatch

from agent.config import AgentConfig

DEFAULT_RAG_TOOL_NAMES = ("rag_explore", "rag_search", "rag_get_context")


def _render_tool_result(value: Any) -> str:
    """Render dispatch output as stable JSON for LLM tool messages."""
    return json.dumps(value, ensure_ascii=False)


def _invoke_rag_tool(tool_name: str, config: AgentConfig, **kwargs: Any) -> str:
    result = dispatch(tool_name, kwargs, config=config)
    return _render_tool_result(result)


def _make_rag_runner(tool_name: str, config: AgentConfig):
    """Return a real function for LangGraph type-hint inspection."""

    def _run(**kwargs: Any) -> str:
        return _invoke_rag_tool(tool_name, config, **kwargs)

    _run.__name__ = tool_name
    return _run


def create_rag_tools(
    config: AgentConfig,
    *,
    include_list_chunks: bool = False,
) -> list[StructuredTool]:
    """Create LangChain tools from rag.TOOL_SCHEMAS.

    Chat binds the interactive retrieval tools by default. ``rag_list_chunks``
    is intentionally opt-in because an unfiltered call can dump the entire raw
    JSON backup into the prompt.
    """
    names = list(DEFAULT_RAG_TOOL_NAMES)
    if include_list_chunks:
        names.append("rag_list_chunks")

    tools: list[StructuredTool] = []
    schemas_by_name = {schema["name"]: schema for schema in TOOL_SCHEMAS}
    for name in names:
        schema = schemas_by_name[name]
        runner = _make_rag_runner(name, config)
        tools.append(
            StructuredTool.from_function(
                func=runner,
                name=name,
                description=schema["description"],
                args_schema=schema["input_schema"],
                infer_schema=False,
            )
        )
    return tools
