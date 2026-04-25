"""LangChain tool factory for chat-history retrieval.

Exposes a single `recall_history` tool that the agent can call when the
user references content that has aged out of the recent_turns window.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.config import AgentConfig
from agent.history_rag.store import ChatHistoryStore, get_chat_history_store

TOOL_NAME = "recall_history"
TOOL_DESCRIPTION = (
    "Search older parts of this conversation that have aged out of the visible "
    "recent context. Each evicted user prompt and assistant response is stored "
    "separately, so results may include either a question you saw earlier or an "
    "answer you produced earlier. Use this when the user references content from "
    "earlier that is no longer in the current prompt. Do NOT use this for "
    "general knowledge questions or for content already visible in this turn."
)


class RecallHistoryInput(BaseModel):
    """Input schema for the recall_history tool."""

    query: str = Field(description="Semantic search query over evicted chat turns.")
    k: int = Field(5, description="Number of results to return (default 5).")
    role: Literal["user", "assistant"] | None = Field(
        None,
        description="Filter to only user prompts ('user') or only assistant responses ('assistant'). Omit for both.",
    )


def _format_results(documents) -> str:
    payload = []
    for doc in documents:
        meta = doc.metadata or {}
        payload.append(
            {
                "role": meta.get("role"),
                "text": doc.page_content,
                "turn_id": meta.get("turn_id"),
                "timestamp": meta.get("timestamp"),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def create_history_tool(
    config: AgentConfig,
    store: ChatHistoryStore | None = None,
) -> StructuredTool:
    """Build the recall_history tool. `store` is injectable for tests."""
    if store is None:
        store = get_chat_history_store(config)

    def _run(query: str, k: int = 5, role: str | None = None) -> str:
        documents = store.search(query, k=k, role=role)
        return _format_results(documents)

    _run.__name__ = TOOL_NAME

    return StructuredTool.from_function(
        func=_run,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=RecallHistoryInput,
        infer_schema=False,
    )
