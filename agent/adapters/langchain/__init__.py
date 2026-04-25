"""LangChain adapter that wraps `rag.api` into `@tool` callables."""

from agent.adapters.langchain.rag_tools import create_rag_tools

__all__ = ["create_rag_tools"]
