"""Agent-side adapters for connecting the RAG core to frameworks."""

from agent.adapters.langchain import create_rag_tools

__all__ = ["create_rag_tools"]
