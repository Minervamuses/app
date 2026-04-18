"""LLM providers used by the agent layer (graph, evaluation).

Kept here so the agent does not reach into rag's internals. rag has its own
provider code for its internal tagger; the agent owns its own copy.
"""

from agent.llm.base import BaseLLM
from agent.llm.ollama import OllamaLLM
from agent.llm.openrouter import OpenRouterLLM, get_chat_model

__all__ = ["BaseLLM", "OllamaLLM", "OpenRouterLLM", "get_chat_model"]
