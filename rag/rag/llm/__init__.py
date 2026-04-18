"""LLM providers used internally by rag (e.g. tagger).

Agent consumers should not import from here; they have their own copy at
agent.llm to keep the tool boundary clean.
"""

from rag.llm.base import BaseLLM
from rag.llm.ollama import OllamaLLM
from rag.llm.openrouter import OpenRouterLLM

__all__ = ["BaseLLM", "OllamaLLM", "OpenRouterLLM"]
