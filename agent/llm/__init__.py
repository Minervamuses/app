"""LangChain chat-model access used by the agent layer.

Kept here so the agent does not reach into rag's internals. rag has its own
provider code for its internal tagger; the agent owns its own copy.
"""

from agent.llm.ollama import get_ollama_chat_model
from agent.llm.openrouter import get_chat_model, get_openrouter_chat_model
from agent.llm.text import invoke_text

__all__ = [
    "get_chat_model",
    "get_ollama_chat_model",
    "get_openrouter_chat_model",
    "invoke_text",
]
