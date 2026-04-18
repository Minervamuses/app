"""Embedder module — text embedding strategies."""

from rag.embedder.base import BaseEmbedder
from rag.embedder.ollama import OllamaEmbedder

__all__ = ["BaseEmbedder", "OllamaEmbedder"]
