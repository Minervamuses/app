"""Embedder module — text embedding strategies."""

from kms.embedder.base import BaseEmbedder
from kms.embedder.ollama import OllamaEmbedder

__all__ = ["BaseEmbedder", "OllamaEmbedder"]
