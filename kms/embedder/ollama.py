"""Ollama-based embedder — local embedding via Ollama server."""

from langchain_ollama import OllamaEmbeddings

from kms.config import KMSConfig
from kms.embedder.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Embed text using a local Ollama model (e.g., bge-m3)."""

    def __init__(self, config: KMSConfig):
        self.config = config
        self._embeddings = OllamaEmbeddings(model=config.embed_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self._embeddings.embed_query(text)
