"""Retriever module — search strategies."""

from rag.retriever.base import BaseRetriever
from rag.retriever.vector import VectorRetriever

__all__ = ["BaseRetriever", "VectorRetriever"]
