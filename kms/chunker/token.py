"""Token-based chunker — GraphRAG-style sliding window over tokens."""

import tiktoken
from langchain.schema import Document

from kms.chunker.base import BaseChunker
from kms.config import KMSConfig


class TokenChunker(BaseChunker):
    """Split text using a token-based sliding window.

    Encodes text into tokens using tiktoken, then creates chunks
    of fixed token size with overlap. Format-agnostic — works on
    any text (code, markdown, meeting notes, papers).
    """

    def __init__(self, config: KMSConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self._enc = tiktoken.get_encoding(config.encoding_model)

    def chunk(self, text: str, pid: str) -> list[Document]:
        """Split text into token-windowed chunks.

        Args:
            text: The full document text to chunk.
            pid: Paper/document identifier for metadata.

        Returns:
            List of Documents with page_content and metadata.
        """
        tokens = self._enc.encode(text)
        if not tokens:
            return []

        documents: list[Document] = []
        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self._enc.decode(tokens[start:end])

            documents.append(Document(
                page_content=chunk_text,
                metadata={
                    "pid": pid,
                    "chunk_id": chunk_id,
                    "mode": "raw",
                    "token_count": end - start,
                },
            ))

            chunk_id += 1
            if end == len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap

        return documents
