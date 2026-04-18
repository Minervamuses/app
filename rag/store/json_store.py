"""JSON file-based document store."""

import json
from pathlib import Path

from langchain_core.documents import Document

from rag.store.base import BaseStore


class JSONStore(BaseStore):
    """File-based store for BM25 and backup."""

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self._docs: list[dict] = []
        if self.json_path.exists():
            with self.json_path.open("r", encoding="utf-8") as f:
                self._docs = json.load(f)

    def _save(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False, indent=2)

    def add(self, documents: list[Document]) -> None:
        """Add documents to JSON file."""
        for doc in documents:
            self._docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })
        self._save()

    def get(self, pid: str | None = None) -> list[Document]:
        """Retrieve documents, optionally filtered by pid."""
        docs = []
        for entry in self._docs:
            if pid and entry.get("metadata", {}).get("pid") != pid:
                continue
            docs.append(Document(
                page_content=entry["page_content"],
                metadata=entry.get("metadata", {}),
            ))
        return docs

    def delete(self, pid: str) -> None:
        """Delete all documents matching a pid."""
        self._docs = [
            d for d in self._docs
            if d.get("metadata", {}).get("pid") != pid
        ]
        self._save()
