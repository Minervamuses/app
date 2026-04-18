"""Framework-neutral data types for the KMS public API."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hit:
    """A single retrieval hit."""

    pid: str
    chunk_id: int
    text: str
    file_path: str
    category: str | None = None
    file_type: str | None = None
    folder: str | None = None
    date: int = 0
    tags: list[str] = field(default_factory=list)
    score: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FolderSummary:
    """Summary metadata for one folder in the knowledge base."""

    folder: str
    category: str
    tags: list[str]
    summary: str


@dataclass(frozen=True)
class Inventory:
    """Summary of what the knowledge base contains."""

    categories: dict[str, int]
    tags: list[str]
    date_range: tuple[int, int] | None
    folders: list[FolderSummary]


@dataclass(frozen=True)
class ContextChunk:
    """A single chunk in a document context window."""

    chunk_id: int
    text: str
    is_target: bool


@dataclass(frozen=True)
class ContextWindow:
    """A chunk plus its neighbours within the same document."""

    pid: str
    target_chunk_id: int
    chunks: list[ContextChunk]
    total_chunks_in_doc: int
