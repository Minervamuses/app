"""Ingest a project repo into the KMS.

Usage:
    python -m kms.cli.ingest          # Ingest parent repo
    python -m kms.cli.ingest -r /path  # Ingest specific repo
    python -m kms.cli.ingest file.md   # Ingest single file
    python -m kms.cli.ingest -h
"""

import argparse
import json
import re
from pathlib import Path

from langchain_core.documents import Document

from kms.chunker.token import TokenChunker
from kms.config import KMSConfig, GENERAL_COLLECTION, SUMMARY_COLLECTION
from kms.store.chroma_store import ChromaStore
from kms.store.document_store import DocumentStore
from kms.store.json_store import JSONStore
from kms.tagger.llm_tagger import LLMTagger

# Regex for YYYYMMDD date folders (e.g. Research_notes/20260306/)
_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")

# File extensions to ingest as text
TEXT_EXTENSIONS = {
    # Docs
    ".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml",
    # Python
    ".py",
    # Web
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".vue", ".svelte",
    # Config
    ".ini", ".cfg", ".conf", ".env.example",
    # Data
    ".sql", ".sh", ".bash", ".zsh",
    # Other code
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb",
    # PL/SQL (legacy)
    ".pck", ".pkb", ".pks", ".plsql",
}

# Directories to always skip
SKIP_DIRS = {
    "app",          # ourselves
    ".git",
    ".github",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".claude",
    ".opencode",
    ".cursor",
    "volumes",
    "dist",
    "build",
}


def _extract_date(rel_path: str) -> int:
    """Extract date as YYYYMMDD integer from path if a date folder exists.

    Returns 0 if no date folder is found.  Stored as int so ChromaDB
    supports $gte/$lte range queries.
    """
    for part in Path(rel_path).parts:
        m = _DATE_RE.fullmatch(part)
        if m:
            return int(f"{m.group(1)}{m.group(2)}{m.group(3)}")
    return 0


def _sanitize_collection_name(name: str) -> str:
    """Sanitize a directory name into a valid ChromaDB collection name.

    ChromaDB requires: 3-63 chars, starts/ends with alphanumeric,
    only alphanumeric, underscores, hyphens.
    """
    sanitized = name.lower().replace(" ", "_").replace("-", "_")
    # Strip non-alphanumeric/underscore chars
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    # Ensure minimum length
    if len(sanitized) < 3:
        sanitized = sanitized + "___"[:3 - len(sanitized)]
    return sanitized[:63]


def _get_collection(rel_path: str) -> str:
    """Map a file's relative path to its collection — top-level dir name."""
    parts = Path(rel_path).parts
    if not parts or len(parts) == 1:
        return GENERAL_COLLECTION
    return _sanitize_collection_name(parts[0])


def _should_ingest(path: Path) -> bool:
    """Check if a file should be ingested."""
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name in {"Makefile", "Dockerfile", "Procfile", ".gitignore", ".env.example"}:
        return True
    return False


def _get_file_preview(path: Path) -> str:
    """Get the first non-empty line of a file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    return stripped[:120]
        return ""
    except (UnicodeDecodeError, PermissionError):
        return ""


def _collect_folders(root: Path) -> dict[str, list[Path]]:
    """Group ingestable files by their parent directory."""
    folders: dict[str, list[Path]] = {}
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        parts = file_path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        if not _should_ingest(file_path):
            continue
        folder_rel = str(file_path.parent.relative_to(root))
        if folder_rel == ".":
            folder_rel = ""
        folders.setdefault(folder_rel, []).append(file_path)
    return folders


def _tag_folders(folders: dict[str, list[Path]], root: Path, config: KMSConfig) -> dict[str, dict]:
    """Use LLM to tag and summarize each folder.

    Returns dict mapping folder_rel -> {"tags": [...], "summary": "..."}.
    """
    tagger = LLMTagger(config)
    folder_meta: dict[str, dict] = {}

    print(f"Tagging {len(folders)} folders...")
    for folder_rel, files in sorted(folders.items()):
        file_names = [f.name for f in files]
        file_previews = {f.name: _get_file_preview(f) for f in files[:10]}

        folder_display = folder_rel or "(root)"
        meta = tagger.tag(folder_rel or "(project root)", file_names, file_previews)
        collection = _get_collection(folder_rel + "/dummy") if folder_rel else GENERAL_COLLECTION
        folder_meta[folder_rel] = {
            "tags": meta.tags,
            "summary": meta.summary,
            "collection": collection,
        }
        print(f"  [{collection}] {folder_display} -> {meta.tags}")
        print(f"    summary: {meta.summary}")

    return folder_meta


def ingest_repo(
    repo_root: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[int, int]:
    """Ingest all text files in the parent repo into per-directory collections.

    Creates:
    - One ChromaDB collection per top-level directory (auto-discovered)
    - A summaries collection with one entry per folder
    - A JSON backup of all chunks

    Args:
        repo_root: Path to repo root. Defaults to parent of app/.
        config: Pipeline configuration.

    Returns:
        Tuple of (files_ingested, total_chunks).
    """
    config = config or KMSConfig()
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]

    if not root.is_dir():
        raise FileNotFoundError(f"Repo root not found: {root}")

    # Phase 1: Collect and tag folders
    folders = _collect_folders(root)
    folder_meta = _tag_folders(folders, root, config)

    # Save folder metadata for inspection
    meta_path = Path(config.persist_dir) / "folder_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(folder_meta, f, ensure_ascii=False, indent=2)
    print(f"\nFolder metadata saved to {meta_path}")

    # Phase 2: Ingest folder summaries into summaries collection
    summary_store = ChromaStore(SUMMARY_COLLECTION, config)
    summary_docs = []
    for folder_rel, meta in folder_meta.items():
        summary = meta["summary"]
        if not summary:
            continue
        date = _extract_date(folder_rel)
        collection = _get_collection(folder_rel + "/dummy")
        summary_docs.append(Document(
            page_content=summary,
            metadata={
                "folder": folder_rel or "(root)",
                "tags": str(meta["tags"]),
                "collection": collection,
                "date": date,
                "pid": f"summary:{folder_rel or 'root'}",
                "chunk_id": 0,
            },
        ))
    summary_store.add(summary_docs)
    print(f"Ingested {len(summary_docs)} folder summaries into '{SUMMARY_COLLECTION}' collection")

    # Phase 3: Chunk files and store into per-module collections
    chunker = TokenChunker(config)
    json_store = JSONStore(config.raw_json_path())

    # Cache ChromaStore instances per collection
    collection_stores: dict[str, ChromaStore] = {}

    files_ingested = 0
    total_chunks = 0
    collection_counts: dict[str, int] = {}

    print(f"\nIngesting files...")
    for folder_rel, files in sorted(folders.items()):
        meta = folder_meta.get(folder_rel, {"tags": [], "summary": ""})
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            if not text.strip():
                continue

            rel_path = str(file_path.relative_to(root))
            collection_name = _get_collection(rel_path)
            date = _extract_date(rel_path)

            # Get or create ChromaStore for this collection
            if collection_name not in collection_stores:
                collection_stores[collection_name] = ChromaStore(collection_name, config)

            docs = chunker.chunk(text, rel_path)

            # Enrich metadata (minimal — no more purpose/module/tags as filters)
            for doc in docs:
                doc.metadata["file_path"] = rel_path
                doc.metadata["file_type"] = file_path.suffix.lower()
                doc.metadata["folder"] = folder_rel
                doc.metadata["date"] = date

            if docs:
                collection_stores[collection_name].add(docs)
                json_store.add(docs)
                files_ingested += 1
                total_chunks += len(docs)
                collection_counts[collection_name] = collection_counts.get(collection_name, 0) + len(docs)
                print(f"  [{collection_name}] {rel_path} ({len(docs)} chunks)")

    print(f"\nCollection breakdown:")
    for name, count in sorted(collection_counts.items()):
        print(f"  {name}: {count} chunks")

    return files_ingested, total_chunks


def ingest_single(
    file_path: str,
    pid: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[str, int]:
    """Ingest a single file into its module's collection (no LLM tagging).

    Args:
        file_path: Path to the source file.
        pid: Document identifier. Defaults to slugified filename.
        config: Pipeline configuration.

    Returns:
        Tuple of (pid, chunk_count).
    """
    config = config or KMSConfig()
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    pid_val = pid or path.stem.lower().replace(" ", "-").replace("_", "-")

    chunker = TokenChunker(config)
    chroma = ChromaStore(config.raw_collection, config)
    json_store = JSONStore(config.raw_json_path())
    store = DocumentStore(chroma, json_store)

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return pid_val, 0

    if not text.strip():
        return pid_val, 0

    docs = chunker.chunk(text, pid_val)
    if docs:
        store.add(docs)
    return pid_val, len(docs)


def main():
    parser = argparse.ArgumentParser(description="Ingest into the KMS.")
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="File path for single-file ingest. Omit to ingest parent repo.",
    )
    parser.add_argument("-p", "--pid", help="Override pid (single-file mode only)")
    parser.add_argument(
        "-r", "--repo",
        help="Repo root path (repo mode). Defaults to parent of app/.",
    )
    args = parser.parse_args()

    if args.target:
        pid, count = ingest_single(args.target, pid=args.pid)
        print(f"ingested pid={pid}, chunks={count}")
    else:
        print("Ingesting repo...")
        files, chunks = ingest_repo(repo_root=args.repo)
        print(f"\nDone: {files} files, {chunks} chunks")


if __name__ == "__main__":
    main()
