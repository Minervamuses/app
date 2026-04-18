"""Ingest a project repo into the KMS.

Usage:
    python -m rag.cli.ingest          # Ingest parent repo
    python -m rag.cli.ingest -r /path  # Ingest specific repo
    python -m rag.cli.ingest file.md   # Ingest single file
    python -m rag.cli.ingest -h
"""

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from rag.chunker.token import TokenChunker
from rag.config import KMSConfig, KNOWLEDGE_COLLECTION
from rag.store.chroma_store import ChromaStore
from rag.store.document_store import DocumentStore
from rag.store.json_store import JSONStore
from rag.tagger.llm_tagger import LLMTagger
from rag.utils.paths import extract_date

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
        folder_meta[folder_rel] = {
            "tags": meta.tags,
            "summary": meta.summary,
        }
        print(f"  {folder_display} -> {meta.tags}")
        print(f"    summary: {meta.summary}")

    return folder_meta


def ingest_repo(
    repo_root: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[int, int]:
    """Ingest all text files into a single 'knowledge' collection with rich metadata.

    Creates:
    - One ChromaDB collection ('knowledge') with category/tags metadata per chunk
    - A JSON backup of all chunks
    - folder_meta.json with per-folder tags and summaries

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

    # Save folder metadata
    meta_path = Path(config.folder_meta_path())
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(folder_meta, f, ensure_ascii=False, indent=2)
    print(f"\nFolder metadata saved to {meta_path}")

    # Phase 2: Chunk + write to single collection
    chunker = TokenChunker(config)
    json_store = JSONStore(config.raw_json_path())
    chroma = ChromaStore(KNOWLEDGE_COLLECTION, config)

    files_ingested = 0
    total_chunks = 0

    print(f"\nIngesting files...")
    for folder_rel, files in sorted(folders.items()):
        meta = folder_meta.get(folder_rel, {"tags": [], "summary": ""})
        tags = meta.get("tags", [])
        category = tags[0] if tags else "unknown"

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            if not text.strip():
                continue

            rel_path = str(file_path.relative_to(root))
            date = extract_date(rel_path)

            docs = chunker.chunk(text, rel_path)

            for doc in docs:
                doc.metadata["file_path"] = rel_path
                doc.metadata["file_type"] = file_path.suffix.lower()
                doc.metadata["folder"] = folder_rel
                doc.metadata["date"] = date
                doc.metadata["category"] = category
                doc.metadata["tags"] = json.dumps(tags)

            if docs:
                chroma.add(docs)
                json_store.add(docs)
                files_ingested += 1
                total_chunks += len(docs)
                print(f"  [{category}] {rel_path} ({len(docs)} chunks)")

    return files_ingested, total_chunks


def ingest_single(
    file_path: str,
    pid: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[str, int]:
    """Ingest a single file into the knowledge collection (no LLM tagging).

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
    chroma = ChromaStore(KNOWLEDGE_COLLECTION, config)
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
