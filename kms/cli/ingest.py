"""Ingest a project repo into the KMS."""

import argparse
from pathlib import Path

from kms.chunker.token import TokenChunker
from kms.config import KMSConfig
from kms.store.chroma_store import ChromaStore
from kms.store.document_store import DocumentStore
from kms.store.json_store import JSONStore

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
    # Also ingest extensionless files that look like text configs
    if path.name in {"Makefile", "Dockerfile", "Procfile", ".gitignore", ".env.example"}:
        return True
    return False


def _make_pid(file_path: Path, repo_root: Path) -> str:
    """Create a pid from the file's relative path."""
    rel = file_path.relative_to(repo_root)
    return str(rel).replace("/", "--").replace("\\", "--").lower()


def ingest_file(
    file_path: Path,
    pid: str,
    chunker: TokenChunker,
    store: DocumentStore,
) -> int:
    """Ingest a single file. Returns chunk count."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return 0

    if not text.strip():
        return 0

    docs = chunker.chunk(text, pid)
    if docs:
        store.add(docs)
    return len(docs)


def ingest_repo(
    repo_root: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[int, int]:
    """Ingest all text files in the parent repo, excluding app/.

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

    chunker = TokenChunker(config)
    chroma = ChromaStore(config.raw_collection, config)
    json_store = JSONStore(config.raw_json_path())
    store = DocumentStore(chroma, json_store)

    files_ingested = 0
    total_chunks = 0

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        # Skip files inside excluded directories
        parts = file_path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in parts):
            continue

        if not _should_ingest(file_path):
            continue

        pid = _make_pid(file_path, root)
        count = ingest_file(file_path, pid, chunker, store)
        if count > 0:
            files_ingested += 1
            total_chunks += count
            print(f"  {pid} ({count} chunks)")

    return files_ingested, total_chunks


def ingest_single(
    file_path: str,
    pid: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[str, int]:
    """Ingest a single file.

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

    count = ingest_file(path, pid_val, chunker, store)
    return pid_val, count


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
        print(f"Ingesting repo...")
        files, chunks = ingest_repo(repo_root=args.repo)
        print(f"\nDone: {files} files, {chunks} chunks")


if __name__ == "__main__":
    main()
