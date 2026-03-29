"""Ingest a project repo into the KMS."""

import argparse
import json
import re
from pathlib import Path

from kms.chunker.token import TokenChunker
from kms.config import KMSConfig
from kms.store.chroma_store import ChromaStore
from kms.store.document_store import DocumentStore
from kms.store.json_store import JSONStore
from kms.tagger.llm_tagger import LLMTagger

# Regex for YYYYMMDD date folders (e.g. Research_notes/20260306/)
_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")

# Map file suffixes / path patterns to a purpose label
_PURPOSE_MAP = [
    (lambda p: "/test" in str(p).lower() or p.name.startswith("test_"), "test"),
    (lambda p: p.suffix.lower() in {".md", ".txt", ".rst"} and "research_notes" in str(p).lower(), "research-log"),
    (lambda p: p.suffix.lower() in {".md", ".txt", ".rst"}, "documentation"),
    (lambda p: p.name in {"Makefile", "Dockerfile", "docker-compose.yaml", "docker-compose.yml",
                          ".gitignore", ".env.example"} or p.suffix.lower() in {".toml", ".yml", ".yaml", ".ini", ".cfg", ".conf"}, "config"),
    (lambda p: p.suffix.lower() in {".sql", ".pck", ".pkb", ".pks", ".plsql"}, "legacy-reference"),
    (lambda p: p.suffix.lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp"}, "implementation"),
]


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


def _extract_module(rel_path: str) -> str:
    """Extract top-level directory as module name."""
    parts = Path(rel_path).parts
    return parts[0] if parts else ""


def _extract_submodule(rel_path: str) -> str:
    """Extract second-level directory as submodule name."""
    parts = Path(rel_path).parts
    return parts[1] if len(parts) > 1 else ""


def _extract_purpose(file_path: Path, rel_path: str) -> str:
    """Classify file purpose from its path and extension."""
    for matcher, label in _PURPOSE_MAP:
        if matcher(file_path):
            return label
    return "other"


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


def _tag_folders(folders: dict[str, list[Path]], root: Path, config: KMSConfig) -> dict[str, list[str]]:
    """Use LLM to tag each folder with hierarchical labels."""
    tagger = LLMTagger(config)
    folder_tags: dict[str, list[str]] = {}

    print(f"Tagging {len(folders)} folders...")
    for folder_rel, files in sorted(folders.items()):
        file_names = [f.name for f in files]
        file_previews = {f.name: _get_file_preview(f) for f in files[:10]}

        folder_display = folder_rel or "(root)"
        tags = tagger.tag(folder_rel or "(project root)", file_names, file_previews)
        folder_tags[folder_rel] = tags
        print(f"  {folder_display} -> {tags}")

    return folder_tags


def ingest_repo(
    repo_root: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[int, int]:
    """Ingest all text files in the parent repo with LLM-assigned layer tags.

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
    folder_tags = _tag_folders(folders, root, config)

    # Save tags for inspection
    tags_path = Path(config.persist_dir) / "folder_tags.json"
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    with tags_path.open("w", encoding="utf-8") as f:
        json.dump(folder_tags, f, ensure_ascii=False, indent=2)
    print(f"\nFolder tags saved to {tags_path}")

    # Phase 2: Chunk and store with enriched metadata
    chunker = TokenChunker(config)
    chroma = ChromaStore(config.raw_collection, config)
    json_store = JSONStore(config.raw_json_path())
    store = DocumentStore(chroma, json_store)

    files_ingested = 0
    total_chunks = 0

    print(f"\nIngesting files...")
    for folder_rel, files in sorted(folders.items()):
        tags = folder_tags.get(folder_rel, [])
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            if not text.strip():
                continue

            rel_path = str(file_path.relative_to(root))
            docs = chunker.chunk(text, rel_path)

            # Enrich metadata
            date = _extract_date(rel_path)
            module = _extract_module(rel_path)
            submodule = _extract_submodule(rel_path)
            purpose = _extract_purpose(file_path, rel_path)

            for doc in docs:
                doc.metadata["file_path"] = rel_path
                doc.metadata["file_type"] = file_path.suffix.lower()
                doc.metadata["folder"] = folder_rel
                doc.metadata["tags"] = tags
                doc.metadata["module"] = module
                doc.metadata["submodule"] = submodule
                doc.metadata["purpose"] = purpose
                doc.metadata["date"] = date

            if docs:
                store.add(docs)
                files_ingested += 1
                total_chunks += len(docs)
                print(f"  {rel_path} ({len(docs)} chunks)")

    return files_ingested, total_chunks


def ingest_single(
    file_path: str,
    pid: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[str, int]:
    """Ingest a single file (no LLM tagging).

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
