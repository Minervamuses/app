"""Ingest a document into the KMS."""

import argparse
from pathlib import Path

from kms.chunker.token import TokenChunker
from kms.config import KMSConfig
from kms.store.chroma_store import ChromaStore
from kms.store.document_store import DocumentStore
from kms.store.json_store import JSONStore


def ingest(
    file_path: str,
    pid: str | None = None,
    config: KMSConfig | None = None,
) -> tuple[str, int]:
    """Ingest a document: read, chunk, embed, store.

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
    text = path.read_text(encoding="utf-8")

    chunker = TokenChunker(config)
    docs = chunker.chunk(text, pid_val)

    chroma = ChromaStore(config.raw_collection, config)
    json_store = JSONStore(config.raw_json_path())
    store = DocumentStore(chroma, json_store)
    store.add(docs)

    return pid_val, len(docs)


def main():
    parser = argparse.ArgumentParser(description="Ingest a document into the KMS.")
    parser.add_argument("file_path", help="Path to the source file")
    parser.add_argument("-p", "--pid", help="Override pid; default from filename")
    args = parser.parse_args()

    pid, count = ingest(args.file_path, pid=args.pid)
    print(f"ingested pid={pid}, chunks={count}")


if __name__ == "__main__":
    main()
