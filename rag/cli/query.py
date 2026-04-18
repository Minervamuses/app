"""Interactive query interface for the KMS."""

import argparse

from rag.config import KMSConfig, KNOWLEDGE_COLLECTION
from rag.retriever.vector import VectorRetriever
from rag.store.chroma_store import ChromaStore


def query_once(
    question: str,
    k: int = 5,
    config: KMSConfig | None = None,
) -> list[dict]:
    """Run a single query and return results.

    Args:
        question: The search query.
        k: Number of results to return.
        config: Pipeline configuration.

    Returns:
        List of dicts with pid, chunk_id, score, and text preview.
    """
    config = config or KMSConfig()
    chroma = ChromaStore(KNOWLEDGE_COLLECTION, config)
    retriever = VectorRetriever(chroma)
    docs = retriever.retrieve(question, k)

    results = []
    for doc in docs:
        results.append({
            "pid": doc.metadata.get("pid", "?"),
            "chunk_id": doc.metadata.get("chunk_id", "?"),
            "text": doc.page_content,
        })
    return results


def interactive(config: KMSConfig, k: int = 5) -> None:
    """Run an interactive query loop."""
    chroma = ChromaStore(KNOWLEDGE_COLLECTION, config)
    retriever = VectorRetriever(chroma)

    print(f"KMS Query (k={k}). Type 'q' to quit.\n")

    while True:
        try:
            question = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in ("q", "quit", "exit"):
            break

        docs = retriever.retrieve(question, k)

        if not docs:
            print("  No results.\n")
            continue

        for i, doc in enumerate(docs, 1):
            pid = doc.metadata.get("pid", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  [{i}] {pid} (chunk {chunk_id})")
            print(f"      {preview}...")
            print()


def main():
    parser = argparse.ArgumentParser(description="Query the KMS.")
    parser.add_argument("question", nargs="?", help="Single question. Omit for interactive mode.")
    parser.add_argument("-k", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    config = KMSConfig()

    if args.question:
        results = query_once(args.question, k=args.k, config=config)
        for i, r in enumerate(results, 1):
            preview = r["text"][:200].replace("\n", " ")
            print(f"  [{i}] {r['pid']} (chunk {r['chunk_id']})")
            print(f"      {preview}...")
            print()
    else:
        interactive(config, k=args.k)


if __name__ == "__main__":
    main()
