# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Lab KMS** (Knowledge Management System) is a multi-layer RAG system for a research lab with 7-10 master students working on different topics. It ingests heterogeneous corpora (code repos, meeting records, research notes, markdown, papers) and organizes them into hierarchical topic layers for scoped retrieval. The system serves as a knowledge bridge across all students' research domains.

## Architecture Pattern

This project **strictly follows** the architecture pattern established in `~/rag/rag/`. Every module must conform to this structure:

### Module Structure

```
kms/
├── __init__.py
├── config.py                    # KMSConfig dataclass (single source of truth)
├── chunker/
│   ├── __init__.py              # Export base + all implementations
│   ├── base.py                  # BaseChunker(ABC)
│   └── token.py                 # TokenChunker(BaseChunker)
├── store/
│   ├── __init__.py
│   ├── base.py                  # BaseStore(ABC)
│   └── ...
├── embedder/
│   ├── __init__.py
│   ├── base.py                  # BaseEmbedder(ABC)
│   └── ...
├── retriever/
│   ├── __init__.py
│   ├── base.py                  # BaseRetriever(ABC)
│   └── ...
├── llm/
│   ├── __init__.py
│   ├── base.py                  # BaseLLM(ABC)
│   └── ...
└── cli/
    ├── __init__.py
    └── ...                      # CLI entry points
```

### Rules (from ~/rag pattern)

1. **ABC first**: Every module has `base.py` defining an ABC with `@abstractmethod` methods.
2. **Naming**: ABC = `Base{Module}` (e.g., `BaseChunker`). Implementation = `{Concrete}{Module}` (e.g., `TokenChunker`).
3. **Config injection**: All components receive `KMSConfig` via constructor. No global state.
4. **`__init__.py` exports**: Import base + all concrete implementations. Define `__all__`.
5. **Composition over inheritance**: Combine components by wrapping, not deep class hierarchies.
6. **CLI pattern**: Each CLI file has a reusable function + `main()` with argparse + `if __name__ == "__main__"` guard.
7. **Metadata**: All Documents use consistent metadata dict with at minimum `pid` and `chunk_id`.
8. **Factory pattern**: Use factory functions as single source of truth for assembling component combinations.

### Chunking Strategy

Using GraphRAG-style **token-based chunking** as the universal chunker. This handles all corpus formats (code, markdown, meeting notes, papers) without format-specific logic. The chunks are intermediate — retrieval quality comes from the embedding and layer routing, not chunk boundaries.

## Build & Run

```bash
cd app/
poetry install

# Run tests
poetry run pytest tests/ -v

# Single test
poetry run pytest tests/test_something.py::test_name -v
```

## Environment Variables

Stored in `app/.env` (git-ignored). Key variables:
- `OPENROUTER_API_KEY` — for LLM calls
- Embedding model config (TBD)
- Vector store config (TBD)

## Coding Standards

- **Imports**: Absolute only (e.g., `from kms.chunker.base import BaseChunker`)
- **Style**: PEP 8
- **Git**: Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`)
- **Type hints**: Required on all public method signatures
- **Docstrings**: Required on all ABC methods and public functions
