"""Reproducibility metadata and store fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import AgentConfig
from agent.evaluation.datasets import dataset_hash
from agent.paths import find_app_root

REQUIRED_STORE_FILES = ("chroma.sqlite3", "raw.json", "folder_meta.json")


class ReproError(RuntimeError):
    """Base class for reproducibility metadata failures."""


class StoreArtifactError(ReproError):
    """Raised when the eval store fixture is incomplete."""


class StoreFingerprintMismatch(ReproError):
    """Raised when an actual store fingerprint does not match the expected one."""


def collect_repro_metadata(
    config: AgentConfig,
    *,
    dataset_path: str | Path | None = None,
    dataset_id: str | None = None,
    seed: int | None = None,
    n_samples: int = 1,
    include_store_fingerprint: bool = False,
    expected_store_fingerprint: str | None = None,
    store_collection: Any | None = None,
    app_root: str | Path | None = None,
    rag_root: str | Path | None = None,
) -> dict:
    """Collect version, dataset, model, seed, and optional store metadata."""
    app_root_path = Path(app_root) if app_root is not None else find_app_root()
    rag_root_path = Path(rag_root) if rag_root is not None else app_root_path.parent / "rag"

    metadata = {
        "timestamp": _utc_timestamp(),
        "agent_git": git_metadata(app_root_path),
        "rag_git": git_metadata(rag_root_path),
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash(dataset_path) if dataset_path else None,
        "models": model_metadata(config),
        "seed": seed,
        "n_samples": n_samples,
    }

    if include_store_fingerprint:
        store = compute_store_fingerprint(config, collection=store_collection)
        if expected_store_fingerprint is not None:
            ensure_store_fingerprint(store["content_hash"], expected_store_fingerprint)
        metadata["store"] = store

    return metadata


def git_metadata(path: str | Path) -> dict:
    """Return HEAD sha and dirty flag for a git repository path."""
    repo = Path(path)
    sha = _git(repo, "rev-parse", "HEAD")
    dirty_output = _git(repo, "status", "--porcelain")
    return {
        "root": str(repo),
        "sha": sha,
        "dirty": bool(dirty_output),
        "available": sha is not None,
    }


def model_metadata(config: AgentConfig) -> dict:
    """Return model identifiers that affect evaluation behavior."""
    return {
        "llm_model": config.llm_model,
        "gen_llm_model": config.gen_llm_model,
        "judge_llm_model": config.judge_llm_model,
        "filter_llm_model": config.filter_llm_model,
        "thinking_reviewer_model": config.thinking_reviewer_model,
        "thinking_rewrite_model": config.thinking_rewrite_model,
        "thinking_repair_model": config.thinking_repair_model,
        "embed_model": config.embed_model,
        "tagger_model": config.tagger_model,
    }


def compute_store_fingerprint(
    config: AgentConfig,
    *,
    collection: Any | None = None,
    collection_name: str | None = None,
    require_artifacts: bool = True,
) -> dict:
    """Return the Chroma content hash plus store artifact manifest."""
    if collection is None:
        collection = load_chroma_collection(config, collection_name=collection_name)

    manifest = store_artifact_manifest(config.persist_dir)
    if require_artifacts and manifest["missing"]:
        raise StoreArtifactError(
            "store fixture is incomplete; restore the eval snapshot before running: "
            + ", ".join(manifest["missing"])
        )

    return {
        "persist_dir": config.persist_dir,
        "collection": collection_name or _default_collection_name(),
        "content_hash": chroma_collection_fingerprint(collection),
        "artifact_manifest": manifest,
    }


def load_chroma_collection(config: AgentConfig, *, collection_name: str | None = None):
    """Load the underlying Chroma collection without constructing an embedder."""
    from rag.store.chroma_store import ChromaStore

    name = collection_name or _default_collection_name()
    store = ChromaStore(name, config, use_embeddings=False)
    collection = getattr(store._store, "_collection", None)
    if collection is not None:
        return collection

    client = getattr(store._store, "_client", None)
    if client is not None:
        return client.get_collection(name)

    raise ReproError("could not access underlying Chroma collection")


def chroma_collection_fingerprint(collection: Any) -> str:
    """Hash ids, documents, metadatas, and embeddings from a Chroma collection."""
    records = chroma_collection_records(collection)
    payload = _canonical_json(records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chroma_collection_records(collection: Any) -> list[dict]:
    """Return canonical Chroma collection records sorted by id."""
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = _get_collection_field(data, "ids")
    documents = _get_collection_field(data, "documents")
    metadatas = _get_collection_field(data, "metadatas")
    embeddings = _get_collection_field(data, "embeddings")

    records = []
    for idx, item_id in enumerate(ids):
        records.append({
            "id": item_id,
            "document": documents[idx] if idx < len(documents) else None,
            "metadata": metadatas[idx] if idx < len(metadatas) else None,
            "embedding": embeddings[idx] if idx < len(embeddings) else None,
        })
    records.sort(key=lambda item: str(item["id"]))
    return _normalize(records)


def ensure_store_fingerprint(actual: str, expected: str) -> None:
    """Fail fast when a store content fingerprint differs from the frozen value."""
    if actual != expected:
        raise StoreFingerprintMismatch(
            f"store fingerprint mismatch: expected {expected}, got {actual}"
        )


def store_artifact_manifest(persist_dir: str | Path) -> dict:
    """Inspect required Chroma/raw JSON fixture artifacts without hashing bytes."""
    root = Path(persist_dir)
    missing: list[str] = []
    files: list[str] = []
    if not root.exists():
        return {
            "persist_dir": str(root),
            "files": [],
            "hnsw_indexes": [],
            "missing": [str(root)],
        }

    for file_name in REQUIRED_STORE_FILES:
        path = root / file_name
        if path.is_file():
            files.append(file_name)
        else:
            missing.append(file_name)

    indexes = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        index_files = sorted(item.name for item in child.iterdir() if item.is_file())
        if "index_metadata.pickle" not in index_files:
            continue
        bin_files = [name for name in index_files if name.endswith(".bin")]
        if not bin_files:
            missing.append(f"{child.name}/*.bin")
        indexes.append({"dir": child.name, "files": index_files})

    if not indexes:
        missing.append("hnsw index directory")

    return {
        "persist_dir": str(root),
        "files": sorted(files),
        "hnsw_indexes": indexes,
        "missing": missing,
    }


def _git(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _get_collection_field(data: dict, key: str) -> Any:
    value = data.get(key)
    return [] if value is None else value


def _default_collection_name() -> str:
    from rag.config import KNOWLEDGE_COLLECTION

    return KNOWLEDGE_COLLECTION


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
