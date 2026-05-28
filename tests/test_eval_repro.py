"""Tests for evaluator reproducibility metadata."""

import json

import pytest

from agent.config import AgentConfig
from agent.evaluation.repro import (
    StoreArtifactError,
    StoreFingerprintMismatch,
    chroma_collection_fingerprint,
    collect_repro_metadata,
    compute_store_fingerprint,
    ensure_store_fingerprint,
    store_artifact_manifest,
)


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def get(self, include):
        assert include == ["documents", "metadatas", "embeddings"]
        return {
            "ids": [row["id"] for row in self.rows],
            "documents": [row["document"] for row in self.rows],
            "metadatas": [row["metadata"] for row in self.rows],
            "embeddings": [row["embedding"] for row in self.rows],
        }


def _rows():
    return [
        {
            "id": "b",
            "document": "second",
            "metadata": {"pid": "p2", "chunk_id": 0, "tags": ["beta"]},
            "embedding": [0.2, 0.3],
        },
        {
            "id": "a",
            "document": "first",
            "metadata": {"pid": "p1", "chunk_id": 0, "category": "docs"},
            "embedding": [0.0, 0.1],
        },
    ]


def _complete_store(root):
    root.mkdir(parents=True, exist_ok=True)
    for file_name in ("chroma.sqlite3", "raw.json", "folder_meta.json"):
        (root / file_name).write_text("{}", encoding="utf-8")
    index_dir = root / "12345678-1234-5678-1234-567812345678"
    index_dir.mkdir()
    (index_dir / "index_metadata.pickle").write_bytes(b"metadata")
    (index_dir / "header.bin").write_bytes(b"bin")


def test_chroma_collection_fingerprint_is_order_stable():
    forward = chroma_collection_fingerprint(FakeCollection(_rows()))
    reverse = chroma_collection_fingerprint(FakeCollection(list(reversed(_rows()))))

    assert forward == reverse


def test_chroma_collection_fingerprint_changes_when_content_changes():
    original = chroma_collection_fingerprint(FakeCollection(_rows()))
    changed_rows = _rows()
    changed_rows[0] = {**changed_rows[0], "document": "changed"}

    assert chroma_collection_fingerprint(FakeCollection(changed_rows)) != original


def test_ensure_store_fingerprint_raises_on_mismatch():
    with pytest.raises(StoreFingerprintMismatch, match="fingerprint mismatch"):
        ensure_store_fingerprint("actual", "expected")


def test_store_artifact_manifest_checks_required_files_and_hnsw_index(tmp_path):
    store_dir = tmp_path / "store"
    _complete_store(store_dir)

    manifest = store_artifact_manifest(store_dir)

    assert manifest["missing"] == []
    assert manifest["files"] == ["chroma.sqlite3", "folder_meta.json", "raw.json"]
    assert manifest["hnsw_indexes"][0]["files"] == [
        "header.bin",
        "index_metadata.pickle",
    ]


def test_compute_store_fingerprint_fails_when_artifacts_are_missing(tmp_path):
    cfg = AgentConfig(persist_dir=str(tmp_path / "missing-store"))

    with pytest.raises(StoreArtifactError, match="restore the eval snapshot"):
        compute_store_fingerprint(cfg, collection=FakeCollection(_rows()))


def test_collect_repro_metadata_includes_versions_dataset_models_and_store(tmp_path):
    dataset_path = tmp_path / "dev.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "claim": "c1",
                "split": "dev",
                "inputs": {},
                "gold": {},
                "provenance": {
                    "source": "unit-test",
                    "labeler": "tester",
                    "date": "2026-05-28",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    store_dir = tmp_path / "store"
    _complete_store(store_dir)
    cfg = AgentConfig(persist_dir=str(store_dir), llm_model="test-llm")

    expected_fingerprint = chroma_collection_fingerprint(FakeCollection(_rows()))
    metadata = collect_repro_metadata(
        cfg,
        dataset_path=dataset_path,
        dataset_id="c1/dev",
        seed=7,
        n_samples=3,
        include_store_fingerprint=True,
        expected_store_fingerprint=expected_fingerprint,
        store_collection=FakeCollection(_rows()),
    )

    assert metadata["dataset_id"] == "c1/dev"
    assert metadata["dataset_hash"]
    assert metadata["models"]["llm_model"] == "test-llm"
    assert metadata["models"]["embed_model"] == cfg.embed_model
    assert metadata["seed"] == 7
    assert metadata["n_samples"] == 3
    assert metadata["agent_git"]["sha"]
    assert metadata["store"]["content_hash"] == expected_fingerprint
