"""Tests for JSONL eval dataset loading."""

import hashlib
import json

import pytest

from agent.evaluation.datasets import (
    DatasetSchemaError,
    dataset_file_path,
    dataset_hash,
    load_claim_dataset,
    load_dataset,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _case(case_id="c1-routing"):
    return {
        "id": case_id,
        "claim": "c1",
        "split": "dev",
        "inputs": {"messages": ["How does the scoring module work?"]},
        "gold": {
            "expected_tool_family": "rag",
            "expected_tools_include": ["rag_search"],
        },
        "provenance": {
            "source": "unit-test",
            "labeler": "tester",
            "date": "2026-05-28",
        },
    }


def test_load_dataset_accepts_valid_jsonl(tmp_path):
    path = tmp_path / "dev.jsonl"
    _write_jsonl(path, [_case()])

    cases = load_dataset(path, claim="c1", split="dev")

    assert len(cases) == 1
    assert cases[0].id == "c1-routing"
    assert cases[0].inputs["messages"] == ["How does the scoring module work?"]
    assert cases[0].gold["expected_tool_family"] == "rag"
    assert cases[0].line_number == 1


def test_load_dataset_rejects_bad_schema(tmp_path):
    path = tmp_path / "bad.jsonl"
    _write_jsonl(path, [{**_case(), "provenance": {"source": "unit-test"}}])

    with pytest.raises(DatasetSchemaError, match="labeler"):
        load_dataset(path, claim="c1", split="dev")


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dupes.jsonl"
    _write_jsonl(path, [_case(), _case()])

    with pytest.raises(DatasetSchemaError, match="duplicate id"):
        load_dataset(path)


def test_load_dataset_rejects_claim_split_mismatch(tmp_path):
    path = tmp_path / "dev.jsonl"
    _write_jsonl(path, [_case()])

    with pytest.raises(DatasetSchemaError, match="does not match expected"):
        load_dataset(path, claim="c2", split="dev")


def test_load_claim_dataset_uses_repo_root_layout(tmp_path):
    path = dataset_file_path("c1", "dev", root=tmp_path)
    _write_jsonl(path, [_case()])

    cases = load_claim_dataset("c1", "dev", root=tmp_path)

    assert path == tmp_path / "eval" / "datasets" / "c1" / "dev.jsonl"
    assert [case.id for case in cases] == ["c1-routing"]


def test_dataset_hash_hashes_file_bytes(tmp_path):
    path = tmp_path / "dev.jsonl"
    content = json.dumps(_case(), sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")

    assert dataset_hash(path) == hashlib.sha256(content.encode("utf-8")).hexdigest()
