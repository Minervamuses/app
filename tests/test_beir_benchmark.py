"""Tests for the minimal BEIR benchmark adapter."""

import json

import pytest

from agent.evaluation.benchmarks.beir import (
    evaluate_beir_lexical,
    lexical_retrieve,
    load_beir_dataset,
    ndcg_at_k_from_qrels,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_tiny_beir(root):
    _write_jsonl(
        root / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Alpha", "text": "protein dna binding"},
            {"_id": "d2", "title": "Beta", "text": "climate policy"},
            {"_id": "d3", "title": "Gamma", "text": "protein folding"},
        ],
    )
    _write_jsonl(
        root / "queries.jsonl",
        [
            {"_id": "q1", "text": "protein dna"},
            {"_id": "q2", "text": "climate"},
        ],
    )
    qrels = root / "qrels" / "test.tsv"
    qrels.parent.mkdir(parents=True)
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t1\n"
        "q2\td2\t1\n",
        encoding="utf-8",
    )


def test_load_beir_dataset_reads_standard_files(tmp_path):
    _write_tiny_beir(tmp_path)

    dataset = load_beir_dataset(tmp_path)

    assert dataset.corpus["d1"] == "Alpha protein dna binding"
    assert dataset.queries == {"q1": "protein dna", "q2": "climate"}
    assert dataset.qrels == {"q1": {"d1": 1}, "q2": {"d2": 1}}


def test_lexical_retrieve_ranks_by_token_overlap():
    results = lexical_retrieve(
        {"d1": "protein dna binding", "d2": "protein", "d3": "climate"},
        {"q1": "protein dna"},
        top_k=2,
    )

    assert list(results["q1"]) == ["d1", "d2"]


def test_ndcg_at_k_from_qrels_scores_rankings():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    results = {"q1": {"d1": 10.0}, "q2": {"d3": 9.0, "d2": 1.0}}

    assert ndcg_at_k_from_qrels(qrels, results, 2) == pytest.approx(
        (1.0 + 0.6309297535714575) / 2
    )


def test_evaluate_beir_lexical_returns_ndcg(tmp_path):
    _write_tiny_beir(tmp_path)

    scores = evaluate_beir_lexical(tmp_path, top_k=10)

    assert scores == {"ndcg@10": 1.0}
