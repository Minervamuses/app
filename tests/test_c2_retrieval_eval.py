"""Tests for the C2 retrieval evaluator."""

from dataclasses import dataclass

import pytest

from agent.config import AgentConfig
from agent.evaluation.claims.c2_retrieval import (
    C2RetrievalEvaluator,
    load_c2_fixture,
    score_c2_predictions,
)
from agent.evaluation.datasets import load_claim_dataset
from agent.evaluation.repro import StoreFingerprintMismatch


@dataclass(frozen=True)
class FakeHit:
    pid: str
    chunk_id: int


def test_c2_dataset_and_fixture_load_from_repo_root():
    cases = load_claim_dataset("c2", "dev")
    fixture = load_c2_fixture()

    assert len(cases) == 3
    assert fixture["content_hash"]
    assert fixture["embed_model"] == "bge-m3"


def test_score_c2_predictions_reports_rank_metrics():
    scores = score_c2_predictions(
        [("p1", 0), ("p2", 0)],
        [{"pid": "p2", "chunk_id": 0}],
        5,
    )

    assert scores["recall@5"] == 1.0
    assert scores["mrr"] == 0.5
    assert scores["ndcg@5"] == pytest.approx(0.6309297535714575)


def test_c2_evaluator_scores_injected_search_results(tmp_path):
    case = load_claim_dataset("c2", "dev")[0]

    def search_fn(query, k, filters):
        assert query == "Score.java extends MutationResult id field"
        assert k == 5
        assert filters == {}
        return [
            FakeHit("irrelevant", 0),
            FakeHit("PiDNA1/source_code/PiDNA/src/snoopy/pdb/pidna/Score.java", 0),
        ]

    evaluator = C2RetrievalEvaluator(
        AgentConfig(persist_dir=str(tmp_path)),
        search_fn=search_fn,
        enforce_fingerprint=False,
    )
    result = evaluator.evaluate([case])

    assert result.name == "C2Retrieval"
    assert result.scores["recall@5"] == 1.0
    assert result.scores["mrr"] == 0.5
    assert result.metadata["store_fingerprint_checked"] is False


def test_c2_evaluator_fails_fast_on_fingerprint_mismatch(tmp_path):
    evaluator = C2RetrievalEvaluator(
        AgentConfig(persist_dir=str(tmp_path)),
        search_fn=lambda query, k, filters: [],
        expected_store_fingerprint="expected",
        store_fingerprint_fn=lambda: "actual",
    )

    with pytest.raises(StoreFingerprintMismatch):
        evaluator.evaluate([load_claim_dataset("c2", "dev")[0]])
