"""Tests for deterministic ranked retrieval metrics."""

import pytest

from agent.evaluation.metrics.ranking import (
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_ranked_retrieval,
)


def test_recall_at_k_counts_relevant_chunk_keys():
    predicted = [("p1", 0), ("p2", 0), ("p3", 1)]
    relevant = [{"pid": "p2", "chunk_id": 0}, {"pid": "p4", "chunk_id": 0}]

    assert recall_at_k(predicted, relevant, 2) == 0.5


def test_mrr_returns_reciprocal_rank_of_first_relevant_hit():
    predicted = [("p1", 0), ("p2", 0), ("p3", 1)]
    relevant = [("p3", 1), ("p4", 0)]

    assert mrr(predicted, relevant) == pytest.approx(1 / 3)


def test_ndcg_at_k_uses_binary_relevance():
    predicted = [("p1", 0), ("p2", 0), ("p3", 1)]
    relevant = [("p2", 0), ("p3", 1)]

    assert ndcg_at_k(predicted, relevant, 3) == pytest.approx(0.6934264)


def test_score_ranked_retrieval_reports_named_metrics():
    predicted = [("p1", 0), ("p2", 0)]
    relevant = [("p2", 0)]

    assert score_ranked_retrieval(predicted, relevant, 10) == {
        "recall@10": 1.0,
        "mrr": 0.5,
        "ndcg@10": pytest.approx(0.6309297535714575),
    }


def test_empty_gold_scores_zero():
    assert recall_at_k([("p1", 0)], [], 10) == 0.0
    assert mrr([("p1", 0)], []) == 0.0
    assert ndcg_at_k([("p1", 0)], [], 10) == 0.0
