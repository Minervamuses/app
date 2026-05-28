"""Deterministic ranked retrieval metrics."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

ChunkKey = tuple[str, int]


def recall_at_k(predicted: Iterable, relevant: Iterable, k: int) -> float:
    """Return binary recall@k over `(pid, chunk_id)` keys."""
    gold = set(_normalize_keys(relevant))
    if not gold:
        return 0.0
    hits = set(_normalize_keys(list(predicted)[:k]))
    return len(hits & gold) / len(gold)


def mrr(predicted: Iterable, relevant: Iterable) -> float:
    """Return reciprocal rank of the first relevant predicted key."""
    gold = set(_normalize_keys(relevant))
    if not gold:
        return 0.0
    for rank, key in enumerate(_normalize_keys(predicted), start=1):
        if key in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted: Iterable, relevant: Iterable, k: int) -> float:
    """Return binary nDCG@k for ranked retrieval."""
    gold = set(_normalize_keys(relevant))
    if not gold:
        return 0.0
    ranked = _normalize_keys(list(predicted)[:k])
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, key in enumerate(ranked, start=1)
        if key in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def score_ranked_retrieval(predicted: Iterable, relevant: Iterable, k: int) -> dict[str, float]:
    """Return the standard C2 deterministic retrieval metrics."""
    return {
        f"recall@{k}": recall_at_k(predicted, relevant, k),
        "mrr": mrr(predicted, relevant),
        f"ndcg@{k}": ndcg_at_k(predicted, relevant, k),
    }


def _normalize_keys(items: Iterable) -> list[ChunkKey]:
    keys: list[ChunkKey] = []
    for item in items:
        if isinstance(item, Mapping):
            keys.append((str(item["pid"]), int(item["chunk_id"])))
        else:
            pid, chunk_id = item
            keys.append((str(pid), int(chunk_id)))
    return keys
