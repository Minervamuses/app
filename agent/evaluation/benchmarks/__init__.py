"""Public benchmark adapters for evaluator external checks."""

from agent.evaluation.benchmarks.beir import (
    BEIR_SCIFACT_URL,
    download_beir_dataset,
    evaluate_beir_lexical,
    load_beir_dataset,
    ndcg_at_k_from_qrels,
    lexical_retrieve,
)

__all__ = [
    "BEIR_SCIFACT_URL",
    "download_beir_dataset",
    "evaluate_beir_lexical",
    "load_beir_dataset",
    "ndcg_at_k_from_qrels",
    "lexical_retrieve",
]
