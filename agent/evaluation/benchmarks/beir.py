"""Minimal BEIR/SciFact benchmark spike.

This adapter intentionally avoids a hard BEIR package dependency for now. It
uses the public BEIR JSONL/qrels file format and computes a deterministic
lexical-overlap baseline with nDCG@10. That keeps the spike runnable in this
repo while preserving the external benchmark interface for a stronger retriever.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class BEIRDataset:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def download_beir_dataset(
    *,
    dataset: str = "scifact",
    output_dir: str | Path = "eval/benchmarks",
    url: str = BEIR_SCIFACT_URL,
) -> Path:
    """Download and extract a BEIR dataset zip, returning the dataset directory."""
    if dataset != "scifact":
        raise ValueError("only the SciFact BEIR spike is configured")
    root = Path(output_dir)
    target_dir = root / dataset
    if (target_dir / "corpus.jsonl").exists():
        return target_dir
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"{dataset}.zip"
    urllib.request.urlretrieve(url, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    if not target_dir.exists():
        raise FileNotFoundError(f"download did not produce {target_dir}")
    return target_dir


def evaluate_beir_lexical(
    data_dir: str | Path,
    *,
    split: str = "test",
    top_k: int = 10,
    limit_queries: int | None = None,
) -> dict[str, float]:
    """Run a deterministic lexical-overlap baseline and return nDCG@k."""
    dataset = load_beir_dataset(data_dir, split=split, limit_queries=limit_queries)
    results = lexical_retrieve(dataset.corpus, dataset.queries, top_k=top_k)
    return {f"ndcg@{top_k}": ndcg_at_k_from_qrels(dataset.qrels, results, top_k)}


def load_beir_dataset(
    data_dir: str | Path,
    *,
    split: str = "test",
    limit_queries: int | None = None,
) -> BEIRDataset:
    """Load corpus, queries, and qrels from a BEIR-format dataset directory."""
    root = Path(data_dir)
    corpus = {
        row["_id"]: " ".join(
            part for part in (row.get("title", ""), row.get("text", "")) if part
        )
        for row in _read_jsonl(root / "corpus.jsonl")
    }
    queries = {
        row["_id"]: row["text"]
        for row in _read_jsonl(root / "queries.jsonl")
    }
    qrels = _read_qrels(root / "qrels" / f"{split}.tsv")
    if limit_queries is not None:
        allowed = set(list(qrels)[:limit_queries])
        queries = {qid: text for qid, text in queries.items() if qid in allowed}
        qrels = {qid: rels for qid, rels in qrels.items() if qid in allowed}
    else:
        queries = {qid: text for qid, text in queries.items() if qid in qrels}
    return BEIRDataset(corpus=corpus, queries=queries, qrels=qrels)


def lexical_retrieve(
    corpus: dict[str, str],
    queries: dict[str, str],
    *,
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    """Rank documents by normalized token-overlap score."""
    doc_tokens = {doc_id: set(_tokens(text)) for doc_id, text in corpus.items()}
    results: dict[str, dict[str, float]] = {}
    for query_id, query in queries.items():
        query_tokens = set(_tokens(query))
        scored = []
        for doc_id, tokens in doc_tokens.items():
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(max(len(tokens), 1))
            scored.append((doc_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        results[query_id] = {
            doc_id: score for doc_id, score in scored[:top_k]
        }
    return results


def ndcg_at_k_from_qrels(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k: int,
) -> float:
    """Compute mean graded nDCG@k from BEIR qrels and run scores."""
    values = []
    for query_id, relevant in qrels.items():
        ranking = sorted(
            results.get(query_id, {}).items(),
            key=lambda item: (-item[1], item[0]),
        )[:k]
        dcg = sum(
            _gain(relevant.get(doc_id, 0)) / math.log2(rank + 1)
            for rank, (doc_id, _score) in enumerate(ranking, start=1)
        )
        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(
            _gain(score) / math.log2(rank + 1)
            for rank, score in enumerate(ideal, start=1)
        )
        values.append(dcg / idcg if idcg else 0.0)
    return sum(values) / len(values) if values else 0.0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if line_number == 1 and parts[0] in {"query-id", "query_id"}:
                continue
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 tab-separated columns")
            query_id, doc_id, score = parts
            qrels.setdefault(query_id, {})[doc_id] = int(score)
    return qrels


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def _gain(relevance: int) -> float:
    return (2 ** relevance) - 1
