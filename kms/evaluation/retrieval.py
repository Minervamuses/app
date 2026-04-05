"""Retrieval evaluator — tests search tool hit rate with and without metadata filters.

Generates questions from sampled chunks via LLM, then checks whether the
source chunk appears in search results. Measures the value of metadata filters
by comparing hit rates with and without them.
"""

import json
import random
from pathlib import Path

from kms.config import KMSConfig, KNOWLEDGE_COLLECTION
from kms.evaluation.base import BaseEvaluator, EvalResult, _extract_json
from kms.llm.openrouter import OpenRouterLLM
from kms.retriever.vector import VectorRetriever
from kms.store.chroma_store import ChromaStore
from kms.store.json_store import JSONStore
from kms.tool.search import _build_where

GENERATE_PROMPT = """You are a test-case generator. Given a chunk from a knowledge base, generate a natural question that requires finding this chunk to answer.

file_path: {file_path}
category: {category}
tags: {tags}
date: {date}

Content:
{content}

Return a JSON object with:
1. "question": a natural question (in the language of the content)
2. "expected_query": the best semantic search query for finding this chunk
3. "expected_filters": metadata filters that would help narrow results (object with optional keys: category, file_type, date_from, date_to)
4. "difficulty": "easy", "medium", or "hard"

Return ONLY the JSON object, no explanation."""


class RetrievalEvaluator(BaseEvaluator):
    """Evaluate search quality by generating questions from chunks and measuring hit rate."""

    def __init__(self, config: KMSConfig | None = None):
        self.config = config or KMSConfig()
        self._llm = OpenRouterLLM(config=self.config)

    def generate(self, n: int = 30, output_path: str | None = None) -> list[dict]:
        """Sample chunks and generate retrieval test cases via LLM.

        Args:
            n: Number of test cases to generate.
            output_path: Optional path to save generated cases as JSON.

        Returns:
            List of test case dicts with question, expected_query, expected_filters,
            difficulty, and ground_truth (pid + chunk_id).
        """
        json_store = JSONStore(self.config.raw_json_path())
        all_docs = json_store.get()

        if len(all_docs) < n:
            n = len(all_docs)

        sampled = random.sample(all_docs, n)
        cases = []

        for doc in sampled:
            meta = doc.metadata
            prompt = GENERATE_PROMPT.format(
                file_path=meta.get("file_path", "?"),
                category=meta.get("category", "?"),
                tags=meta.get("tags", "[]"),
                date=meta.get("date", 0),
                content=doc.page_content[:1500],
            )

            try:
                response = self._llm.invoke(prompt, max_tokens=300, temperature=0.0)
                data = _extract_json(response)
            except (ValueError, json.JSONDecodeError):
                continue

            cases.append({
                "question": data.get("question", ""),
                "expected_query": data.get("expected_query", ""),
                "expected_filters": data.get("expected_filters", {}),
                "difficulty": data.get("difficulty", "medium"),
                "ground_truth": {
                    "pid": meta.get("pid"),
                    "chunk_id": meta.get("chunk_id"),
                },
            })

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)

        return cases

    def evaluate(self, cases: list[dict], k: int = 5) -> EvalResult:
        """Run retrieval evaluation: search with and without filters, compute hit rates.

        Args:
            cases: Test cases from generate() or loaded from JSON.
            k: Number of results to retrieve per query.

        Returns:
            EvalResult with hit_rate_no_filter, hit_rate_with_filter, and filter_lift.
        """
        store = ChromaStore(KNOWLEDGE_COLLECTION, self.config)
        retriever = VectorRetriever(store)

        hits_no_filter = 0
        hits_with_filter = 0
        details = []

        for case in cases:
            query = case.get("expected_query", case["question"])
            gt = case["ground_truth"]
            gt_pid = gt["pid"]
            gt_cid = gt["chunk_id"]
            filters = case.get("expected_filters", {})

            # Search without filters
            docs_plain = retriever.retrieve(query, k=k)
            hit_plain = any(
                d.metadata.get("pid") == gt_pid and d.metadata.get("chunk_id") == gt_cid
                for d in docs_plain
            )

            # Search with filters
            where = _build_where(
                category=filters.get("category"),
                file_type=filters.get("file_type"),
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
            )
            docs_filtered = retriever.retrieve(query, k=k, where=where)
            hit_filtered = any(
                d.metadata.get("pid") == gt_pid and d.metadata.get("chunk_id") == gt_cid
                for d in docs_filtered
            )

            hits_no_filter += hit_plain
            hits_with_filter += hit_filtered

            details.append({
                "question": case["question"],
                "difficulty": case.get("difficulty"),
                "hit_no_filter": hit_plain,
                "hit_with_filter": hit_filtered,
                "ground_truth": gt,
            })

        total = len(cases)
        rate_plain = hits_no_filter / total if total else 0
        rate_filtered = hits_with_filter / total if total else 0

        return EvalResult(
            name="Retrieval",
            total=total,
            scores={
                "hit_rate_no_filter": rate_plain,
                "hit_rate_with_filter": rate_filtered,
                "filter_lift": rate_filtered - rate_plain,
            },
            details=details,
        )
