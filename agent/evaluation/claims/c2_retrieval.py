"""C2 deterministic retrieval evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.config import AgentConfig
from agent.evaluation.base import BaseEvaluator, EvalResult
from agent.evaluation.datasets import EvalCase, dataset_file_path, load_claim_dataset
from agent.evaluation.metrics.ranking import score_ranked_retrieval
from agent.evaluation.repro import compute_store_fingerprint, ensure_store_fingerprint
from agent.paths import find_app_root

SearchFn = Callable[[str, int, dict], list[Any]]
FingerprintFn = Callable[[], str]


class C2RetrievalEvaluator(BaseEvaluator):
    """Evaluate C2 retrieval quality by calling `rag.api.search()` directly."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        search_fn: SearchFn | None = None,
        enforce_fingerprint: bool = True,
        expected_store_fingerprint: str | None = None,
        store_fingerprint_fn: FingerprintFn | None = None,
    ):
        self.config = config or AgentConfig()
        self.search_fn = search_fn
        self.enforce_fingerprint = enforce_fingerprint
        self.expected_store_fingerprint = expected_store_fingerprint
        self.store_fingerprint_fn = store_fingerprint_fn

    def generate(self, n: int = 0, output_path: str | None = None) -> list[dict]:
        """Load the frozen C2 dev dataset."""
        del n
        cases = [case.raw for case in load_claim_dataset("c2", "dev")]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(cases, file_obj, ensure_ascii=False, indent=2)
        return cases

    def evaluate(self, cases: list[EvalCase | dict]) -> EvalResult:
        """Run ranked retrieval cases and aggregate deterministic metrics."""
        expected = self.expected_store_fingerprint
        if expected is None and self.enforce_fingerprint:
            expected = load_c2_fixture()["content_hash"]
        if self.enforce_fingerprint and expected is not None:
            actual = (
                self.store_fingerprint_fn()
                if self.store_fingerprint_fn is not None
                else compute_store_fingerprint(self.config)["content_hash"]
            )
            ensure_store_fingerprint(actual, expected)

        details = []
        metric_sums: dict[str, float] = {}
        for raw_case in cases:
            case = _raw_case(raw_case)
            query = case["inputs"]["query"]
            k = int(case["gold"].get("k") or case["inputs"].get("k") or self.config.default_k)
            filters = dict(case["inputs"].get("filters") or {})
            hits = self._search(query, k, filters)
            predicted = [_hit_key(hit) for hit in hits]
            scores = score_c2_predictions(predicted, case["gold"]["relevant"], k)
            for metric, value in scores.items():
                metric_sums[metric] = metric_sums.get(metric, 0.0) + value
            details.append({
                "id": case["id"],
                "query": query,
                "k": k,
                "filters": filters,
                "prediction": [
                    {"pid": pid, "chunk_id": chunk_id}
                    for pid, chunk_id in predicted
                ],
                "gold": case["gold"],
                "scores": scores,
            })

        total = len(cases)
        aggregate = {
            metric: value / total if total else 0.0
            for metric, value in sorted(metric_sums.items())
        }
        return EvalResult(
            name="C2Retrieval",
            total=total,
            scores=aggregate,
            details=details,
            metadata={
                "claim": "c2",
                "store_fingerprint_checked": self.enforce_fingerprint,
            },
        )

    def _search(self, query: str, k: int, filters: dict) -> list[Any]:
        if self.search_fn is not None:
            return self.search_fn(query, k, filters)
        from rag.api import search

        return search(query, k=k, config=self.config, **filters)


def score_c2_predictions(
    predicted: list[tuple[str, int]],
    relevant: list[dict],
    k: int,
) -> dict[str, float]:
    """Score one C2 ranked prediction list against relevant chunk ids."""
    return score_ranked_retrieval(predicted, relevant, k)


def load_c2_fixture(*, root: str | Path | None = None) -> dict:
    """Load the frozen C2 store fixture metadata."""
    base = Path(root) if root is not None else find_app_root()
    path = base / "eval" / "datasets" / "c2" / "fixture.json"
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _raw_case(case: EvalCase | dict) -> dict:
    return case.raw if isinstance(case, EvalCase) else dict(case)


def _hit_key(hit: Any) -> tuple[str, int]:
    if isinstance(hit, Mapping):
        return str(hit["pid"]), int(hit["chunk_id"])
    return str(getattr(hit, "pid")), int(getattr(hit, "chunk_id"))
