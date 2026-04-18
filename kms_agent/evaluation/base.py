"""Abstract base class for evaluators."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM response text, handling code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.index("{")
    end = cleaned.rindex("}") + 1
    return json.loads(cleaned[start:end])


@dataclass
class EvalResult:
    """Aggregated evaluation result for a test suite."""

    name: str
    total: int
    scores: dict[str, float]
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the evaluation."""
        lines = [f"=== {self.name} ({self.total} cases) ==="]
        for metric, value in self.scores.items():
            if metric.endswith("_raw"):
                lines.append(f"  {metric}: {value:.2f}")
            else:
                lines.append(f"  {metric}: {value:.2%}")
        return "\n".join(lines)


class BaseEvaluator(ABC):
    """Abstract base class for evaluation strategies."""

    @abstractmethod
    def generate(self, n: int, output_path: str | None = None) -> list[dict]:
        """Generate evaluation test cases.

        Args:
            n: Number of test cases to generate.
            output_path: Optional path to save generated cases as JSON.

        Returns:
            List of test case dicts.
        """

    @abstractmethod
    def evaluate(self, cases: list[dict]) -> EvalResult:
        """Run evaluation on a set of test cases.

        Args:
            cases: List of test case dicts (from generate() or loaded from JSON).

        Returns:
            EvalResult with aggregated scores and per-case details.
        """
