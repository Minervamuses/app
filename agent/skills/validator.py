"""Deterministic validation checks for active skill responses."""

from __future__ import annotations

import re
from collections.abc import Callable

PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
CITATION_MARKER_RE = re.compile(
    r"(\[[^\]]+\]|\([A-Z][A-Za-z-]+,\s*\d{4}\)|\bdoi\s*:|\bDOI\s*:)",
    re.IGNORECASE,
)

Validator = Callable[[str], list[str]]


def no_uncited_percentages(text: str) -> list[str]:
    """Require citation markers for percentage claims."""
    if PERCENT_RE.search(text) and not CITATION_MARKER_RE.search(text):
        return [
            "Quantitative claims with percentages need a supplied source, citation marker, or explicit placeholder."
        ]
    return []


VALIDATORS_BY_SKILL: dict[str, list[Validator]] = {
    "academic-paper-writing": [no_uncited_percentages],
}


def validate_skill_output(
    *,
    active_skill: str | None,
    text: str,
) -> list[str]:
    """Return skill policy violations for a final assistant response."""
    if not active_skill:
        return []
    violations: list[str] = []
    for validator in VALIDATORS_BY_SKILL.get(active_skill, []):
        violations.extend(validator(text))
    return violations
