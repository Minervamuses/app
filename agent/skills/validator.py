"""Deterministic validation checks for active skill responses."""

from __future__ import annotations

import re

PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
CITATION_MARKER_RE = re.compile(
    r"(\[[^\]]+\]|\([A-Z][A-Za-z-]+,\s*\d{4}\)|\bdoi\s*:|\bDOI\s*:)",
    re.IGNORECASE,
)


def validate_skill_output(
    *,
    active_skill: str | None,
    text: str,
) -> list[str]:
    """Return skill policy violations for a final assistant response."""
    if not active_skill:
        return []
    if active_skill != "academic-paper-writing":
        return []

    violations: list[str] = []
    if PERCENT_RE.search(text) and not CITATION_MARKER_RE.search(text):
        violations.append(
            "Quantitative claims with percentages need a supplied source, citation marker, or explicit placeholder."
        )
    return violations
