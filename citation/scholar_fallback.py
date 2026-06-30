"""Source-page DOI inspection and Google-Scholar-oriented DOI lookup.

These routes are intentionally best-effort and honest about their limits:

  * ``inspect_source_page`` fetches the candidate's own landing page (via the
    Web Search MCP) and looks for a DOI or an inline BibTeX block. Publisher /
    arXiv pages frequently expose one of these. The capture workflow only uses
    the DOI; inline BibTeX is reported but not used as a source of truth.
  * ``try_scholar_doi`` runs a single Google-Scholar-oriented lookup for a DOI.
    Scholar is CAPTCHA-guarded and not a stable API: when it blocks us we detect
    that and report it. We never solve a CAPTCHA and never fabricate BibTeX.
"""

from __future__ import annotations

import logging
import re

from citation.crossref import extract_doi
from citation.discovery import _coerce_text
from citation.models import PaperCandidate
from citation.runtime import PAGE_CONTENT_TOOL, SUMMARIES_TOOL, CitationRuntime

logger = logging.getLogger("citation.scholar")

# A BibTeX entry embedded in page text: @type{key, ... }
_INLINE_BIBTEX = re.compile(r"@\w+\s*\{[^@]*?title\s*=.*?\n\s*\}", re.IGNORECASE | re.DOTALL)
_CAPTCHA_HINTS = (
    "captcha",
    "unusual traffic",
    "not a robot",
    "are you a human",
    "please show you're not a robot",
    "enablejs",
)


async def _fetch_page(runtime: CitationRuntime, url: str, *, notes: list[str]) -> str:
    tool = runtime.require_web_tool(PAGE_CONTENT_TOOL)
    try:
        result = await tool.ainvoke({"url": url})
    except Exception as exc:  # noqa: BLE001 - reported to caller
        notes.append(f"page fetch failed for {url}: {type(exc).__name__}: {exc}")
        return ""
    return _coerce_text(result)


def _looks_blocked(text: str) -> bool:
    low = text.lower()
    if any(hint in low for hint in _CAPTCHA_HINTS):
        return True
    return len(text.strip()) < 80  # near-empty extraction == blocked/JS wall


async def inspect_source_page(
    runtime: CitationRuntime,
    candidate: PaperCandidate,
) -> tuple[str | None, str | None, list[str]]:
    """Fetch the candidate's source page; return ``(doi, inline_bibtex, notes)``.

    Either value may be None. A DOI lets the caller route back through the
    reliable Crossref/DOI BibTeX path; inline BibTeX is only reported so the
    trace explains what was found.
    """
    notes: list[str] = []
    if not candidate.url:
        notes.append("candidate has no source URL to inspect")
        return None, None, notes

    text = await _fetch_page(runtime, candidate.url, notes=notes)
    if not text:
        return None, None, notes
    if _looks_blocked(text):
        notes.append(f"source page looked blocked/empty: {candidate.url}")
        return None, None, notes

    doi = extract_doi(text)
    if doi:
        notes.append(f"DOI found on source page: {doi}")

    inline = _INLINE_BIBTEX.search(text)
    inline_bibtex = inline.group(0).strip() if inline else None
    if inline_bibtex:
        notes.append("inline BibTeX block found on source page")

    return doi, inline_bibtex, notes


async def try_scholar_doi(
    runtime: CitationRuntime,
    candidate: PaperCandidate,
) -> tuple[str | None, list[str]]:
    """Best-effort Google-Scholar-oriented DOI lookup.

    Returns ``(doi_or_none, notes)``. We do a single scholar-scoped search for
    the exact title and try to surface a DOI from the result text. BibTeX is
    intentionally not accepted here; after DOI resolution, capture must go
    through the Crossref/DOI route.
    """
    notes: list[str] = []
    tool = runtime.web_tools.get(SUMMARIES_TOOL)
    if tool is None:
        notes.append("Scholar DOI lookup unavailable: web search MCP not loaded")
        return None, notes

    query = f'{candidate.title} DOI site:scholar.google.com'
    try:
        result = await tool.ainvoke({"query": query, "limit": 3})
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Scholar DOI lookup failed: {type(exc).__name__}: {exc}")
        return None, notes

    text = _coerce_text(result)
    if _looks_blocked(text):
        notes.append("Google Scholar appears CAPTCHA-guarded / returned nothing usable")
        return None, notes

    doi = extract_doi(text)
    if doi:
        notes.append(f"Scholar DOI lookup found DOI: {doi}")
        return doi, notes

    notes.append("Scholar DOI lookup produced no DOI")
    return None, notes


async def try_scholar_bibtex(
    runtime: CitationRuntime,
    candidate: PaperCandidate,
) -> tuple[str | None, list[str]]:
    """Backward-compatible wrapper; BibTeX is no longer returned from Scholar."""
    _doi, notes = await try_scholar_doi(runtime, candidate)
    if _doi:
        notes.append("Scholar BibTeX fallback disabled; use DOI/Crossref route instead")
    return None, notes
