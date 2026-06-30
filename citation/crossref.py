"""DOI discovery, Crossref metadata matching, and BibTeX retrieval.

Uses only the standard library (``urllib``) so the prototype adds no
dependency. Network endpoints:
  * ``https://api.crossref.org/works`` — bibliographic search.
  * ``https://api.crossref.org/works/{doi}/transform/application/x-bibtex``
    and ``https://doi.org/{doi}`` (Accept: application/x-bibtex) — BibTeX.

We never treat Crossref's first hit as ground truth: every candidate is scored
locally on title similarity, year, and author overlap before we trust a DOI.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

from citation.models import CrossrefMatch, PaperCandidate

logger = logging.getLogger("citation.crossref")

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_TRAILING_JUNK = re.compile(r"[).,;:'\"\]]+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

CROSSREF_SEARCH = "https://api.crossref.org/works"
CROSSREF_TRANSFORM = "https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
DOI_NEGOTIATE = "https://doi.org/{doi}"

# Decision thresholds. High confidence requires a strong title match that is
# also clearly ahead of the runner-up; otherwise the result is "ambiguous".
HIGH_CONFIDENCE = 0.85
MIN_TITLE_SIM_FOR_HIGH = 0.82
AMBIGUITY_DELTA = 0.08
MIN_PLAUSIBLE = 0.60


def _user_agent() -> str:
    mailto = os.getenv("CROSSREF_MAILTO", "").strip()
    base = "citation-prototype/0.1 (isolated experiment)"
    return f"{base} mailto:{mailto}" if mailto else base


def extract_doi(*texts: str | None) -> str | None:
    """Return the first DOI found across the given text fragments, or None."""
    for text in texts:
        if not text:
            continue
        m = _DOI_RE.search(text)
        if m:
            return _clean_doi(m.group(0))
    return None


def _clean_doi(doi: str) -> str:
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return _TRAILING_JUNK.sub("", doi).strip()


def _norm(text: str) -> str:
    return _NON_ALNUM.sub(" ", (text or "").lower()).strip()


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _surnames(authors: list[str]) -> set[str]:
    out: set[str] = set()
    for a in authors:
        token = a.strip().split()[-1] if a.strip() else ""
        token = _NON_ALNUM.sub("", token.lower())
        if len(token) >= 2:
            out.add(token)
    return out


def _http_get(url: str, *, accept: str | None = None, timeout: float = 20.0) -> tuple[int, bytes]:
    headers = {"User-Agent": _user_agent()}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed hosts
        return resp.status, resp.read()


# --- Crossref search ------------------------------------------------------


def search_crossref(title: str, *, rows: int = 5, timeout: float = 20.0) -> list[dict]:
    """Query Crossref by bibliographic title; return raw ``items`` (may be empty)."""
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": title,
            "rows": str(max(1, min(rows, 20))),
            "select": "DOI,title,author,issued,score",
        }
    )
    url = f"{CROSSREF_SEARCH}?{params}"
    status, body = _http_get(url, accept="application/json", timeout=timeout)
    if status != 200:
        raise RuntimeError(f"Crossref search HTTP {status}")
    data = json.loads(body.decode("utf-8", errors="replace"))
    return data.get("message", {}).get("items", []) or []


def _item_to_match(candidate: PaperCandidate, item: dict) -> CrossrefMatch:
    titles = item.get("title") or []
    cr_title = titles[0] if titles else ""
    authors = [
        " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
        or a.get("name", "")
        for a in (item.get("author") or [])
    ]
    authors = [a for a in authors if a]
    year = None
    parts = (item.get("issued") or {}).get("date-parts") or []
    if parts and parts[0]:
        try:
            year = int(parts[0][0])
        except (TypeError, ValueError):
            year = None

    title_sim = _title_similarity(candidate.title, cr_title)

    year_matches: bool | None = None
    if candidate.year and year:
        year_matches = abs(candidate.year - year) <= 1  # tolerate preprint/year drift

    author_overlap: bool | None = None
    cand_surnames = _surnames(candidate.authors)
    if cand_surnames:
        author_overlap = bool(cand_surnames & _surnames(authors))

    confidence = title_sim
    if year_matches is True:
        confidence += 0.05
    elif year_matches is False:
        confidence -= 0.25
    if author_overlap is True:
        confidence += 0.05
    elif author_overlap is False:
        confidence -= 0.10
    confidence = max(0.0, min(1.0, confidence))

    return CrossrefMatch(
        doi=_clean_doi(item.get("DOI", "")),
        title=cr_title,
        authors=authors,
        year=year,
        crossref_score=float(item.get("score", 0.0) or 0.0),
        title_similarity=round(title_sim, 4),
        year_matches=year_matches,
        author_overlap=author_overlap,
        confidence=round(confidence, 4),
    )


def rank_matches(candidate: PaperCandidate, items: list[dict]) -> list[CrossrefMatch]:
    """Score every Crossref item locally and return them best-confidence first."""
    matches = [_item_to_match(candidate, it) for it in items if it.get("DOI")]
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches


def classify_matches(matches: list[CrossrefMatch]) -> tuple[str, CrossrefMatch | None]:
    """Classify the ranked matches into a decision tier.

    Returns ``(tier, best_match_or_none)`` where tier is one of:
      * ``"high"``      — confident single match, safe to auto-use.
      * ``"ambiguous"`` — plausible but a close runner-up exists, or mid score.
      * ``"low"``       — best match too weak to trust.
      * ``"none"``      — no matches at all.
    """
    if not matches:
        return "none", None
    best = matches[0]
    runner = matches[1] if len(matches) > 1 else None

    if best.confidence < MIN_PLAUSIBLE:
        return "low", best

    close_runner = (
        runner is not None
        and runner.confidence >= MIN_PLAUSIBLE
        and (best.confidence - runner.confidence) < AMBIGUITY_DELTA
    )
    strong = (
        best.confidence >= HIGH_CONFIDENCE
        and best.title_similarity >= MIN_TITLE_SIM_FOR_HIGH
        and best.year_matches is not False
    )
    if strong and not close_runner:
        return "high", best
    return "ambiguous", best


# --- BibTeX retrieval -----------------------------------------------------


def fetch_bibtex_for_doi(doi: str, *, timeout: float = 25.0) -> tuple[str | None, list[str]]:
    """Retrieve BibTeX for ``doi`` via content negotiation.

    Returns ``(bibtex_or_none, notes)``. Tries the Crossref transform endpoint
    first, then doi.org content negotiation. Never fabricates: a failed
    retrieval returns ``None`` with explanatory notes.
    """
    notes: list[str] = []
    doi = _clean_doi(doi)
    encoded = urllib.parse.quote(doi, safe="/")
    attempts = [
        ("crossref-transform", CROSSREF_TRANSFORM.format(doi=encoded), "application/x-bibtex"),
        ("doi.org-negotiate", DOI_NEGOTIATE.format(doi=encoded), "application/x-bibtex"),
    ]
    for label, url, accept in attempts:
        try:
            status, body = _http_get(url, accept=accept, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - report, try next route
            notes.append(f"{label}: request failed ({type(exc).__name__}: {exc})")
            continue
        if status != 200:
            notes.append(f"{label}: HTTP {status}")
            continue
        text = body.decode("utf-8", errors="replace").strip()
        if text.startswith("@"):
            notes.append(f"{label}: BibTeX retrieved")
            return text, notes
        notes.append(f"{label}: response was not BibTeX (got {text[:60]!r}...)")
    return None, notes
