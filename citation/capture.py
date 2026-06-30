"""Citation-capture orchestration for one chosen paper.

Route order (faithful to the spec, never fabricating):
  1. Resolve a DOI — from the discovery snippet/URL, then the source page,
     then a verified Crossref bibliographic search.
  2. With a trusted DOI, retrieve BibTeX via the Crossref/DOI route.
  3. If that fails, try to resolve an alternate DOI and retry the same
     Crossref/DOI route.
  4. If every DOI/Crossref route fails, write nothing and return the failure
     trace.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Awaitable

from citation import bibtex
from citation.crossref import (
    classify_matches,
    extract_doi,
    fetch_bibtex_for_doi,
    rank_matches,
    search_crossref,
)
from citation.models import CaptureResult, CrossrefMatch, PaperCandidate
from citation.scholar_fallback import inspect_source_page, try_scholar_doi
from citation.runtime import CitationRuntime

logger = logging.getLogger("citation.capture")
ProgressCallback = Callable[[str], None]

# Async callback used to resolve an ambiguous Crossref result interactively.
# Receives the candidate + ranked matches; returns the chosen DOI or None.
ConfirmCallback = Callable[
    [PaperCandidate, list[CrossrefMatch]], Awaitable[str | None]
]


def _emit(progress_cb: ProgressCallback | None, message: str) -> None:
    if progress_cb is not None:
        progress_cb(message)


async def _resolve_doi(
    runtime: CitationRuntime,
    candidate: PaperCandidate,
    *,
    confirm_cb: ConfirmCallback | None,
    result: CaptureResult,
    allow_direct: bool = True,
    exclude: set[str] | None = None,
    progress_cb: ProgressCallback | None = None,
) -> str | None:
    """Resolve a trustworthy DOI for the selected candidate."""
    exclude = exclude or set()

    # 1. DOI sitting in the discovery snippet/URL.
    if allow_direct:
        _emit(progress_cb, "doi: checking selected candidate URL/snippet")
        doi = candidate.doi or extract_doi(candidate.url, candidate.snippet)
        if doi:
            if _doi_key(doi) not in exclude:
                result.notes.append(f"DOI taken from discovery result: {doi}")
                return doi
            result.notes.append(f"discovery DOI already failed, skipping: {doi}")

    # 2. DOI (or inline BibTeX) on the candidate's own landing page.
    _emit(progress_cb, "doi: inspecting selected paper source page")
    started = time.perf_counter()
    page_doi, inline_bibtex, page_notes = await inspect_source_page(runtime, candidate)
    _emit(progress_cb, f"doi: source-page inspection finished in {time.perf_counter() - started:.1f}s")
    result.notes.extend(page_notes)
    if page_doi:
        if _doi_key(page_doi) not in exclude:
            return page_doi
        result.notes.append(f"source-page DOI already failed, skipping: {page_doi}")
    if inline_bibtex:
        result.notes.append(
            "inline BibTeX ignored: selected-paper capture must use DOI/Crossref"
        )

    # 3. Crossref bibliographic search, verified locally against title/year/author.
    if not candidate.title.strip():
        result.notes.append("no title available for Crossref search")
        return await _resolve_doi_from_scholar(
            runtime, candidate, result, exclude, progress_cb=progress_cb
        )
    try:
        _emit(progress_cb, f"doi: searching Crossref by title={candidate.title!r}")
        started = time.perf_counter()
        items = await asyncio.to_thread(search_crossref, candidate.title, rows=5)
        _emit(progress_cb, f"doi: Crossref title search returned in {time.perf_counter() - started:.1f}s")
    except Exception as exc:  # noqa: BLE001
        result.notes.append(f"Crossref search failed: {type(exc).__name__}: {exc}")
        return await _resolve_doi_from_scholar(
            runtime, candidate, result, exclude, progress_cb=progress_cb
        )

    matches = rank_matches(candidate, items)
    matches = [m for m in matches if _doi_key(m.doi) not in exclude]
    if not matches:
        result.notes.append("Crossref returned no usable records for this title")
        return await _resolve_doi_from_scholar(
            runtime, candidate, result, exclude, progress_cb=progress_cb
        )

    tier, best = classify_matches(matches)
    result.notes.append(
        f"Crossref best match: tier={tier} confidence={best.confidence} "
        f"title_sim={best.title_similarity} year_match={best.year_matches} "
        f"author_overlap={best.author_overlap} doi={best.doi!r} "
        f"title={best.title!r}"
    )

    if tier == "high":
        return best.doi
    if tier == "ambiguous":
        if confirm_cb is None:
            result.notes.append(
                "Crossref match is ambiguous and no interactive confirmation is "
                "available (auto mode) — refusing to guess a DOI"
            )
            return await _resolve_doi_from_scholar(
                runtime, candidate, result, exclude, progress_cb=progress_cb
            )
        chosen = await confirm_cb(candidate, matches)
        if chosen:
            result.notes.append(f"user confirmed Crossref DOI: {chosen}")
            return chosen
        result.notes.append("user declined the ambiguous Crossref matches")
        return await _resolve_doi_from_scholar(
            runtime, candidate, result, exclude, progress_cb=progress_cb
        )

    result.notes.append("Crossref best match too weak to trust (no confident DOI)")
    return await _resolve_doi_from_scholar(
        runtime, candidate, result, exclude, progress_cb=progress_cb
    )


def _doi_key(doi: str | None) -> str:
    return (doi or "").strip().lower()


async def _resolve_doi_from_scholar(
    runtime: CitationRuntime,
    candidate: PaperCandidate,
    result: CaptureResult,
    exclude: set[str],
    progress_cb: ProgressCallback | None = None,
) -> str | None:
    _emit(progress_cb, "doi: running Scholar-oriented DOI lookup")
    started = time.perf_counter()
    doi, notes = await try_scholar_doi(runtime, candidate)
    _emit(progress_cb, f"doi: Scholar-oriented DOI lookup finished in {time.perf_counter() - started:.1f}s")
    result.notes.extend(notes)
    if doi and _doi_key(doi) not in exclude:
        return doi
    if doi:
        result.notes.append(f"Scholar DOI already failed, skipping: {doi}")
    return None


async def _try_crossref_bibtex(
    runtime: CitationRuntime,
    candidate: PaperCandidate,
    doi: str,
    *,
    result: CaptureResult,
    progress_cb: ProgressCallback | None = None,
) -> CaptureResult | None:
    result.doi = doi
    _emit(progress_cb, f"bibtex: retrieving BibTeX for DOI {doi}")
    started = time.perf_counter()
    bib, notes = await asyncio.to_thread(fetch_bibtex_for_doi, doi)
    _emit(progress_cb, f"bibtex: DOI retrieval finished in {time.perf_counter() - started:.1f}s")
    result.notes.extend(notes)
    if bib and bibtex.looks_like_bibtex(bib):
        return _finalize(runtime, candidate, bib, doi=doi, route="crossref", result=result)
    result.notes.append("DOI/Crossref BibTeX route did not yield valid BibTeX")
    return None


async def capture_citation(
    runtime: CitationRuntime,
    candidate: PaperCandidate,
    *,
    confirm_cb: ConfirmCallback | None = None,
    progress_cb: ProgressCallback | None = None,
) -> CaptureResult:
    """Attempt to capture BibTeX for ``candidate`` and write it to ``cite/``."""
    result = CaptureResult(ok=False)

    doi = await _resolve_doi(
        runtime,
        candidate,
        confirm_cb=confirm_cb,
        result=result,
        progress_cb=progress_cb,
    )

    attempted_dois: set[str] = set()
    if doi:
        attempted_dois.add(_doi_key(doi))
        captured = await _try_crossref_bibtex(
            runtime, candidate, doi, result=result, progress_cb=progress_cb
        )
        if captured is not None:
            return captured
    else:
        result.notes.append("no trustworthy DOI resolved; skipping DOI/Crossref BibTeX")

    if attempted_dois:
        alt_doi = await _resolve_doi(
            runtime,
            candidate,
            confirm_cb=confirm_cb,
            result=result,
            allow_direct=False,
            exclude=attempted_dois,
            progress_cb=progress_cb,
        )
        if alt_doi:
            captured = await _try_crossref_bibtex(
                runtime, candidate, alt_doi, result=result, progress_cb=progress_cb
            )
            if captured is not None:
                return captured

    result.notes.append(
        "ALL DOI/Crossref routes failed — no BibTeX captured. Nothing was written."
    )
    return result


def _finalize(
    runtime: CitationRuntime,
    candidate: PaperCandidate,
    bib: str,
    *,
    doi: str | None,
    route: str,
    result: CaptureResult,
) -> CaptureResult:
    # Prefer the authoritative title from the retrieved BibTeX over the (often
    # breadcrumb-laden) discovery label, so filenames are clean.
    title_for_name = bibtex.extract_bibtex_title(bib) or candidate.title
    out_path = bibtex.write_bibtex(runtime.cite_dir, title_for_name, bib)
    result.ok = True
    result.bibtex = bib
    result.doi = doi
    result.route = route
    result.out_path = str(out_path)
    result.notes.append(f"BibTeX written via {route} route -> {out_path}")
    return result
