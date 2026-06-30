"""Google Scholar-oriented discovery via the Web Search MCP.

Two discovery modes, both grounded in real tool output (never hallucinated):

  * :func:`agentic_discover` (default when an LLM is available) — the model is
    given the web-search tools and *decides the queries itself*. A natural
    language request like "找關於檢索效率的文章" leads the model to search terms
    it judges relevant (e.g. ``BM25``, ``learned sparse retrieval``,
    ``reranking latency``), possibly across several searches, then we extract
    candidates from everything it actually retrieved.
  * :func:`discover_candidates` (fallback) — a single deterministic
    scholar-oriented query, used when no LLM is configured.

Either way the LLM never writes BibTeX. We also do NOT scrape Google Scholar's
result pages directly (CAPTCHA-guarded, not a stable API); we bias normal web
search toward scholarly sources and surface what it returns.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import html
import json
import logging
import re
import time
import urllib.parse
from collections.abc import Callable

from citation.crossref import extract_doi
from citation.models import PaperCandidate
from citation.runtime import (
    FULL_SEARCH_TOOL,
    PAGE_CONTENT_TOOL,
    SUMMARIES_TOOL,
    CitationRuntime,
)

logger = logging.getLogger("citation.discovery")
ProgressCallback = Callable[[str], None]

# Anchors for parsing the mrkrsl/web-search-mcp text format:
#   **1. Title**
#   URL: https://...
#   Description: ...
_RESULT_HEAD = re.compile(r"^\*\*\s*(\d+)\.\s*(.+?)\s*\*\*\s*$")
_URL_LINE = re.compile(r"^URL:\s*(.+?)\s*$")
_DESC_LINE = re.compile(r"^Description:\s*(.*?)\s*$")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

# Scholarly hint appended to the user's topic. Bias the engines (Bing/Brave/
# DuckDuckGo) toward academic sources without pretending Scholar is an API.
_SCHOLAR_HINT = "research paper (site:scholar.google.com/scholar_lookup OR arxiv OR doi)"
_LLM_DISCOVERY_TIMEOUT_SECONDS = 60.0
_LLM_RANK_TIMEOUT_SECONDS = 30.0
_TOOL_TIMEOUT_SECONDS = 30.0


def _emit(progress_cb: ProgressCallback | None, message: str) -> None:
    if progress_cb is not None:
        progress_cb(message)


def build_scholar_query(topic: str) -> str:
    """Compose a scholar-oriented query from the user's free-text topic."""
    topic = topic.strip()
    return f"{topic} {_SCHOLAR_HINT}"


def clean_search_title(raw_title: str) -> str:
    """Turn a search-engine result label into a paper-ish title.

    The Web Search MCP returns display labels rather than structured paper
    titles, for example:

        arXiv arxiv.org › abs › 1706.03762   [1706.03762] Attention Is All You Need

    Keep the useful paper title part and drop obvious source breadcrumbs.
    """
    raw = (raw_title or "").strip()
    if not raw:
        return raw

    # Search labels often separate breadcrumbs from the actual result title
    # with multiple spaces. Keep the final segment when that shape appears.
    parts = [p.strip() for p in re.split(r"\s{2,}", raw) if p.strip()]
    title = parts[-1] if len(parts) > 1 else raw

    title = re.sub(r"\s+-\s+Google Scholar\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\[(?:pdf|html|book|citation)\]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\[[^\]]+\]\s*", "", title)

    # Publisher result labels commonly append venue/breadcrumb text after a
    # pipe. The left side is usually the paper title.
    if " | " in title:
        left = title.split(" | ", 1)[0].strip()
        if len(left) >= 8:
            title = left

    return re.sub(r"\s+", " ", title).strip() or raw


def _metadata_from_url(url: str | None) -> dict[str, object]:
    if not url:
        return {}
    parsed = urllib.parse.urlparse(html.unescape(url))
    qs = urllib.parse.parse_qs(parsed.query)
    meta: dict[str, object] = {}

    title = (qs.get("title") or [""])[0].strip()
    if title:
        meta["title"] = re.sub(r"\s+", " ", title)

    doi = (qs.get("doi") or [""])[0].strip()
    if doi:
        meta["doi"] = doi

    year = (qs.get("publication_year") or [""])[0].strip()
    if year.isdigit():
        meta["year"] = int(year)

    authors = [a.strip() for a in qs.get("author", []) if a.strip()]
    if authors:
        meta["authors"] = authors

    return meta


def _is_generic_search_title(title: str) -> bool:
    return (title or "").strip().lower() in {"google scholar", "scholar"}


def _is_author_profile_candidate(candidate: PaperCandidate) -> bool:
    if not candidate.url:
        return False
    parsed = urllib.parse.urlparse(html.unescape(candidate.url))
    return "scholar.google." in parsed.netloc and parsed.path.startswith("/citations")


def _is_obvious_nonpaper_candidate(candidate: PaperCandidate) -> bool:
    if candidate.doi:
        return False
    if not candidate.url:
        return False
    parsed = urllib.parse.urlparse(html.unescape(candidate.url))
    host = parsed.netloc.lower()
    return host.endswith("wikipedia.org")


def unwrap_redirect_url(url: str | None) -> str | None:
    """Resolve a search-engine redirect wrapper to its real destination.

    Bing returns ``https://www.bing.com/ck/a?...&u=a1<base64url>&...`` links;
    the actual target is base64url-encoded in the ``u`` param (after a 2-char
    prefix like ``a1``). Recovering it lets DOI extraction work on the real URL.
    Returns the original URL unchanged when there is nothing to unwrap.
    """
    if not url:
        return url
    try:
        parsed = urllib.parse.urlparse(url)
        if "bing.com" not in parsed.netloc or "/ck/a" not in parsed.path:
            return url
        u = urllib.parse.parse_qs(parsed.query).get("u", [""])[0]
        if not u:
            return url
        encoded = u[2:] if u[:2].isalnum() and len(u) > 2 else u
        padded = encoded + "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8", errors="strict")
        if decoded.startswith("http"):
            return decoded
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return url
    return url


def _coerce_text(result) -> str:
    """Coerce a LangChain MCP tool result into plain text."""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        parts: list[str] = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(result, dict) and result.get("type") == "text":
        return str(result.get("text", ""))
    return str(result)


def parse_summaries(text: str) -> list[PaperCandidate]:
    """Parse the MCP summaries text block into ordered candidates."""
    candidates: list[PaperCandidate] = []
    current: PaperCandidate | None = None
    collecting_desc = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        head = _RESULT_HEAD.match(line)
        if head:
            if current is not None:
                candidates.append(current)
            current = PaperCandidate(title=clean_search_title(head.group(2)))
            collecting_desc = False
            continue
        if current is None:
            continue
        url_m = _URL_LINE.match(line)
        if url_m:
            current.url = unwrap_redirect_url(url_m.group(1).strip())
            meta = _metadata_from_url(current.url)
            if meta.get("title") and _is_generic_search_title(current.title):
                current.title = str(meta["title"])
            if meta.get("doi"):
                current.doi = str(meta["doi"])
            if meta.get("year") and current.year is None:
                current.year = int(meta["year"])
            if meta.get("authors") and not current.authors:
                current.authors = list(meta["authors"])
            collecting_desc = False
            continue
        desc_m = _DESC_LINE.match(line)
        if desc_m:
            current.snippet = desc_m.group(1).strip()
            collecting_desc = True
            continue
        if line.strip() == "---":
            collecting_desc = False
            continue
        # Any other bold marker (**Full Content:**, **Status:**, ...) ends the
        # description; full-web-search appends large content blocks under these.
        if line.lstrip().startswith("**"):
            collecting_desc = False
            continue
        if collecting_desc and line.strip():
            current.snippet = (current.snippet + " " + line.strip()).strip()

    if current is not None:
        candidates.append(current)

    filtered: list[PaperCandidate] = []
    for cand in candidates:
        cand.doi = cand.doi or extract_doi(cand.url, cand.snippet, cand.title)
        if _is_author_profile_candidate(cand) or _is_obvious_nonpaper_candidate(cand):
            continue
        if cand.year is None:
            cand.year = _guess_year(f"{cand.title} {cand.snippet}")
        if not cand.reason:
            cand.reason = cand.snippet[:160]
        filtered.append(cand)
    return filtered


def _guess_year(text: str) -> int | None:
    matches = _YEAR.findall(text)
    if not matches:
        return None
    # findall returns the captured group ("19"/"20"); re-scan for full years.
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)]
    plausible = [y for y in years if 1950 <= y <= 2035]
    return max(plausible) if plausible else None


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


async def _enrich_with_llm(
    llm,
    topic: str,
    candidates: list[PaperCandidate],
    progress_cb: ProgressCallback | None = None,
) -> None:
    """Best-effort: ask the LLM to fill authors/year and a relevance reason.

    Mutates ``candidates`` in place. Any failure leaves the deterministic
    fields untouched — the LLM never invents a citation, only annotates.
    """
    if llm is None or not candidates:
        return
    payload = [
        {
            "index": i,
            "title": c.title,
            "url": c.url,
            "snippet": c.snippet,
        }
        for i, c in enumerate(candidates)
    ]
    system = (
        "You annotate web search results about an academic topic. "
        "For each result, extract author surnames and publication year ONLY if "
        "they are clearly present in the title/snippet; otherwise leave them "
        "empty/null. Never guess. Also give a one-sentence reason why the result "
        "matches the topic. Reply with a JSON array of objects: "
        '{"index": int, "authors": [str], "year": int|null, "reason": str}.'
    )
    human = f"Topic: {topic}\n\nResults:\n{json.dumps(payload, ensure_ascii=False)}"
    _emit(progress_cb, f"enrichment: asking model to annotate {len(candidates)} candidate(s)")
    started = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([("system", system), ("human", human)]),
            timeout=_LLM_RANK_TIMEOUT_SECONDS,
        )
        content = _strip_code_fence(getattr(resp, "content", "") or "")
        data = json.loads(content)
    except Exception as exc:  # noqa: BLE001 - enrichment is optional
        logger.warning("LLM enrichment failed (%s); using parsed fields only", exc)
        _emit(progress_cb, f"enrichment: failed/timed out after {time.perf_counter() - started:.1f}s; using parsed fields")
        return
    _emit(progress_cb, f"enrichment: model returned in {time.perf_counter() - started:.1f}s")
    by_index = {item.get("index"): item for item in data if isinstance(item, dict)}
    for i, cand in enumerate(candidates):
        item = by_index.get(i)
        if not item:
            continue
        authors = item.get("authors") or []
        if isinstance(authors, list):
            cand.authors = [str(a).strip() for a in authors if str(a).strip()]
        year = item.get("year")
        if isinstance(year, int) and 1950 <= year <= 2035:
            cand.year = year
        reason = item.get("reason")
        if isinstance(reason, str) and reason.strip():
            cand.reason = reason.strip()


async def discover_candidates(
    runtime: CitationRuntime,
    topic: str,
    *,
    limit: int = 6,
    progress_cb: ProgressCallback | None = None,
) -> list[PaperCandidate]:
    """Run one scholar-oriented search and return parsed, enriched candidates.

    Raises:
        WebSearchUnavailable: if the summaries tool is not loaded.
    """
    tool = runtime.require_web_tool(SUMMARIES_TOOL)
    query = build_scholar_query(topic)
    limit = max(1, min(int(limit), 10))
    logger.info("discovery query: %s (limit=%d)", query, limit)
    _emit(progress_cb, f"discovery: running fallback web search query={query!r}")
    started = time.perf_counter()
    result = await tool.ainvoke({"query": query, "limit": limit})
    _emit(progress_cb, f"discovery: fallback web search returned in {time.perf_counter() - started:.1f}s")
    text = _coerce_text(result)
    candidates = parse_summaries(text)
    _emit(progress_cb, f"discovery: parsed {len(candidates)} candidate(s)")
    await _enrich_with_llm(runtime.llm, topic, candidates, progress_cb=progress_cb)
    return candidates


# --- Agentic discovery: the LLM decides what to search ---------------------

_SEARCH_SYSTEM_PROMPT = """You are a research librarian finding ACADEMIC PAPERS.

The user describes what they want in natural language (often Chinese). YOU decide
the search queries — do not just echo their phrasing back as one query.

How to work:
- Break a broad request into a few specific, well-known search terms a
  researcher would actually use. Example: "retrieval efficiency" ->
  ["BM25 efficiency", "learned sparse retrieval", "dense retrieval ANN index",
  "passage reranking latency"]. Pick the angles you judge most relevant.
- Use the `get-web-search-summaries` tool with Google-Scholar-oriented queries.
  Prefer queries that include `site:scholar.google.com`, DOI/arXiv terms, or
  publisher/venue pages likely to expose a DOI. Issue SEVERAL different queries
  (typically 2-4) to broaden coverage. Do NOT bulk-crawl or hammer the engine.
- Most papers are indexed in English: prefer English query terms even when the
  request is in Chinese.
- Only use `get-single-web-page-content` if you must inspect a specific page.
- When you have searched enough to cover the topic, STOP calling tools and reply
  with a one-line note listing the queries you used. The host program extracts
  the candidate paper list from your search results — you do not need to list
  them yourself, and you must never invent papers."""


def _dedupe_candidates(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    """Drop duplicate candidates by normalized URL or title, keeping order."""
    seen_url: set[str] = set()
    seen_title: set[str] = set()
    out: list[PaperCandidate] = []
    for c in candidates:
        url_key = (c.url or "").split("?")[0].rstrip("/").lower()
        title_key = re.sub(r"\W+", " ", (c.title or "").lower()).strip()
        if url_key and url_key in seen_url:
            continue
        if title_key and title_key in seen_title:
            continue
        if url_key:
            seen_url.add(url_key)
        if title_key:
            seen_title.add(title_key)
        out.append(c)
    return out


async def _run_search_agent(
    runtime: CitationRuntime,
    user_request: str,
    *,
    max_rounds: int,
    max_tool_calls: int,
    progress_cb: ProgressCallback | None,
) -> tuple[list[str], list[str]]:
    """Tool-calling loop: the LLM picks queries and runs the web-search tools.

    Returns ``(result_texts, queries_used)`` where ``result_texts`` are the raw
    search-result blocks (for deterministic candidate extraction). Isolated to
    this prototype — it does NOT use the agent graph or skill machinery.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    search_tools = [
        tool
        for name, tool in runtime.web_tools.items()
        if name in (SUMMARIES_TOOL, FULL_SEARCH_TOOL, PAGE_CONTENT_TOOL)
    ]
    llm_with_tools = runtime.llm.bind_tools(search_tools)

    messages: list = [
        SystemMessage(content=_SEARCH_SYSTEM_PROMPT),
        HumanMessage(content=user_request),
    ]
    result_texts: list[str] = []
    queries_used: list[str] = []
    calls = 0

    for _round in range(max_rounds):
        _emit(progress_cb, f"discovery: asking model for search step {_round + 1}")
        started = time.perf_counter()
        try:
            ai = await asyncio.wait_for(
                llm_with_tools.ainvoke(messages),
                timeout=_LLM_DISCOVERY_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to collected results
            logger.warning("search-agent LLM call failed/timed out: %s", exc)
            _emit(progress_cb, f"discovery: model search step failed/timed out after {time.perf_counter() - started:.1f}s")
            break
        _emit(progress_cb, f"discovery: model search step returned in {time.perf_counter() - started:.1f}s")
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            tc_id = tc.get("id")
            name = tc.get("name", "")
            args = tc.get("args", {}) or {}
            if calls >= max_tool_calls:
                messages.append(ToolMessage(
                    content="(search budget exhausted; stop calling tools)",
                    tool_call_id=tc_id, name=name,
                ))
                continue
            tool = runtime.web_tools.get(name)
            if tool is None:
                messages.append(ToolMessage(
                    content=f"(tool {name!r} is not available)",
                    tool_call_id=tc_id, name=name,
                ))
                continue
            calls += 1
            query = args.get("query")
            if query:
                _emit(progress_cb, f"discovery: running {name} query={str(query)!r}")
            else:
                _emit(progress_cb, f"discovery: running {name}")
            started = time.perf_counter()
            try:
                text = _coerce_text(
                    await asyncio.wait_for(
                        tool.ainvoke(args),
                        timeout=_TOOL_TIMEOUT_SECONDS,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad call must not abort
                text = f"(search failed: {type(exc).__name__}: {exc})"
                _emit(progress_cb, f"discovery: {name} failed after {time.perf_counter() - started:.1f}s")
            else:
                _emit(progress_cb, f"discovery: {name} returned in {time.perf_counter() - started:.1f}s")
            messages.append(ToolMessage(
                content=text[:8000], tool_call_id=tc_id, name=name,
            ))
            if name in (SUMMARIES_TOOL, FULL_SEARCH_TOOL):
                result_texts.append(text)
                q = args.get("query")
                if q:
                    queries_used.append(str(q))
        if calls >= max_tool_calls:
            break

    logger.info("agentic discovery queries: %s", queries_used or "(none)")
    return result_texts, queries_used


async def _annotate_and_rank(
    llm,
    user_request: str,
    candidates: list[PaperCandidate],
    limit: int,
    progress_cb: ProgressCallback | None,
) -> list[PaperCandidate]:
    """Annotate (author/year/reason) and rank candidates against the request.

    One LLM call. On any failure, fall back to plain enrichment + discovery
    order so discovery still returns usable candidates.
    """
    if not candidates:
        return []
    payload = [
        {"index": i, "title": c.title, "url": c.url, "snippet": c.snippet[:200]}
        for i, c in enumerate(candidates)
    ]
    system = (
        "You are ranking web search results against a user's paper request. "
        "For each result extract author surnames and year ONLY if clearly present "
        "(never guess), write a one-sentence reason it matches the request, and a "
        "relevance score 0.0-1.0. Reply with a JSON array of objects: "
        '{"index": int, "authors": [str], "year": int|null, "reason": str, '
        '"relevance": float}.'
    )
    human = f"Request: {user_request}\n\nResults:\n{json.dumps(payload, ensure_ascii=False)}"
    _emit(progress_cb, f"ranking: asking model to rank {len(candidates)} candidate(s)")
    started = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([("system", system), ("human", human)]),
            timeout=_LLM_RANK_TIMEOUT_SECONDS,
        )
        data = json.loads(_strip_code_fence(getattr(resp, "content", "") or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("annotate/rank failed (%s); using discovery order", exc)
        _emit(progress_cb, f"ranking: failed/timed out after {time.perf_counter() - started:.1f}s; using discovery order")
        return candidates[:limit]
    _emit(progress_cb, f"ranking: model returned in {time.perf_counter() - started:.1f}s")

    scored: list[tuple[float, PaperCandidate]] = []
    by_index = {item.get("index"): item for item in data if isinstance(item, dict)}
    for i, cand in enumerate(candidates):
        item = by_index.get(i)
        relevance = 0.0
        if item:
            authors = item.get("authors") or []
            if isinstance(authors, list):
                cand.authors = [str(a).strip() for a in authors if str(a).strip()]
            year = item.get("year")
            if isinstance(year, int) and 1950 <= year <= 2035:
                cand.year = year
            reason = item.get("reason")
            if isinstance(reason, str) and reason.strip():
                cand.reason = reason.strip()
            try:
                relevance = float(item.get("relevance", 0.0))
            except (TypeError, ValueError):
                relevance = 0.0
        scored.append((relevance, cand))

    # Keep only plausibly-relevant items; if the model scored nothing, keep all.
    relevant = [c for r, c in sorted(scored, key=lambda t: t[0], reverse=True) if r >= 0.15]
    ordered = relevant or [c for _r, c in sorted(scored, key=lambda t: t[0], reverse=True)]
    return ordered[:limit]


async def agentic_discover(
    runtime: CitationRuntime,
    user_request: str,
    *,
    limit: int = 6,
    max_rounds: int = 6,
    max_tool_calls: int = 8,
    pool_cap: int = 20,
    progress_cb: ProgressCallback | None = None,
) -> list[PaperCandidate]:
    """LLM-driven discovery: the model chooses the queries, we ground the result.

    Falls back to the single-query :func:`discover_candidates` when no LLM is
    configured or the model issued no usable searches.

    Raises:
        WebSearchUnavailable: if the web-search summaries tool is not loaded.
    """
    runtime.require_web_tool(SUMMARIES_TOOL)
    if runtime.llm is None:
        logger.info("no LLM configured; using single-query discovery")
        _emit(progress_cb, "discovery: no LLM configured; using fallback search")
        return await discover_candidates(
            runtime, user_request, limit=limit, progress_cb=progress_cb
        )

    result_texts, _queries = await _run_search_agent(
        runtime,
        user_request,
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
        progress_cb=progress_cb,
    )

    pool: list[PaperCandidate] = []
    for text in result_texts:
        pool.extend(parse_summaries(text))
    pool = _dedupe_candidates(pool)[:pool_cap]
    _emit(progress_cb, f"discovery: collected {len(pool)} unique candidate(s)")

    if not pool:
        logger.info("agent issued no usable searches; falling back to single query")
        _emit(progress_cb, "discovery: agent search produced no parseable candidates; using fallback search")
        return await discover_candidates(
            runtime, user_request, limit=limit, progress_cb=progress_cb
        )

    return await _annotate_and_rank(
        runtime.llm, user_request, pool, limit, progress_cb=progress_cb
    )
