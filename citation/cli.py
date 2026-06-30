"""CLI entry point for the isolated citation-capture prototype.

Run from the repo root::

    python -m citation.cli "幫我找關於檢索效率的文章"
    python -m citation.cli "papers on RAG citation hallucination" --auto

You give a natural-language request; the LLM decides the search queries itself
(e.g. it may search "BM25", "reranking latency", ...) and the candidate list is
extracted from what it actually retrieved.

This is a standalone experiment. It reuses the host project's config, chat
model, and Web Search MCP, but is NOT part of the agent graph or skill system.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from citation.capture import capture_citation
from citation.discovery import agentic_discover
from citation.models import CrossrefMatch, PaperCandidate
from citation.runtime import WebSearchUnavailable, build_runtime


async def _ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip()


def _print_candidates(candidates: list[PaperCandidate]) -> None:
    print(f"\nFound {len(candidates)} candidate paper(s):\n")
    for i, c in enumerate(candidates, start=1):
        print(f"  [{i}] {c.short_label()}")
        if c.doi:
            print(f"       DOI:    {c.doi}")
        if c.url:
            print(f"       URL:    {c.url}")
        if c.reason:
            print(f"       Reason: {c.reason}")
        print()


async def _select_candidate(
    candidates: list[PaperCandidate],
    *,
    auto: bool,
) -> PaperCandidate | None:
    if auto:
        print(f"[auto] selecting first candidate: {candidates[0].short_label()}")
        return candidates[0]
    while True:
        raw = await _ask(f"Select a paper [1-{len(candidates)}], or 'q' to quit: ")
        if raw.lower() in {"q", "quit", "exit", ""}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        print("  (invalid selection; enter one paper number)")


async def _confirm_crossref(
    candidate: PaperCandidate,
    matches: list[CrossrefMatch],
) -> str | None:
    print("\nCrossref returned multiple plausible matches — please confirm:\n")
    top = matches[:5]
    for i, m in enumerate(top, start=1):
        print(f"  [{i}] {m.title}")
        print(
            f"       DOI={m.doi}  year={m.year}  conf={m.confidence} "
            f"title_sim={m.title_similarity} year_match={m.year_matches} "
            f"author_overlap={m.author_overlap}"
        )
    while True:
        raw = await _ask(f"Choose match [1-{len(top)}], or 'n' to reject all: ")
        if raw.lower() in {"n", "no", "", "q"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(top):
            return top[int(raw) - 1].doi
        print("  (invalid selection)")


def _print_trace(result) -> None:
    print("---- capture trace ----")
    for note in result.notes:
        print(f"  - {note}")
    print("-----------------------\n")


def _make_progress_printer():
    started = time.perf_counter()

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - started
        print(f"[+{elapsed:5.1f}s] {message}", flush=True)

    return progress


async def _auto_capture(runtime, candidates, *, max_attempts, progress_cb=None):
    """Smoke-test mode: try ranked candidates until one captures cleanly.

    Auto mode never prompts, so an ambiguous Crossref match is refused (no
    confirm callback). Returns ``(result_or_none, chosen_candidate)``.
    """
    attempts = min(max_attempts, len(candidates))
    last_chosen = candidates[0]
    for i in range(attempts):
        chosen = candidates[i]
        last_chosen = chosen
        print(f"\n[auto] attempt {i + 1}/{attempts}: {chosen.short_label()}")
        result = await capture_citation(
            runtime, chosen, confirm_cb=None, progress_cb=progress_cb
        )
        _print_trace(result)
        if result.ok:
            return result, chosen
    return None, last_chosen


async def _run(args: argparse.Namespace) -> int:
    progress = _make_progress_printer()
    progress("runtime: loading config, model, and Web Search MCP")
    runtime = await build_runtime(load_mcp=True)
    progress(f"runtime: ready with {len(runtime.web_tools)} web-search tool(s)")

    if not runtime.web_tools:
        print(
            "ERROR: Web Search MCP is not available.\n"
            "  Enable it in .env: AGENT_ENABLE_MCP_WEB_SEARCH=1 and set\n"
            "  AGENT_MCP_WEB_SEARCH_COMMAND / AGENT_MCP_WEB_SEARCH_ARGS, then\n"
            "  make sure the MCP server starts (a writable ~/.cache helps).",
            file=sys.stderr,
        )
        return 2

    request = args.request or await _ask("What papers are you looking for? ")
    if not request.strip():
        print("ERROR: no request provided.", file=sys.stderr)
        return 2

    print(f"\nLetting the agent decide how to search for: {request!r} ...")
    if runtime.llm is None:
        print("  (no LLM configured — falling back to a single literal query)")
    try:
        candidates = await agentic_discover(
            runtime, request, limit=args.limit, progress_cb=progress
        )
    except WebSearchUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    progress(f"discovery: finished with {len(candidates)} candidate(s)")

    if not candidates:
        print(
            "No candidate papers found. The agent's searches returned no "
            "parseable results — try rephrasing or broadening the request.",
            file=sys.stderr,
        )
        return 3

    _print_candidates(candidates)

    if args.auto:
        result, chosen = await _auto_capture(
            runtime,
            candidates,
            max_attempts=args.auto_attempts,
            progress_cb=progress,
        )
        if result is None:
            print(
                f"FAILED: none of the top {min(args.auto_attempts, len(candidates))} "
                "candidates yielded a high-confidence citation. Nothing was written.",
                file=sys.stderr,
            )
            return 1
    else:
        chosen = await _select_candidate(candidates, auto=False)
        if chosen is None:
            print("No paper selected. Exiting.")
            return 0
        print(f"\nCapturing citation for: {chosen.short_label()} ...\n")
        result = await capture_citation(
            runtime,
            chosen,
            confirm_cb=_confirm_crossref,
            progress_cb=progress,
        )
        _print_trace(result)

    if result.ok:
        print(f"SUCCESS via {result.route} route.")
        print(f"  Paper: {chosen.short_label()}")
        if result.doi:
            print(f"  DOI:   {result.doi}")
        print(f"  File:  {result.out_path}")
        return 0

    print(
        "FAILED: no BibTeX could be captured for this paper. "
        "See the trace above for which step failed "
        "(MCP / Scholar / Crossref / DOI / write). Nothing was written.",
        file=sys.stderr,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m citation.cli",
        description=(
            "Isolated citation-capture prototype: the agent decides how to search "
            "(LLM-driven, scholar-oriented, via the Web Search MCP), then does "
            "selected-paper DOI resolution and Crossref/DOI BibTeX retrieval "
            "into citation/cite/."
        ),
    )
    parser.add_argument(
        "request", nargs="?",
        help="Natural-language request, e.g. '幫我找關於檢索效率的文章'. The agent "
             "decides the actual search queries.",
    )
    parser.add_argument(
        "--limit", type=int, default=6,
        help="Max candidate papers to present after ranking (default 6).",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Smoke-test mode: walk ranked candidates until one yields a "
             "high-confidence citation (never prompts).",
    )
    parser.add_argument(
        "--auto-attempts", type=int, default=4,
        help="In --auto mode, how many top candidates to try (default 4).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        rc = asyncio.run(_run(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
