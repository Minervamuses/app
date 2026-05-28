"""Pure deterministic scoring for tool routing expectations."""

from __future__ import annotations

RAG_TOOL_NAMES = ("rag_explore", "rag_search", "rag_get_context")
HISTORY_TOOL_NAMES = ("recall_history",)
WEB_TOOL_NAMES = (
    "full-web-search",
    "get-web-search-summaries",
    "get-single-web-page-content",
)
ALL_BEHAVIOR_TOOL_NAMES = (*RAG_TOOL_NAMES, *HISTORY_TOOL_NAMES, *WEB_TOOL_NAMES)
TOOL_FAMILIES = {
    "rag": set(RAG_TOOL_NAMES),
    "history": set(HISTORY_TOOL_NAMES),
    "web": set(WEB_TOOL_NAMES),
}
RAG_FORBIDDEN = [*HISTORY_TOOL_NAMES, *WEB_TOOL_NAMES]
HISTORY_FORBIDDEN = [*RAG_TOOL_NAMES, *WEB_TOOL_NAMES]
WEB_FORBIDDEN = [*RAG_TOOL_NAMES, *HISTORY_TOOL_NAMES]


def missing_required_tools(case: dict, available: set[str]) -> set[str]:
    """Return required tools the agent cannot call in this run.

    Hard requirements: ``expected_first_tool`` (when not None) and every
    entry of ``expected_tools_include`` must be loaded. The ``_in`` /
    ``_any_of`` sets are satisfied if at least one entry is loaded, so they
    only flag missing when every listed tool is absent.
    """
    missing: set[str] = set()
    first = case.get("expected_first_tool")
    if first and first not in available:
        missing.add(first)
    for tool in case.get("expected_tools_include", []) or []:
        if tool not in available:
            missing.add(tool)

    for field in ("expected_first_tool_in", "expected_tools_any_of"):
        options = list(case.get(field, []) or [])
        if options and not any(opt in available for opt in options):
            missing.update(options)
    return missing


def score_tool_expectations(
    case: dict,
    actual_tools: list[str],
    actual_args: list[dict],
) -> dict[str, bool]:
    """Score one expected tool-routing case against an actual tool trace."""
    scores: dict[str, bool] = {}
    actual_count = len(actual_tools)

    expected_first = case.get("expected_first_tool")
    allowed_first = list(case.get("expected_first_tool_in", []))
    if expected_first is not None:
        allowed_first.append(expected_first)

    if allowed_first:
        scores["first_tool"] = (actual_tools[0] in allowed_first) if actual_tools else False
    elif "expected_first_tool" in case:
        scores["no_tool"] = actual_count == 0

    expected_count = case.get("expected_tool_count", {})
    if expected_count:
        scores["count_ok"] = expected_count["min"] <= actual_count <= expected_count["max"]

    expected_include = case.get("expected_tools_include", [])
    if expected_include:
        scores["tools_covered"] = all(t in actual_tools for t in expected_include)

    expected_any = case.get("expected_tools_any_of", [])
    if expected_any:
        scores["tools_any"] = any(t in actual_tools for t in expected_any)

    expected_forbidden = case.get("expected_tools_forbidden", [])
    if expected_forbidden:
        scores["forbidden_ok"] = not any(t in actual_tools for t in expected_forbidden)

    expected_family = case.get("expected_tool_family")
    if expected_family:
        family_tools = TOOL_FAMILIES.get(expected_family, set())
        scores["tool_family"] = any(t in family_tools for t in actual_tools)

    expected_filters = case.get("expected_filters_include", [])
    if expected_filters:
        search_args = [a for t, a in zip(actual_tools, actual_args) if t == "rag_search"]
        scores["filters_used"] = (
            any(all(f in args for f in expected_filters) for args in search_args)
            if search_args else False
        )

    return scores
