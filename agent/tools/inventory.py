"""Single source of truth for the agent's local base tool inventory.

This module owns three things so prompts, graph binding, skill policy, and
evaluators cannot drift apart:

1. the declarative metadata for the local base tools (knowledge-base search,
   chat-history recall, file reading, shell);
2. the ordered tool-name lists consumed by the graph, the session, and the
   evaluators;
3. the prompt block describing those tools, their selection policy, and the
   base workflow.

Only :func:`build_base_tools` instantiates tools (and the ``recall_history``
store). :func:`base_tool_names`, :func:`behavior_tool_names`, and
:func:`render_base_tool_prompt` read static metadata only, so importing this
module or rendering the prompt never touches Chroma, the history store, or any
external service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent.adapters.langchain.rag_tools import create_rag_tools
from agent.config import AgentConfig
from agent.history_rag import create_history_tool
from agent.tools.bash import create_bash_tool
from agent.tools.read_file import create_read_file_tool


@dataclass(frozen=True)
class BaseToolDoc:
    """Static description of one local base tool."""

    name: str
    family: str
    section: str
    description: str


# Declarative metadata: the ordered local base tools. Order here is the single
# source for graph binding order, name lists, and the rendered prompt.
BASE_TOOL_DOCS: tuple[BaseToolDoc, ...] = (
    BaseToolDoc(
        name="rag_explore",
        family="rag",
        section="Local knowledge base tools (always available):",
        description=(
            "Discover what's in the indexed knowledge base: categories, tags, "
            "date ranges, folder summaries.\n"
            "   Use this first when you're unsure what the knowledge base contains."
        ),
    ),
    BaseToolDoc(
        name="rag_search",
        family="rag",
        section="Local knowledge base tools (always available):",
        description=(
            "Semantic search with optional filters (folder_prefix, category, "
            "file_type, date range).\n"
            "   Use specific queries. You can search multiple times with "
            "different queries or filters."
        ),
    ),
    BaseToolDoc(
        name="rag_get_context",
        family="rag",
        section="Local knowledge base tools (always available):",
        description=(
            "Expand a search result by retrieving surrounding chunks from the "
            "same file.\n"
            "   Use when a result looks relevant but you need more context."
        ),
    ),
    BaseToolDoc(
        name="recall_history",
        family="history",
        section="Conversation history tool (always available):",
        description=(
            "Search persisted prior chat turns from this user, including older "
            "parts of the current session and previous CLI sessions. Each user "
            "prompt and each assistant response is stored as a separate entry; "
            "results carry role, turn_id, and timestamp.\n"
            "   Use when the user references earlier chat content that you "
            "cannot see in the current prompt.\n"
            "   Do NOT call this for content already visible in the current "
            "conversation.\n"
            "   Do NOT use this as a substitute for rag_search on general "
            "knowledge questions."
        ),
    ),
    BaseToolDoc(
        name="read_file",
        family="file",
        section="Local file-reading tool (always available):",
        description=(
            "Read a local UTF-8 text file from an absolute or cwd-relative path.\n"
            "   Use this for local drafts, notes, reviewer comments, journal "
            "guidelines, and other local text files."
        ),
    ),
    BaseToolDoc(
        name="bash",
        family="shell",
        section="Shell tool (always available, but every call is gated):",
        description=(
            "Execute a shell command. EVERY CALL PROMPTS THE USER FOR APPROVAL "
            "before\n"
            "   execution; the user reads the `description` you supply to decide "
            "whether to allow it.\n"
            "   Use only when no narrower tool fits — for example, listing or "
            "finding files when the\n"
            "   path is unknown (`ls`, `find`), quick disk inspection (`wc`, "
            "`du`, `head`), or one-off\n"
            "   pipelines the user explicitly asked for. Prefer `read_file` / "
            "`rag_search` when they\n"
            "   apply. Always include a one-sentence `description` explaining "
            "your intent — vague\n"
            "   descriptions will be rejected by the user. If a call returns "
            "`approved: false`, do\n"
            "   not retry the same command; ask the user for a more specific "
            "path or alternative."
        ),
    ),
)

RAG_TOOL_NAMES: tuple[str, ...] = tuple(
    doc.name for doc in BASE_TOOL_DOCS if doc.family == "rag"
)
HISTORY_TOOL_NAMES: tuple[str, ...] = tuple(
    doc.name for doc in BASE_TOOL_DOCS if doc.family == "history"
)
FILE_TOOL_NAMES: tuple[str, ...] = tuple(
    doc.name for doc in BASE_TOOL_DOCS if doc.family == "file"
)
SHELL_TOOL_NAMES: tuple[str, ...] = tuple(
    doc.name for doc in BASE_TOOL_DOCS if doc.family == "shell"
)
# Tools the behavior evaluator classifies as "local" (non-RAG, non-history).
LOCAL_TOOL_NAMES: tuple[str, ...] = FILE_TOOL_NAMES + SHELL_TOOL_NAMES
# Web behavior tool names are frozen here so the evaluator taxonomy keeps a
# stable universe even though these tools are provided by MCP at runtime.
WEB_BEHAVIOR_TOOL_NAMES: tuple[str, ...] = (
    "full-web-search",
    "get-web-search-summaries",
    "get-single-web-page-content",
)

_BASE_TOOL_NAMES: tuple[str, ...] = tuple(doc.name for doc in BASE_TOOL_DOCS)

_TOOL_SELECTION_POLICY = """Tool selection policy:
- Questions about the indexed project or research notes → prefer `rag_explore` / `rag_search` / `rag_get_context`.
- Questions about earlier chat history that is no longer visible → prefer `recall_history`.
- Questions about local files → prefer `read_file`.
- Filesystem enumeration or shell ops the user explicitly asked for → use `bash` (always with a clear description).
- Questions needing live external information → prefer Web Search MCP.
- Questions about remote GitHub repos, PRs, issues, or Actions → prefer GitHub MCP.
- If a tool family is not listed in the bound tools for this session, treat it as unavailable and fall back to what you have."""

_BASE_WORKFLOW = """Workflow:
- If the question is vague or you don't know the structure of the knowledge base, start with rag_explore.
- Use rag_search with appropriate filters based on what you learned from rag_explore.
- Use rag_get_context only when a result is clearly relevant but you need more of its surrounding text.
- Use read_file when the answer depends on a local file.
- After at most 1-3 rag_search calls, stop searching and synthesize your answer. Don't keep searching for perfection.
- Give up gracefully: if the search results are empty, repetitive, or unrelated to the question, do NOT keep re-searching and do NOT call rag_get_context on irrelevant results. Instead, stop and state plainly that the indexed knowledge base does not contain enough evidence to answer.
- Do NOT make up information. Only answer based on tool results or your conversation with the user."""


def _dedupe(names: Iterable[str]) -> list[str]:
    """Order-preserving de-duplication of tool names."""
    return list(dict.fromkeys(name for name in names if name))


def _extra_tool_names(extra_tools: list | None) -> list[str]:
    return [getattr(tool, "name", str(tool)) for tool in (extra_tools or [])]


def base_tool_names(extra_tools: list | None = None) -> list[str]:
    """Return the ordered local base tool names plus any extra tool names.

    Local base tools always win on name collisions: a same-named extra tool is
    dropped so the bound graph, the prompt, and the evaluators agree.
    """
    return _dedupe([*_BASE_TOOL_NAMES, *_extra_tool_names(extra_tools)])


def behavior_tool_names(extra_tools: list | None = None) -> list[str]:
    """Return the tool-name universe scored by the behavior evaluator.

    This is the local base behavior tools plus the frozen web behavior tool
    names (provided by MCP at runtime), then any extra tool names. Keeping the
    web names here prevents the RAG/WEB/ALL forbidden universes from shrinking
    when MCP tools are not loaded.
    """
    return _dedupe(
        [*_BASE_TOOL_NAMES, *WEB_BEHAVIOR_TOOL_NAMES, *_extra_tool_names(extra_tools)]
    )


def build_base_tools(
    config: AgentConfig,
    history_store=None,
    extra_tools: list | None = None,
) -> list:
    """Instantiate the local base tools, then append de-duplicated extra tools.

    The returned tool list is ordered to match :func:`base_tool_names`. Local
    base tools win on name collisions, so a same-named extra tool is ignored
    rather than bound twice.
    """
    tools = [
        *create_rag_tools(config),
        create_history_tool(config, store=history_store),
        create_read_file_tool(config),
        create_bash_tool(config),
    ]
    seen = {getattr(tool, "name", str(tool)) for tool in tools}
    for extra in extra_tools or []:
        name = getattr(extra, "name", str(extra))
        if name in seen:
            continue
        seen.add(name)
        tools.append(extra)
    return tools


def render_base_tool_prompt() -> str:
    """Render the base tool descriptions, selection policy, and workflow.

    Reads only the static :data:`BASE_TOOL_DOCS` metadata and the policy/
    workflow text. It does not call :func:`build_base_tools`, instantiate the
    ``recall_history`` tool, or read any store.
    """
    blocks: list[str] = []
    last_section: str | None = None
    for index, doc in enumerate(BASE_TOOL_DOCS, start=1):
        if doc.section != last_section:
            blocks.append(doc.section)
            last_section = doc.section
        blocks.append(f"{index}. **{doc.name}** — {doc.description}")
    body = "\n\n".join(blocks)
    return "\n\n".join([body, _TOOL_SELECTION_POLICY, _BASE_WORKFLOW])
