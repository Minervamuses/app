"""LangChain tool factory for reading a text file from disk.

Exposes a single `read_file` tool the agent can call when it needs the
contents of a specific file. Mirrors the factory pattern used in
`agent.history_rag.tool` and `agent.adapters.langchain.rag_tools`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langgraph.prebuilt import InjectedState
from langchain_core.tools import StructuredTool
from pydantic import Field

from agent.config import AgentConfig

TOOL_NAME = "read_file"
TOOL_DESCRIPTION = (
    "Read the contents of a text file from disk. Accepts absolute or "
    "working-directory-relative paths. When a skill is active, relative paths "
    "starting with references/, assets/, or scripts/ are resolved only against "
    "the active skill root and do not fall back to the working directory. "
    "Other relative paths are resolved from the working directory. Rejects files larger than 1 MB. "
    "Returns a JSON object with `path`, `size`, and `content`. On failure "
    "returns a JSON object with an `error` field."
)

MAX_BYTES = 1_048_576
SKILL_RESOURCE_DIRS = frozenset({"references", "assets", "scripts"})
SENSITIVE_BASENAME_PREFIXES = ("credentials", "token", "secret", "secrets")


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _is_skill_resource_path(path: Path) -> bool:
    return bool(path.parts) and path.parts[0] in SKILL_RESOURCE_DIRS


def _would_escape_skill_root(path: Path, root: Path) -> bool:
    return not (root / path).resolve().is_relative_to(root)


def _is_sensitive_path(path: Path) -> bool:
    parts = {part.casefold() for part in path.parts}
    if ".ssh" in parts:
        return True

    name = path.name.casefold()
    if name == ".env" or name.startswith(".env."):
        return True
    if name == "id_rsa" or name.startswith("id_rsa."):
        return True
    return any(name.startswith(prefix) for prefix in SENSITIVE_BASENAME_PREFIXES)


def _read_file(path: str, skill_root: str | None = None) -> str:
    raw_path = Path(path).expanduser()
    if skill_root and not raw_path.is_absolute():
        root = Path(skill_root).expanduser().resolve()
        if _would_escape_skill_root(raw_path, root):
            return _error(f"path escapes active skill root: {path}")
        if _is_skill_resource_path(raw_path):
            return _read_resolved_file((root / raw_path).resolve())

    resolved = raw_path.resolve()
    return _read_resolved_file(resolved)


def _read_resolved_file(resolved: Path) -> str:

    if _is_sensitive_path(resolved):
        return _error("path blocked by sensitive denylist")

    if not resolved.exists():
        return _error(f"path does not exist: {resolved}")
    if not resolved.is_file():
        return _error(f"path is not a regular file: {resolved}")

    size = resolved.stat().st_size
    if size > MAX_BYTES:
        return _error(f"file too large: {size} bytes (limit {MAX_BYTES})")

    content = resolved.read_text(encoding="utf-8", errors="replace")
    return json.dumps(
        {"path": str(resolved), "size": size, "content": content},
        ensure_ascii=False,
    )


def create_read_file_tool(config: AgentConfig) -> StructuredTool:
    """Build the read_file tool. `config` accepted for factory symmetry."""
    del config

    def _run(
        path: Annotated[
            str,
            Field(description="Absolute or cwd-relative path to a UTF-8 text file."),
        ],
        state: Annotated[dict, InjectedState] | None = None,
    ) -> str:
        skill_root = state.get("skill_root") if isinstance(state, dict) else None
        return _read_file(path, skill_root=skill_root)

    _run.__name__ = TOOL_NAME

    return StructuredTool.from_function(
        func=_run,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        infer_schema=True,
    )
