"""LangChain tool factory for reading a text file from disk.

Exposes a single `read_file` tool the agent can call when it needs the
contents of a specific file. Mirrors the factory pattern used in
`agent.history_rag.tool` and `agent.adapters.langchain.rag_tools`.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.config import AgentConfig

TOOL_NAME = "read_file"
TOOL_DESCRIPTION = (
    "Read the contents of a text file from disk. Accepts absolute or "
    "working-directory-relative paths. Rejects files larger than 1 MB. "
    "Returns a JSON object with `path`, `size`, and `content`. On failure "
    "returns a JSON object with an `error` field."
)

MAX_BYTES = 1_048_576


class ReadFileInput(BaseModel):
    """Input schema for the read_file tool."""

    path: str = Field(
        description="Absolute or cwd-relative path to a UTF-8 text file."
    )


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _read_file(path: str) -> str:
    resolved = Path(path).expanduser().resolve()

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

    def _run(path: str) -> str:
        return _read_file(path)

    _run.__name__ = TOOL_NAME

    return StructuredTool.from_function(
        func=_run,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=ReadFileInput,
        infer_schema=False,
    )
