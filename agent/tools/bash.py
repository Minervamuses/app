"""LangChain tool factory for executing shell commands with user approval.

Every call prompts the user for y/n approval before executing. Non-interactive
environments (eval harness, pytest, piped stdin) auto-deny so that the tool
cannot hang or silently execute commands without supervision.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.config import AgentConfig
from agent.paths import find_app_root

TOOL_NAME = "bash"
TOOL_DESCRIPTION = (
    "Execute a shell command. EVERY CALL PROMPTS THE USER FOR APPROVAL "
    "before execution; the user reads `description` to decide. Use only "
    "when no narrower tool fits (e.g., listing/finding files when path is "
    "unknown, or quick disk inspection like ls/find/wc). Prefer read_file "
    "or rag_search whenever they apply. Always include a `description` "
    "explaining your intent in one sentence; vague descriptions get denied."
)

MAX_OUTPUT_BYTES = 256_000
DEFAULT_TIMEOUT_SEC = 30
MAX_TIMEOUT_SEC = 300


class BashInput(BaseModel):
    """Input schema for the bash tool."""

    command: str = Field(description="The shell command to execute.")
    description: str = Field(
        description=(
            "One-sentence human-readable explanation of why this command "
            "is being run; the user reads this to decide whether to approve."
        )
    )
    timeout_sec: int = Field(
        default=DEFAULT_TIMEOUT_SEC,
        description=(
            f"Seconds to wait before killing the command "
            f"(default {DEFAULT_TIMEOUT_SEC}, max {MAX_TIMEOUT_SEC})."
        ),
    )


def _denied(reason: str, command: str) -> str:
    return json.dumps(
        {"approved": False, "error": reason, "command": command},
        ensure_ascii=False,
    )


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_OUTPUT_BYTES:
        return text
    head = encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    return f"{head}\n[truncated, original {len(encoded)} bytes]"


def _render_approval_prompt(command: str, description: str) -> str:
    bar = "─" * 60
    return (
        f"\n{bar}\n"
        f"[bash] Agent wants to run a shell command.\n\n"
        f"  Why: {description or '(no description provided)'}\n"
        f"  Cmd: {command}\n\n"
        f"Approve? [y/N] (Enter = no) "
    )


def _user_approves(command: str, description: str) -> bool:
    """Show approval prompt on stderr, read y/n from stdin.

    Default-deny on empty input or anything other than y/yes (case-insensitive).
    Auto-deny when stdin is not a TTY (e.g., eval harness, CI, piped chat).
    """
    if not sys.stdin.isatty():
        return False
    sys.stderr.write(_render_approval_prompt(command, description))
    sys.stderr.flush()
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _run_bash(command: str, description: str, timeout_sec: int) -> str:
    if not sys.stdin.isatty():
        return _denied("non-interactive environment; bash auto-denied", command)

    if not _user_approves(command, description):
        return _denied(f"user denied execution of: {command}", command)

    capped_timeout = max(1, min(int(timeout_sec or DEFAULT_TIMEOUT_SEC), MAX_TIMEOUT_SEC))
    cwd = str(find_app_root())

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=capped_timeout,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "approved": True,
                "error": f"timeout after {capped_timeout}s",
                "command": command,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "approved": True,
            "exit_code": completed.returncode,
            "stdout": _truncate(completed.stdout or ""),
            "stderr": _truncate(completed.stderr or ""),
            "command": command,
            "cwd": cwd,
        },
        ensure_ascii=False,
    )


def create_bash_tool(config: AgentConfig) -> StructuredTool:
    """Build the bash tool. `config` accepted for factory symmetry."""
    del config

    def _run(command: str, description: str, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> str:
        return _run_bash(command, description, timeout_sec)

    _run.__name__ = TOOL_NAME

    return StructuredTool.from_function(
        func=_run,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=BashInput,
        infer_schema=False,
    )
