"""CLI entry point for the agent package."""

import argparse
import asyncio
import unicodedata
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from agent.config import AgentConfig
from agent.session import ChatSession, DEFAULT_RECURSION_LIMIT

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

_EXIT_COMMANDS = {"q", "quit", "exit"}
LineReader = Callable[[str], Awaitable[str]]


def _normalize_cli_command(value: str) -> str:
    """Normalize short CLI commands without mutating messages sent to the agent."""
    normalized = unicodedata.normalize("NFKC", value.strip())
    visible_chars = [
        char
        for char in normalized
        if unicodedata.category(char) not in {"Cc", "Cf"}
    ]
    return "".join(visible_chars).strip().casefold()


def _is_exit_input(value: str) -> bool:
    command = _normalize_cli_command(value)
    return not command or command in _EXIT_COMMANDS


def _print_progress(node_name: str, new_msgs: list) -> None:
    """Stream tool-call activity to the user without dumping payloads."""
    from langchain_core.messages import AIMessage, ToolMessage

    for msg in new_msgs:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                name = call.get("name", "?")
                print(f"  → calling {name}", flush=True)
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "?")
            content = getattr(msg, "content", "") or ""
            errored = (
                getattr(msg, "status", None) == "error"
                or (isinstance(content, str) and content.startswith("Tool error:"))
            )
            symbol = "✗" if errored else "✓"
            suffix = " errored" if errored else " returned"
            print(f"  {symbol} {name}{suffix}", flush=True)


async def _default_read_line(prompt: str) -> str:
    return input(prompt)


async def _run(
    args: argparse.Namespace,
    read_line: LineReader | None = None,
) -> None:
    config = AgentConfig()
    session = await ChatSession.create(
        config,
        recursion_limit=args.max_turns,
        load_mcp=not args.no_mcp,
        progress_cb=_print_progress,
    )
    reader = read_line or _default_read_line

    print("Agent Chat (LangGraph mode). Type 'q' to quit.\n")

    try:
        while True:
            try:
                raw_input = await reader(">> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if _is_exit_input(raw_input):
                break

            user_input = raw_input.strip()
            try:
                response = await session.turn(user_input)
            except GraphRecursionError:
                response = (
                    f"(agent hit recursion limit of {session.recursion_limit} tool "
                    "rounds without settling. Try rephrasing or narrowing the question.)"
                )
            except Exception as exc:
                response = f"(agent error: {type(exc).__name__}: {exc})"
            print(f"\n{response}\n")
    finally:
        await session.flush_recent_turns()


def main():
    parser = argparse.ArgumentParser(
        description="Conversational agent interface over the RAG core. "
        "Uses LangGraph with tool-calling to let the LLM search the knowledge base."
    )
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_RECURSION_LIMIT,
        help=f"Max recursion depth per turn (default: {DEFAULT_RECURSION_LIMIT})",
    )
    parser.add_argument(
        "--no-mcp", action="store_true",
        help="Disable MCP tool loading even if configured via environment.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
