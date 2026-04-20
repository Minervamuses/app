"""CLI entry point for the agent package."""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from agent.config import AgentConfig
from agent.session import ChatSession, DEFAULT_RECURSION_LIMIT

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)


async def _run(args: argparse.Namespace) -> None:
    config = AgentConfig()
    session = await ChatSession.create(
        config,
        recursion_limit=args.max_turns,
        load_mcp=not args.no_mcp,
    )

    print("Agent Chat (LangGraph mode). Type 'q' to quit.\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            break

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
