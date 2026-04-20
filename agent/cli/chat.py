"""CLI entry point for the agent package."""

import argparse
import asyncio

from agent.config import AgentConfig
from agent.session import ChatSession, DEFAULT_RECURSION_LIMIT


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

        response = session.turn(user_input)
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
