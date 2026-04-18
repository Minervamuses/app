"""CLI entry point for the agent package."""

import argparse

from rag.config import KMSConfig
from agent.session import ChatSession, DEFAULT_RECURSION_LIMIT


def main():
    parser = argparse.ArgumentParser(
        description="Conversational agent interface over the RAG core. "
        "Uses LangGraph with tool-calling to let the LLM search the knowledge base."
    )
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_RECURSION_LIMIT,
        help=f"Max recursion depth per turn (default: {DEFAULT_RECURSION_LIMIT})",
    )
    args = parser.parse_args()

    config = KMSConfig()
    session = ChatSession(config, recursion_limit=args.max_turns)

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


if __name__ == "__main__":
    main()
