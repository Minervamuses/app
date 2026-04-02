"""Multi-turn conversational query interface for the KMS.

Uses LangGraph to orchestrate tool-calling: the LLM decides when and how
to search the knowledge base, can search multiple times per turn, and
synthesizes answers from retrieved results.

Usage:
    python -m kms.cli.chat
    python -m kms.cli.chat --max-turns 10
    python -m kms.cli.chat -h
"""

import argparse

from langchain_core.messages import HumanMessage, SystemMessage

from kms.agent.graph import build_graph
from kms.config import KMSConfig

SYSTEM_PROMPT = """You are a knowledge base assistant. You help users find information stored in an indexed repository.

You have three tools:

1. **explore** — Discover what's in the knowledge base: categories, tags, date ranges, folder summaries.
   Use this first when you're unsure what the knowledge base contains.

2. **search** — Semantic search with optional filters (category, file_type, date range).
   Use specific queries. You can search multiple times with different queries or filters.

3. **get_context** — Expand a search result by retrieving surrounding chunks from the same file.
   Use when a result looks relevant but you need more context.

Workflow:
- If the question is vague or you don't know the structure of the knowledge base, start with explore.
- Use search with appropriate filters based on what you learned from explore.
- Use get_context if you need to see more around a promising result.
- After 1-3 searches, synthesize your answer. Don't keep searching for perfection.
- Do NOT make up information. Only answer based on tool results or your conversation with the user."""

DEFAULT_RECURSION_LIMIT = 16


class ChatSession:
    """Multi-turn conversational retrieval session backed by LangGraph."""

    def __init__(self, config: KMSConfig, recursion_limit: int = DEFAULT_RECURSION_LIMIT):
        self.graph = build_graph(config)
        self.recursion_limit = recursion_limit
        self.run_config = {
            "configurable": {"thread_id": "default"},
            "recursion_limit": recursion_limit,
        }
        # Seed conversation with system prompt
        self.graph.invoke(
            {"messages": [SystemMessage(content=SYSTEM_PROMPT)]},
            config=self.run_config,
        )

    def turn(self, user_input: str) -> str:
        """Process one conversation turn. Returns the final text response."""
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.run_config,
        )
        return result["messages"][-1].content or ""


def main():
    parser = argparse.ArgumentParser(
        description="Conversational query interface for the KMS. "
        "Uses LangGraph with tool-calling to let the LLM search the knowledge base."
    )
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_RECURSION_LIMIT,
        help=f"Max recursion depth per turn (default: {DEFAULT_RECURSION_LIMIT})",
    )
    args = parser.parse_args()

    config = KMSConfig()
    session = ChatSession(config, recursion_limit=args.max_turns)

    print("KMS Chat (LangGraph mode). Type 'q' to quit.\n")

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
