"""Multi-turn conversational query interface for the KMS.

Uses tool-calling: the LLM decides when and how to search the knowledge
base, can search multiple times per turn, and synthesizes answers from
retrieved results.

Usage:
    python -m kms.cli.chat
    python -m kms.cli.chat --max-turns 10
    python -m kms.cli.chat -h
"""

import argparse
import json

from kms.config import KMSConfig
from kms.llm.openrouter import OpenRouterLLM
from kms.tool.search import SearchTool

SYSTEM_PROMPT = """You are a research lab knowledge assistant. You help users find information across multiple research projects stored in a knowledge base.

When answering questions:
1. If the user's question is vague, ask clarifying questions first — do NOT search blindly.
2. When you have enough context, use the search tool to find relevant information.
3. You can search multiple times with different queries or filters to gather comprehensive results.
4. After searching, synthesize a clear answer based on the results. Cite file paths when relevant.
5. If results aren't helpful, try different search terms or broader/narrower filters.

Do NOT make up information. Only answer based on search results or your conversation with the user."""

# Max tool calls per turn to prevent runaway loops
DEFAULT_MAX_TOOL_ROUNDS = 5


class ChatSession:
    """Multi-turn conversational retrieval session with tool calling."""

    def __init__(self, config: KMSConfig, max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS):
        self.config = config
        self.llm = OpenRouterLLM(config=config)
        self.search_tool = SearchTool(config)
        self.tools = [self.search_tool.schema()]
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_tool_rounds = max_tool_rounds

    def turn(self, user_input: str) -> str:
        """Process one conversation turn. Returns the final text response."""
        self.messages.append({"role": "user", "content": user_input})

        for _round in range(self.max_tool_rounds):
            response = self.llm.chat(
                messages=self.messages,
                tools=self.tools,
                max_tokens=1024,
                temperature=0.3,
            )

            if not response.has_tool_calls:
                # LLM is done — either a clarifying question or final answer
                text = response.content or ""
                self.messages.append({"role": "assistant", "content": text})
                return text

            # Build assistant message with tool calls for the message history
            assistant_msg: dict = {"role": "assistant", "content": response.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
            self.messages.append(assistant_msg)

            # Execute each tool call and add results
            for tc in response.tool_calls:
                print(f"  [search: \"{tc.arguments.get('query', '')}\"", end="")
                filters = {k: v for k, v in tc.arguments.items() if k not in ("query", "k")}
                if filters:
                    print(f" filters={json.dumps(filters)}", end="")
                print("]")

                result = self.search_tool.execute(tc.arguments)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Safety fallback — shouldn't normally reach here
        return "(Reached maximum search rounds. Please try a more specific question.)"


def main():
    parser = argparse.ArgumentParser(
        description="Conversational query interface for the KMS. "
        "Uses tool-calling to let the LLM search the knowledge base."
    )
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TOOL_ROUNDS,
        help=f"Max search rounds per turn (default: {DEFAULT_MAX_TOOL_ROUNDS})",
    )
    args = parser.parse_args()

    config = KMSConfig()
    session = ChatSession(config, max_tool_rounds=args.max_turns)

    print("KMS Chat (tool-calling mode). Type 'q' to quit.\n")

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
