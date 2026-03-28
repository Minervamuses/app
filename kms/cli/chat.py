"""Multi-turn conversational query interface for the KMS."""

import argparse
import json

from kms.config import KMSConfig
from kms.llm.openrouter import OpenRouterLLM
from kms.retriever.vector import VectorRetriever
from kms.store.chroma_store import ChromaStore

SYSTEM_PROMPT = """You are a research lab knowledge assistant. You help users find information across multiple research projects.

You have access to a knowledge base with documents tagged by layers:
- layer_1: broad category (e.g. research-notes, source-code, web-frontend, documentation, tests, data, legacy-code)
- layer_2: topic (e.g. scoring, mutation, database, progress-reports, setup)
- layer_3: specific subtopic

Available layer_1 values: {layer_1_values}

Your job:
1. When the user's question is vague, ask clarifying questions to narrow down what they want (topic? time range? code or notes?)
2. When you have enough context, respond with a SEARCH command so the system can retrieve relevant chunks
3. After receiving search results, synthesize a clear answer for the user

To trigger a search, output a JSON block like this:
```search
{{"query": "the search query", "filters": {{"layer_1": "research-notes"}}, "k": 5}}
```

Filter keys can be: layer_1, layer_2, layer_3, file_type (e.g. ".py", ".md")
Only include filters you are confident about. Omit filters to search broadly.

After search results are provided, answer the user's question based on them. If the results aren't helpful, you can search again with different terms or filters.

Do NOT make up information. Only answer based on search results."""


class ChatSession:
    """Multi-turn conversational retrieval session."""

    def __init__(self, config: KMSConfig):
        self.config = config
        self.llm = OpenRouterLLM(config=config)
        self.chroma = ChromaStore(config.raw_collection, config)
        self.retriever = VectorRetriever(self.chroma)
        self.history: list[dict[str, str]] = []

        # Load available layer values for the system prompt
        self.layer_1_values = self._get_layer_values()

    def _get_layer_values(self) -> str:
        """Get unique layer_1 values from stored tags."""
        tags_path = self.config.persist_dir + "/folder_tags.json"
        try:
            with open(tags_path, "r") as f:
                folder_tags = json.load(f)
            layer_1s = set()
            for tags in folder_tags.values():
                if tags:
                    layer_1s.add(tags[0])
            return ", ".join(sorted(layer_1s))
        except FileNotFoundError:
            return "(unknown — no folder_tags.json found)"

    def _build_messages(self, user_input: str) -> list[dict]:
        """Build message list for LLM call."""
        system = SYSTEM_PROMPT.format(layer_1_values=self.layer_1_values)
        messages = [{"role": "system", "content": system}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _search(self, search_cmd: dict) -> str:
        """Execute a search and return formatted results."""
        query = search_cmd.get("query", "")
        filters = search_cmd.get("filters", {})
        k = search_cmd.get("k", 5)

        # Use pid_filter if layer filters are specified
        # For now, retrieve broadly and filter by metadata
        docs = self.retriever.retrieve(query, k=k * 3)

        # Apply metadata filters
        filtered = []
        for doc in docs:
            match = True
            for key, val in filters.items():
                if doc.metadata.get(key) != val:
                    match = False
                    break
            if match:
                filtered.append(doc)

        results = filtered[:k]

        if not results:
            return "No results found for this search."

        parts = []
        for i, doc in enumerate(results, 1):
            pid = doc.metadata.get("pid", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            tags = doc.metadata.get("tags", [])
            preview = doc.page_content[:500]
            parts.append(f"[Result {i}] {pid} (chunk {chunk_id}, tags: {tags})\n{preview}")

        return "\n\n".join(parts)

    def _parse_search_command(self, text: str) -> dict | None:
        """Extract search command from LLM response."""
        marker = "```search"
        if marker not in text:
            return None

        try:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            return None

    def turn(self, user_input: str) -> str:
        """Process one conversation turn. Returns the final response to show the user."""
        self.history.append({"role": "user", "content": user_input})

        # Call LLM
        messages = self._build_messages("")
        # Remove the duplicate — last history entry is the user message
        messages = messages[:-1]

        response = self.llm.invoke(
            prompt=self._format_messages(messages),
            max_tokens=1024,
            temperature=0.3,
        )

        # Check if LLM wants to search
        search_cmd = self._parse_search_command(response)

        if search_cmd:
            # Execute search
            search_results = self._search(search_cmd)

            # Show user what we're searching for
            filters_str = json.dumps(search_cmd.get("filters", {}))
            print(f"  [searching: \"{search_cmd['query']}\" filters={filters_str}]")

            # Feed results back to LLM for synthesis
            self.history.append({"role": "assistant", "content": response})
            self.history.append({"role": "user", "content": f"Search results:\n\n{search_results}"})

            messages = self._format_messages(
                [{"role": "system", "content": SYSTEM_PROMPT.format(layer_1_values=self.layer_1_values)}]
                + self.history
            )

            final_response = self.llm.invoke(
                prompt=messages,
                max_tokens=1024,
                temperature=0.3,
            )

            self.history.append({"role": "assistant", "content": final_response})
            return final_response
        else:
            # LLM is asking a clarifying question or responding directly
            self.history.append({"role": "assistant", "content": response})
            return response

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages into a single prompt string for the LLM."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
        return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Conversational query interface for the KMS.")
    parser.add_argument("-k", type=int, default=5, help="Max results per search (default: 5)")
    args = parser.parse_args()

    config = KMSConfig()
    session = ChatSession(config)

    print("KMS Chat. Type 'q' to quit.\n")

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
