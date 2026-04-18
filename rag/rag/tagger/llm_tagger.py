"""LLM-based folder tagger — uses an LLM to assign tags and generate summaries."""

import json

from rag.config import KMSConfig
from rag.llm.openrouter import OpenRouterLLM
from rag.tagger.base import BaseTagger, FolderMeta

PROMPT_TEMPLATE = """You are a research project classifier. Given a folder path and its contents, return a JSON object with:
1. "tags": 2-4 hierarchical tags from broad to specific (lowercase kebab-case)
2. "summary": a 1-2 sentence natural language summary of what this folder contains

Rules for tags:
- First tag: broad category (e.g. "source-code", "research-notes", "documentation", "config", "legacy-code", "web-frontend", "web-backend", "data", "tests")
- Following tags: increasingly specific topic (e.g. "scoring", "mutation", "etl", "deployment", "debugging")

Rules for summary:
- Describe the PURPOSE and CONTENT of the folder
- Mention key topics, algorithms, or systems discussed
- Be specific enough that someone could decide if this folder is relevant to their question

Return ONLY a JSON object, no explanation.

Folder: {folder_path}
Files: {file_list}
Previews:
{previews}

JSON:"""


class LLMTagger(BaseTagger):
    """Assign tags and generate summaries for folders using an LLM."""

    def __init__(self, config: KMSConfig | None = None):
        self.config = config or KMSConfig()
        self.llm = OpenRouterLLM(config=self.config)

    def tag(self, folder_path: str, file_names: list[str], file_previews: dict[str, str]) -> FolderMeta:
        """Assign tags and summary to a folder via LLM."""
        file_list = ", ".join(file_names[:20])
        previews = "\n".join(
            f"  {name}: {preview}"
            for name, preview in list(file_previews.items())[:10]
        )

        prompt = PROMPT_TEMPLATE.format(
            folder_path=folder_path,
            file_list=file_list,
            previews=previews,
        )

        response = self.llm.invoke(prompt, max_tokens=200, temperature=0.0)

        try:
            # Extract JSON object from response
            start = response.index("{")
            end = response.rindex("}") + 1
            data = json.loads(response[start:end])
            tags = data.get("tags", [])
            summary = data.get("summary", "")
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                return FolderMeta(tags=tags, summary=summary)
        except (ValueError, json.JSONDecodeError):
            pass

        # Fallback
        return FolderMeta(
            tags=[folder_path.split("/")[0].lower().replace(" ", "-")],
            summary=f"Contents of {folder_path}",
        )
