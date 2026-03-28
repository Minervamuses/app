"""LLM-based folder tagger — uses an LLM to assign hierarchical tags."""

import json

from kms.config import KMSConfig
from kms.llm.openrouter import OpenRouterLLM
from kms.tagger.base import BaseTagger

PROMPT_TEMPLATE = """You are a research project classifier. Given a folder path and its contents, assign 2-4 hierarchical tags from broad to specific.

Rules:
- First tag: broad category (e.g. "source-code", "research-notes", "documentation", "config", "legacy-code", "web-frontend", "web-backend", "data", "tests")
- Following tags: increasingly specific topic (e.g. "scoring", "mutation", "etl", "deployment", "debugging")
- Use lowercase kebab-case
- Return ONLY a JSON array of strings, no explanation

Folder: {folder_path}
Files: {file_list}
Previews:
{previews}

Tags:"""


class LLMTagger(BaseTagger):
    """Assign hierarchical tags to folders using an LLM."""

    def __init__(self, config: KMSConfig | None = None):
        self.config = config or KMSConfig()
        self.llm = OpenRouterLLM(config=self.config)

    def tag(self, folder_path: str, file_names: list[str], file_previews: dict[str, str]) -> list[str]:
        """Assign hierarchical tags to a folder via LLM."""
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

        response = self.llm.invoke(prompt, max_tokens=100, temperature=0.0)

        try:
            # Extract JSON array from response
            start = response.index("[")
            end = response.index("]") + 1
            tags = json.loads(response[start:end])
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                return tags
        except (ValueError, json.JSONDecodeError):
            pass

        # Fallback: return folder path as single tag
        return [folder_path.split("/")[0].lower().replace(" ", "-")]
