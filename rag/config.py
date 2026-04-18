"""Central configuration for the RAG workspace."""

import os
from dataclasses import dataclass, field

# Single collection for all knowledge chunks
KNOWLEDGE_COLLECTION = "knowledge"


def _default_persist_dir() -> str:
    """Resolve the store directory without peeking at rag's own install path.

    Host projects embed rag as a tool and decide where its data lives. Order:
      1. ``KMS_STORE_DIR`` env var (explicit host override).
      2. ``./store`` relative to the caller's CWD (the host project root when
         the host invokes rag from there).
    """
    env = os.environ.get("KMS_STORE_DIR")
    if env:
        return env
    return os.path.join(os.getcwd(), "store")


@dataclass
class KMSConfig:
    """Central configuration for the RAG workspace."""

    # Storage
    persist_dir: str = field(default_factory=_default_persist_dir)

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 100
    encoding_model: str = "o200k_base"

    # Embedding
    embed_model: str = "bge-m3"

    # LLM
    llm_model: str = "z-ai/glm-5"

    # Evaluation LLMs
    gen_llm_model: str = "google/gemini-3.1-pro-preview"
    judge_llm_model: str = "openai/gpt-5.2"
    filter_llm_model: str = "llama3.1:8b"

    # Retrieval
    default_k: int = 10

    # Agent context controls
    agent_max_messages: int = 20
    agent_max_tool_interactions: int = 4

    # Collection names
    raw_collection: str = "raw"

    def raw_json_path(self) -> str:
        """Path to the raw chunks JSON file."""
        return f"{self.persist_dir}/raw.json"

    def folder_meta_path(self) -> str:
        """Path to the folder metadata JSON file."""
        return f"{self.persist_dir}/folder_meta.json"
