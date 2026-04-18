"""Central configuration for the RAG workspace."""

from dataclasses import dataclass
from pathlib import Path

# Resolve store path relative to the workspace root (parent of rag/)
_WORKSPACE_DIR = Path(__file__).resolve().parents[1]

# Single collection for all knowledge chunks
KNOWLEDGE_COLLECTION = "knowledge"


@dataclass
class KMSConfig:
    """Central configuration for the RAG workspace."""

    # Storage
    persist_dir: str = str(_WORKSPACE_DIR / "store")

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
