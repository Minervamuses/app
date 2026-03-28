"""Central configuration for the KMS pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

# Resolve store path relative to app/ directory (parent of kms/)
_APP_DIR = Path(__file__).resolve().parents[1]


@dataclass
class KMSConfig:
    """Central configuration for the KMS pipeline."""

    # Storage
    persist_dir: str = str(_APP_DIR / "store")

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 100
    encoding_model: str = "o200k_base"

    # Embedding
    embed_model: str = "bge-m3"

    # LLM
    llm_model: str = "google/gemini-3-flash-preview"

    # Retrieval
    default_k: int = 10

    # Collection names
    raw_collection: str = "raw"

    def raw_json_path(self) -> str:
        """Path to the raw chunks JSON file."""
        return f"{self.persist_dir}/raw.json"
