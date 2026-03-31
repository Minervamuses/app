"""Abstract base class for taggers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FolderMeta:
    """Tags and summary for a folder."""

    tags: list[str]
    summary: str


class BaseTagger(ABC):
    """Abstract base class for folder/document taggers."""

    @abstractmethod
    def tag(self, folder_path: str, file_names: list[str], file_previews: dict[str, str]) -> FolderMeta:
        """Assign tags and generate a summary for a folder.

        Args:
            folder_path: Relative path of the folder from repo root.
            file_names: List of file names in the folder.
            file_previews: Dict of filename -> first line of file.

        Returns:
            FolderMeta with tags and a natural language summary.
        """
