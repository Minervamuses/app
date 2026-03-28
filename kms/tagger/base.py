"""Abstract base class for taggers."""

from abc import ABC, abstractmethod


class BaseTagger(ABC):
    """Abstract base class for folder/document taggers."""

    @abstractmethod
    def tag(self, folder_path: str, file_names: list[str], file_previews: dict[str, str]) -> list[str]:
        """Assign hierarchical tags to a folder.

        Args:
            folder_path: Relative path of the folder from repo root.
            file_names: List of file names in the folder.
            file_previews: Dict of filename -> first line of file.

        Returns:
            List of tags from broad to specific, e.g. ["research-notes", "scoring", "debugging"].
        """
