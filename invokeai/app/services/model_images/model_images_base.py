from abc import ABC, abstractmethod
from pathlib import Path

from PIL.Image import Image as PILImageType


class ModelImageFileStorageBase(ABC):
    """Low-level service responsible for storing and retrieving image files."""

    @abstractmethod
    def get(self, model_key: str) -> PILImageType:
        """Retrieves a model image as PIL Image."""
        pass

    @abstractmethod
    def get_path(self, model_key: str) -> Path:
        """Gets the internal path to a model image."""
        pass

    @abstractmethod
    def get_preview_path(self, model_key: str) -> Path | None:
        """Gets the full-size cover path, if one is stored."""
        pass

    @abstractmethod
    def get_url(self, model_key: str) -> str | None:
        """Gets the URL to fetch a model image."""
        pass

    @abstractmethod
    def save(self, image: PILImageType, model_key: str, *, original_bytes: bytes | None = None) -> None:
        """Saves a thumbnail and, for an upload, a full-size cover."""
        pass

    @abstractmethod
    def move(self, model_key: str, new_key: str) -> None:
        """Transfers a thumbnail and its full-size cover to a new key."""
        pass

    @abstractmethod
    def delete(self, model_key: str) -> None:
        """Deletes a model image."""
        pass
