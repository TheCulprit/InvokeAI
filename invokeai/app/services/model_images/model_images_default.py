import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_images.model_images_base import ModelImageFileStorageBase
from invokeai.app.services.model_images.model_images_common import (
    ModelImageFileDeleteException,
    ModelImageFileNotFoundException,
    ModelImageFileSaveException,
)
from invokeai.app.util.misc import uuid_string
from invokeai.app.util.path_safety import is_contained_filename
from invokeai.app.util.thumbnails import make_thumbnail

_PREVIEW_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "AVIF": ".avif", "BMP": ".bmp"}


class ModelImageFileStorageDisk(ModelImageFileStorageBase):
    """Stores images on disk"""

    def __init__(self, model_images_folder: Path):
        self._model_images_folder = model_images_folder
        self._previews_folder = model_images_folder / "previews"
        self._validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, model_key: str) -> PILImageType:
        try:
            path = self.get_path(model_key)

            if not self._validate_path(path):
                raise ModelImageFileNotFoundException

            return Image.open(path)
        except FileNotFoundError as e:
            raise ModelImageFileNotFoundException from e

    def save(self, image: PILImageType, model_key: str, *, original_bytes: bytes | None = None) -> None:
        try:
            self._validate_storage_folders()
            image_path = self.get_path(model_key)
            with TemporaryDirectory(dir=self._model_images_folder) as tmp:
                thumbnail = Path(tmp) / "thumbnail.webp"
                make_thumbnail(image, 256).save(thumbnail, format="webp")

                preview_path = None
                if original_bytes is not None:
                    extension = _PREVIEW_EXTENSIONS.get(image.format or "")
                    preview = Path(tmp) / "preview"
                    if extension is None:
                        # For formats the browser cannot display, keep the dimensions in a lossless PNG.
                        image.convert("RGBA").save(preview, format="png")
                    else:
                        preview.write_bytes(original_bytes)
                    self._previews_folder.mkdir(exist_ok=True)
                    preview_path = self._previews_folder / (model_key + (extension or ".png"))
                    preview.replace(preview_path)

                thumbnail.replace(image_path)
                self._delete_previews(model_key, keep=preview_path)
        except Exception as e:
            raise ModelImageFileSaveException from e

    def _preview_paths(self, model_key: str) -> list[Path]:
        self._validate_key(model_key)
        return [self._previews_folder / (model_key + extension) for extension in _PREVIEW_EXTENSIONS.values()]

    def get_preview_path(self, model_key: str) -> Path | None:
        return next((path for path in self._preview_paths(model_key) if path.is_file()), None)

    def _delete_previews(self, model_key: str, *, keep: Path | None = None) -> None:
        for path in self._preview_paths(model_key):
            if path != keep:
                path.unlink(missing_ok=True)

    def move(self, model_key: str, new_key: str) -> None:
        image_path = self.get_path(model_key)
        new_image_path = self.get_path(new_key)
        if not image_path.is_file():
            raise ModelImageFileNotFoundException
        if image_path == new_image_path:
            return

        preview_path = self.get_preview_path(model_key)
        new_preview_path = None
        with TemporaryDirectory(dir=self._model_images_folder) as tmp:
            thumbnail = Path(tmp) / "thumbnail.webp"
            shutil.copyfile(image_path, thumbnail)
            if preview_path is not None:
                preview = Path(tmp) / "preview"
                shutil.copyfile(preview_path, preview)
                new_preview_path = self._previews_folder / (new_key + preview_path.suffix)
                preview.replace(new_preview_path)
            thumbnail.replace(new_image_path)
        self._delete_previews(new_key, keep=new_preview_path)
        self.delete(model_key)

    def get_path(self, model_key: str) -> Path:
        self._validate_key(model_key)
        path = self._model_images_folder / (model_key + ".webp")

        return path

    def get_url(self, model_key: str) -> str | None:
        try:
            path = self.get_path(model_key)
        except ModelImageFileNotFoundException:
            # This is called while listing every model, so a bad key must not take the whole list down.
            return None
        try:
            if not self._validate_path(path):
                return
        except OSError:
            # e.g. ENAMETOOLONG for an over-long stored key: as above, one bad key must not fail the whole list.
            return None

        url = self._invoker.services.urls.get_model_image_url(model_key)

        # The image URL never changes, so we must add random query string to it to prevent caching
        url += f"?{uuid_string()}"

        return url

    def delete(self, model_key: str) -> None:
        try:
            path = self.get_path(model_key)

            if not self._validate_path(path):
                raise ModelImageFileNotFoundException

            self._delete_previews(model_key)
            path.unlink()

        except Exception as e:
            raise ModelImageFileDeleteException from e

    def _validate_path(self, path: Path) -> bool:
        """Validates the path given for an image."""
        return path.exists()

    @staticmethod
    def _validate_key(model_key: str) -> None:
        """Validates that a model key cannot escape the model images folder.

        Keys are server-generated - a uuid, or a slug for external API models - but they arrive from the client as
        a path parameter, so they are untrusted. This checks containment on the running platform rather than the
        portable rules new keys must meet: a key stored by an older version may be a legal posix filename that
        Windows would reject (`legacy:key`), and its existing cover must stay servable and manageable.

        :raises ModelImageFileNotFoundException: if the key would not name a file inside the folder
        """
        if not is_contained_filename(model_key):
            raise ModelImageFileNotFoundException(f"Invalid model key {model_key!r}")

    def _validate_storage_folders(self) -> None:
        """Checks if the required folders exist and create them if they don't"""
        self._model_images_folder.mkdir(parents=True, exist_ok=True)
