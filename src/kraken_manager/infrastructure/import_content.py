"""Filesystem implementation of the application import-content port."""

from pathlib import Path

from kraken_core.media_types import artifact_media_type
from kraken_core.safe_files import open_regular_read
from kraken_manager.application.imports import ImportContentMetadata, ImportSource


class FilesystemImportContentReader:
    def describe(self, source: ImportSource) -> ImportContentMetadata:
        path = Path(source.source_key)
        return ImportContentMetadata(path.name, artifact_media_type(path.name))

    def iter_bytes(self, source: ImportSource):
        path = Path(source.source_key)
        with open_regular_read(path, root=path.parent) as stream:
            while chunk := stream.read(1024 * 1024):
                yield chunk
