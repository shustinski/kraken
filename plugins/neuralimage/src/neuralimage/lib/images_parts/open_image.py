"""images: open image."""
from __future__ import annotations
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from PIL import Image
from neuralimage.lib.file_retry import retry_file_read
from kraken_core.asset_source import AssetSource

@contextmanager
def open_image(path: Path | str, asset_source: AssetSource | None = None) -> Iterator[Image.Image]:
    if asset_source is None:
        with retry_file_read(lambda: Image.open(path), path=path) as image:
            yield image
    else:
        with asset_source.open(str(path)) as stream, Image.open(stream) as image:
            yield image
