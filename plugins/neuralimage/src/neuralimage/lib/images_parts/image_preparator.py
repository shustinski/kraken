"""images: image preparator."""
from __future__ import annotations
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from PIL import Image
from neuralimage.lib.data_interfaces import SamplePrepareSettings
from kraken_core.asset_source import AssetSource

from neuralimage.lib.images_parts.open_image import open_image

class ImagePreparator:

    def __init__(self, path:Path, parameters:SamplePrepareSettings, *, asset_source: AssetSource | None = None):
        self._params = parameters
        self._path = path
        self._asset_source = asset_source

    @contextmanager
    def _open_image(self) -> Iterator[Image.Image]:
        with open_image(self._path, self._asset_source) as image:
            yield image

    @property
    def image(self):
        self._prepare()
        return self._image

    @property
    def size(self):
        return self._lazy_size()

    def _lazy_size(self):
        if (
            self._params.enable_resize
            and self._params.target_size is not None
            and int(getattr(self._params, 'compression_factor', 1)) <= 1
        ):
            return self._params.target_size
        with self._open_image() as img:
            size = img.size
        if not self._params.enable_crop or self._params.edge_cut is None:
            return size
        return self._size_after_crop(size)

    def _size_after_crop(self, image_size: tuple[int, int]):
        crop_x, crop_y = int(self._params.edge_cut[0]), int(self._params.edge_cut[1])
        width = max(0, int(image_size[0]) - (crop_x * 2))
        height = max(0, int(image_size[1]) - (crop_y * 2))
        return width, height

    def _prepare(self):
        with self._open_image() as img:
            img.load()
            self._image = img.copy()
        self._size = img.size
        if self._params.enable_crop and self._params.edge_cut is not None:
           self._crop()
        if self._params.enable_resize:
            self._resize()

    def _crop(self):
        p = self._params
        top_left_x = p.edge_cut[0]
        top_left_y = p.edge_cut[1]
        bottom_right_x =  self._size[0] - p.edge_cut[0]
        bottom_right_y =  self._size[1] - p.edge_cut[1]
        self._image = self._image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    def _resize(self):
        p = self._params
        compression_factor = max(1, int(getattr(p, 'compression_factor', 1)))
        if compression_factor > 1:
            original_size = self._image.size
            compressed_size = (
                max(1, int(round(original_size[0] / compression_factor))),
                max(1, int(round(original_size[1] / compression_factor))),
            )
            self._image = self._image.resize(compressed_size, resample=Image.Resampling.BILINEAR)
            self._image = self._image.resize(original_size, resample=Image.Resampling.BILINEAR)
            return
        if p.target_size is not None:
            self._image = self._image.resize(p.target_size)
