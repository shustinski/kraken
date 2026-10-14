"""images: sample worker."""
from __future__ import annotations
import os
from pathlib import Path
from neuralimage.lib.data_interfaces import CutSettings
from neuralimage.lib.file_func import filter_files
from kraken_core.asset_source import AssetSource

from neuralimage.lib.images_parts.build_enabled_transform_variants import (
    _augmentation_slot_count,
    _build_enabled_transform_variants,
    _resolve_crops_per_image,
)

from neuralimage.lib.images_parts.open_image import open_image

class SampleWorker:
    VALID_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, path=None, paramns: CutSettings | None = None):
        self._path: Path | None = Path(path) if path is not None else None
        self._img_paths: list[Path] | None = None
        self._img_sizes: list[tuple[int, int]] | None = None
        self._params: CutSettings | None = paramns
        self._total_parts: int = 0

    def set_path(self, path: Path | None):
        normalized_path = Path(path) if path is not None else None
        if self._path == normalized_path:
            return
        self._path = normalized_path
        self._img_paths = None
        self._img_sizes = None
        self._total_parts = 0

    def set_settings(self, setting: CutSettings):
        if self._params == setting:
            return
        self._params = setting
        self._total_parts = 0

    def _calculate_len(self):
        if self._path is None:
            self._total_parts = 0
            return
        if not os.path.isdir(self._path):
            self._total_parts = 0
            return
        if self._params is None:
            self._total_parts = 0
            return
        if self._img_sizes is None:
            self._calculate_samples_from_path()
        else:
            self.calculate_samples_amount()

    def __len__(self):
        self._calculate_len()
        return self._total_parts

    def _calculate_samples_from_path(self):
        """

        Args:
            path: parh for image calculation

        """

        self._img_paths = self.collect_image_paths(
            self._path,
            recursive=bool(getattr(self._params, 'recursive_file_search', False)),
        )
        if len(self._img_paths) == 0:
            self._img_sizes = []
            self._total_parts = 0
            return

        self._img_sizes = self.collect_image_sizes(self._img_paths)
        self.calculate_samples_amount()

    @classmethod
    def collect_image_paths(cls, path: Path, *, recursive: bool = False) -> list[Path]:
        if not path.is_dir():
            return []
        return filter_files(path, cls.VALID_EXTENSIONS, recursive=recursive)

    @staticmethod
    def collect_image_sizes(image_paths: list[Path], *, asset_source: AssetSource | None = None) -> list[tuple[int, int]]:
        image_sizes: list[tuple[int, int]] = []
        for image in image_paths:
            with open_image(image, asset_source) as img:
                width, height = img.size
            image_sizes.append((height, width))
        return image_sizes

    def _get_image_sizes(self):
        self._img_sizes = self.collect_image_sizes(self._img_paths or [])

    def calculate_samples_amount(self):
        if self._params is None or self._path is None:
            self._total_parts = 0
            return 0

        self._total_parts = self.calculate_total_samples(self._img_sizes or [], self._params)
        return self._total_parts

    @classmethod
    def calculate_total_samples(
        cls,
        image_sizes: list[tuple[int, int]],
        params: CutSettings | None,
    ) -> int:
        if params is None:
            return 0

        total_parts = 0
        for image_size in image_sizes:
            total_parts += cls.calculate_image_parts_for_settings(image_size, params)
        return total_parts

    def calculate_image_parts(self, image_size: tuple[int, int]) -> int:
        return self.calculate_image_parts_for_settings(image_size, self._params)

    @staticmethod
    def calculate_image_parts_for_settings(
        image_size: tuple[int, int],
        params: CutSettings | None,
    ) -> int:
        """

        Args:
            image_size: (height,width)

        Returns:
            number of parts
        """

        im_height, im_width = image_size
        if params is None:
            return 0

        if getattr(params, 'random_crop', False):
            frames_in_frame = _resolve_crops_per_image(params)
        else:
            step = params.step
            sample_x_size = params.x_size
            sample_y_size = params.y_size

            width_steps = int((im_width - sample_x_size) / step) + 1
            height_steps = int((im_height - sample_y_size) / step) + 1
            frames_in_frame = width_steps * height_steps

        transform_variants = _build_enabled_transform_variants(
            enable_rotate_180=bool(getattr(params, 'vertical_rotation', False)),
            enable_rotate_90=bool(getattr(params, 'horizontal_rotation', False)),
            enable_flip_x=bool(getattr(params, 'flip_x', False)),
            enable_flip_y=bool(getattr(params, 'flip_y', False)),
            square_patch=int(params.x_size) == int(params.y_size),
        )
        frames_in_frame *= len(transform_variants)
        if getattr(params, 'scale_augmentation', False):
            frames_in_frame *= 2
        augmentation_slots = _augmentation_slot_count(params)
        if augmentation_slots > 1:
            frames_in_frame *= augmentation_slots

        return frames_in_frame
