"""images: sample calculator."""
from __future__ import annotations
from PIL import Image
from neuralimage.lib.data_interfaces import SampleGenerationSettings
from neuralimage.lib.file_retry import retry_file_read

from neuralimage.lib.images_parts.build_enabled_transform_variants import (
    _augmentation_slot_count,
    _build_enabled_transform_variants,
    _resolve_crops_per_image,
)

class SampleCalculator:

    def __init__(self, image_size:tuple[int,int], params:SampleGenerationSettings):
        self._image_size = image_size
        self._params = params
        self._width_steps = 0
        self._height_steps = 0

    @classmethod
    def from_path(cls, path:str, params:SampleGenerationSettings):
        with retry_file_read(lambda: Image.open(path), path=path) as image:
            width, height = image.size
        return cls((height, width), params)

    def __len__(self):
        return self._calculate_parts()

    @property
    def size(self):
        return self._width_steps, self._height_steps

    def _calculate_parts(self):
        if getattr(self._params, 'random_crop', False):
            self._width_steps = _resolve_crops_per_image(self._params)
            self._height_steps = 1
            frames_in_frame = self._width_steps
        else:
            im_height, im_width = self._image_size

            step = self._params.step
            sample_x_size, sample_y_size = self._params.segment_size

            self._width_steps = int((im_width - sample_x_size) / step) + 1
            self._height_steps = int((im_height - sample_y_size) / step) + 1
            frames_in_frame = self._width_steps * self._height_steps

        transform_variants = _build_enabled_transform_variants(
            enable_rotate_180=bool(getattr(self._params, 'vertical_rotation', False)),
            enable_rotate_90=bool(getattr(self._params, 'horizontal_rotation', False)),
            enable_flip_x=bool(getattr(self._params, 'flip_x', False)),
            enable_flip_y=bool(getattr(self._params, 'flip_y', False)),
            square_patch=tuple(getattr(self._params, 'segment_size', (0, 0)))[0]
            == tuple(getattr(self._params, 'segment_size', (0, 0)))[1],
        )
        frames_in_frame *= len(transform_variants)
        if self._params.scale_augmentation:
            frames_in_frame *= 2
        augmentation_slots = _augmentation_slot_count(self._params)
        if augmentation_slots > 1:
            frames_in_frame *= augmentation_slots

        return frames_in_frame
