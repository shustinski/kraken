import os
import importlib
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import PIL.Image
from PIL import Image, ImageFilter
import numpy as np
import random
from neuralimage.lib.data_interfaces import CutSettings, SampleGenerationSettings, SamplePrepareSettings
from neuralimage.lib.file_func import filter_files
from neuralimage.lib.file_retry import retry_file_read
from kraken_core.asset_source import AssetSource

from .images_parts.construction import ConstructionMixin
from .images_parts.build_parts_list import BuildPartsListMixin
from .images_parts.sized_item import SizedItemMixin
from .images_parts.apply_additional_augmentation import ApplyAdditionalAugmentationMixin
from .images_parts.build_enabled_transform_variants import (
    resolved_augmentation_variant_count,
    _augmentation_slot_count,
    _build_enabled_transform_variants,
    _apply_transform_variant,
    _load_sample_fast_cutter_getitem,
    is_sample_fast_cutter_accelerated,
    _resolve_crops_per_image,
)
from .images_parts.sample_worker import SampleWorker
from .images_parts.sample_calculator import SampleCalculator
from .images_parts.open_image import open_image
from .images_parts.image_preparator import ImagePreparator
from .images_parts.save_color import (
    save_color,
    save_binary,
)

_sample_fast_cutter_getitem, _sample_fast_cutter_accelerated = _load_sample_fast_cutter_getitem()


class SampleFastCutter(ConstructionMixin, BuildPartsListMixin, SizedItemMixin, ApplyAdditionalAugmentationMixin):
    """
    Рассчитывет число образцов в кадре и возвращет необходимый кусочк по методу __getitem__
    Соотвутствующий кусочек рассчитывается исходя из парметров
    """

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    resolved_augmentation_variant_count,
    _augmentation_slot_count,
    _build_enabled_transform_variants,
    _apply_transform_variant,
    _load_sample_fast_cutter_getitem,
    is_sample_fast_cutter_accelerated,
    _resolve_crops_per_image,
    SampleWorker,
    SampleCalculator,
    open_image,
    ImagePreparator,
    save_color,
    save_binary,
)
