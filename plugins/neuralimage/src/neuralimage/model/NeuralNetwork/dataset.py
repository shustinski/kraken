import os
import random
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from bisect import bisect_right
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from kraken_core.asset_source import AssetSource
from torch.utils.data import Dataset, get_worker_info
from neuralimage.augmentations import (
    ICDefectAugmentor,
    PCBDefectAugmentor,
    SyntheticTopologyGenerator,
    SyntheticTopologyParameters,
    TechVariationAugmentor,
)
from neuralimage.lib.data_interfaces import TrainingParameters, SampleGenerationSettings, SamplePrepareSettings
from neuralimage.lib.data_interfaces import (
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
)
from neuralimage.lib.images import ImagePreparator, SampleCalculator, SampleFastCutter
from neuralimage.lib.rare_patch_masks import resolve_rare_patch_mask_path
from neuralimage.model.NeuralNetwork.context_utils import (
    PatchWindow,
    build_global_context_image,
    extract_centered_crop,
    normalize_patch_coords,
    normalize_size_pair,
)
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.targets.dataset_hooks import (
    apply_dataset_preprocessing,
    apply_dataset_sem_augmentation,
    maybe_build_supervision_target,
)

from .dataset_parts.construction import ConstructionMixin
from .dataset_parts.epoch import EpochMixin
from .dataset_parts.getitem import GetitemMixin
from .dataset_parts.build_frame_cutter import BuildFrameCutterMixin
from .dataset_parts.build_global_context_image import BuildGlobalContextImageMixin
from .dataset_parts.construction_2 import Construction2Mixin
from .dataset_parts.epoch_2 import Epoch2Mixin
from .dataset_parts.getitem_2 import Getitem2Mixin
from .dataset_parts.build_frame_cutter_2 import BuildFrameCutter2Mixin
from .dataset_parts.sample_topology_parameters import SampleTopologyParametersMixin
from .dataset_parts.extract_binary_single_channel import (
    PatchSampleRequest,
    _unwrap_tech_augmented_mask,
    _extract_binary_single_channel,
    _broadcast_augmented_mask,
    _is_binary_like_plane,
)
from .dataset_parts.apply_binary_tech_augmentation import (
    _apply_binary_tech_augmentation,
    _apply_binary_tech_augmentation_to_pair,
    _apply_binary_tech_augmentation_with_seed,
    _apply_binary_tech_augmentation_to_pair_with_seed,
)
from .dataset_parts.index_in_list import (
    summarise_list,
    index_in_list,
)

class NoCutDataset(
    ConstructionMixin,
    EpochMixin,
    GetitemMixin,
    BuildFrameCutterMixin,
    BuildGlobalContextImageMixin,
    Dataset,
):
    pass

class SyntheticDefectDataset(
    Construction2Mixin,
    Epoch2Mixin,
    Getitem2Mixin,
    BuildFrameCutter2Mixin,
    SampleTopologyParametersMixin,
    Dataset,
):
    """Generate full synthetic frames and cut them into patches like real images."""

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    PatchSampleRequest,
    _unwrap_tech_augmented_mask,
    _extract_binary_single_channel,
    _broadcast_augmented_mask,
    _is_binary_like_plane,
    _apply_binary_tech_augmentation,
    _apply_binary_tech_augmentation_to_pair,
    _apply_binary_tech_augmentation_with_seed,
    _apply_binary_tech_augmentation_to_pair_with_seed,
    summarise_list,
    index_in_list,
)
