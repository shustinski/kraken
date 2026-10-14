"""dataset: construction."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import numpy as np
from kraken_core.asset_source import AssetSource
from neuralimage.augmentations import (
    PCBDefectAugmentor,
    TechVariationAugmentor,
)
from neuralimage.lib.data_interfaces import TrainingParameters, SampleGenerationSettings, SamplePrepareSettings
from neuralimage.lib.data_interfaces import (
    build_pcb_defect_parameters,
    build_tech_augmentation_config,
)
from neuralimage.lib.images import SampleFastCutter
from neuralimage.model.NeuralNetwork.context_utils import (
    normalize_size_pair,
)


class ConstructionMixin:
    def __init__(
        self, samples, settings: TrainingParameters, *,
        apply_train_only_transforms: bool = True, asset_source: AssetSource | None = None,
    ):
        self.samples = samples
        self._asset_source = asset_source
        self._sample_folder = Path(settings.image_path)
        self.colors = settings.colors
        self.shuffle_frames = bool(getattr(settings, 'shuffle', True))
        # Backward-compatible alias used by old code/tests.
        self.shuffle = self.shuffle_frames
        self._prep_settings: SamplePrepareSettings = settings.prepare
        self._cut_settings: SampleGenerationSettings = settings.generation
        self._skip_uniform_labels = bool(getattr(settings, 'skip_uniform_labels', False))
        self._rare_patch_oversampling_enabled = bool(
            getattr(settings, 'rare_patch_oversampling_enabled', False)
        )
        self._rare_patch_oversampling_factor = max(
            1,
            int(getattr(settings, 'rare_patch_oversampling_factor', 2)),
        )
        self._use_context_branch = bool(getattr(settings, 'use_context_branch', False))
        self._use_cross_attention = bool(getattr(settings, 'use_cross_attention', True))
        self._local_crop_size = normalize_size_pair(
            getattr(settings, 'local_crop_size', None),
            fallback=tuple(self._cut_settings.segment_size),
        )
        self._context_crop_size = normalize_size_pair(
            getattr(settings, 'context_crop_size', None),
            fallback=(self._local_crop_size[0] * 2, self._local_crop_size[1] * 2),
        )
        self._context_input_size = normalize_size_pair(
            getattr(settings, 'context_input_size', None),
            fallback=self._local_crop_size,
        )
        self._tech_aug_config = build_tech_augmentation_config(getattr(self._cut_settings, 'tech_aug', None))
        self._tech_aug_binary_tolerance = float(
            getattr(self._tech_aug_config, 'binary_tolerance', 0.15)
        )
        self._pcb_defects = build_pcb_defect_parameters(getattr(settings, 'pcb_defects', None))
        self._apply_train_only_transforms = bool(apply_train_only_transforms)
        self._supervision_targets = getattr(settings, 'supervision_targets', None)
        self._preprocessing = getattr(settings, 'preprocessing', None)
        self._sem_augmentation = getattr(settings, 'sem_augmentation', None)
        self._uncertainty = getattr(settings, 'uncertainty', None)
        self._advanced_validation = bool(getattr(settings, 'advanced_validation', True))
        self._advanced_validation_full_frame = bool(
            getattr(settings, 'advanced_validation_full_frame', True)
        )
        self._advanced_validation_boundary_tolerance = max(
            0, int(getattr(settings, 'advanced_validation_boundary_tolerance', 2))
        )
        self._advanced_validation_include_hd95 = bool(
            getattr(settings, 'advanced_validation_include_hd95', True)
        )
        self._advanced_validation_confidence_bins = max(
            2, int(getattr(settings, 'advanced_validation_confidence_bins', 10))
        )
        self._tech_augmentor = (
            TechVariationAugmentor(self._tech_aug_config)
            if self._apply_train_only_transforms and self._tech_aug_config.enabled
            else None
        )
        self._defect_augmentor = (
            PCBDefectAugmentor(self._pcb_defects)
            if self._apply_train_only_transforms and self._pcb_defects.enabled
            else None
        )
        self._use_defect_mask_as_label = bool(self._pcb_defects.use_defect_mask_as_label)
        self.shuffle_patches_in_frame = bool(
            getattr(self._cut_settings, 'shuffle_patches_in_frame', self.shuffle_frames)
        )
        self._dynamic_frame_lengths = self._resolve_dynamic_frame_lengths()
        self._samples_amount: int = 0
        self._frame_lengths: list[int] = []
        self._lookup_len_list: list[int] = []
        self._epoch_index: int = 0

        self._current_frame_index: int | None = None
        self._current_image_cutter: SampleFastCutter | None = None
        self._frame_cache: OrderedDict[tuple[int, int, bool], SampleFastCutter] = OrderedDict()
        self._global_context_cache: OrderedDict[tuple[int, str, tuple[int, int]], np.ndarray] = OrderedDict()
        self._frame_cache_limit = self._resolve_frame_cache_limit()

        self._create_files_list()
        self._calculate_len()
