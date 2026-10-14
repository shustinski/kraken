"""dataset: construction 2."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
from neuralimage.augmentations import (
    ICDefectAugmentor,
    PCBDefectAugmentor,
)
from neuralimage.lib.data_interfaces import TrainingParameters
from neuralimage.lib.data_interfaces import (
    build_synthetic_defect_generator_parameters,
)
from neuralimage.lib.images import SampleCalculator, SampleFastCutter
from neuralimage.model.NeuralNetwork.context_utils import (
    normalize_size_pair,
)


class Construction2Mixin:
    def __init__(
        self,
        sample_count: int,
        settings: TrainingParameters,
        *,
        apply_train_only_transforms: bool = True,
    ) -> None:
        self.colors = int(settings.colors)
        self._generation = settings.generation
        self._skip_uniform_labels = bool(getattr(settings, 'skip_uniform_labels', False))
        self._config = build_synthetic_defect_generator_parameters(
            getattr(settings, 'synthetic_defect_generator', None)
        )
        self._apply_train_only_transforms = bool(apply_train_only_transforms)
        self.shuffle_frames = bool(getattr(settings, 'shuffle', True))
        self.shuffle_patches_in_frame = bool(
            getattr(self._generation, 'shuffle_patches_in_frame', self.shuffle_frames)
        )
        self._use_context_branch = bool(getattr(settings, 'use_context_branch', False))
        self._use_cross_attention = bool(getattr(settings, 'use_cross_attention', True))
        self._local_crop_size = normalize_size_pair(
            getattr(settings, 'local_crop_size', None),
            fallback=tuple(settings.generation.segment_size),
        )
        self._context_crop_size = normalize_size_pair(
            getattr(settings, 'context_crop_size', None),
            fallback=(self._local_crop_size[0] * 2, self._local_crop_size[1] * 2),
        )
        self._context_input_size = normalize_size_pair(
            getattr(settings, 'context_input_size', None),
            fallback=self._local_crop_size,
        )
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
        self._frame_size_xy = (
            max(int(self._local_crop_size[0]), int(self._config.image_size_xy[0])),
            max(int(self._local_crop_size[1]), int(self._config.image_size_xy[1])),
        )
        self._frame_count = max(0, int(sample_count))
        self._static_parts_per_frame = max(
            0,
            int(
                len(
                    SampleCalculator(
                        (int(self._frame_size_xy[1]), int(self._frame_size_xy[0])),
                        self._generation,
                    )
                )
            ),
        )
        self._dynamic_frame_lengths = self._skip_uniform_labels
        self._epoch_index = 0
        self._frame_order = list(range(self._frame_count))
        self._frame_lengths: list[int] = []
        self._lookup_len_list: list[int] = []
        self._samples_amount: int = 0
        self._current_frame_index: int | None = None
        self._current_frame_cutter: SampleFastCutter | None = None
        self._frame_cache: OrderedDict[tuple[int, int, bool], SampleFastCutter] = OrderedDict()
        self._global_context_cache: OrderedDict[tuple[int, str, tuple[int, int]], np.ndarray] = OrderedDict()
        self._frame_cache_limit = 2
        defect_augmentor_cls = ICDefectAugmentor if self._config.topology_domain == 'ic' else PCBDefectAugmentor
        self._defect_augmentor = (
            defect_augmentor_cls(self._config.defects)
            if self._apply_train_only_transforms and self._config.enabled and self._config.defects.enabled
            else None
        )
        self._calculate_len()
