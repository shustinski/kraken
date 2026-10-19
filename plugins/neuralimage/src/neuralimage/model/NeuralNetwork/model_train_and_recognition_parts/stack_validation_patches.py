"""model_train_and_recognition: stack validation patches."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Mapping
import numpy as np


class StackValidationPatchesMixin:
    @staticmethod
    def _stack_validation_patches(
        frame_cache: _ValidationNoCutFrameExportCache,
        patches: Mapping[int, np.ndarray],
    ) -> np.ndarray | None:
        if len(patches) < int(frame_cache.parts_count):
            return None
        sample_patch = next(iter(patches.values()), None)
        if sample_patch is None or np.asarray(sample_patch).ndim != 3:
            return None
        sample_array = np.asarray(sample_patch)
        stacked = np.zeros(
            (
                int(frame_cache.parts_count),
                int(sample_array.shape[0]),
                int(sample_array.shape[1]),
                int(sample_array.shape[2]),
            ),
            dtype=np.float32,
        )
        for location, patch in patches.items():
            if 0 <= int(location) < int(frame_cache.parts_count):
                stacked[int(location)] = np.asarray(patch, dtype=np.float32)
        return stacked

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.validation_no_cut_frame import _ValidationNoCutFrameExportCache
