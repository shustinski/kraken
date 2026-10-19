"""model_train_and_recognition: collect validation export batch."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import numpy as np
import torch
from neuralimage.model.NeuralNetwork.dataset import NoCutDataset, index_in_list


class CollectValidationExportBatchMixin:
    def _collect_validation_export_batch(
        self,
        export_cache: _ValidationExportCache | None,
        *,
        sample_indices: Any,
        probs: torch.Tensor,
        confidence: torch.Tensor | None = None,
        saved_images: int,
    ) -> None:
        if export_cache is None or probs.ndim < 4:
            return

        resolved_indices = self._resolve_validation_sample_indices(
            sample_indices,
            fallback_start=saved_images,
            batch_size=int(probs.shape[0]),
        )
        if not resolved_indices:
            return

        cpu_probs = probs.detach().cpu().to(dtype=torch.float16).numpy()
        if export_cache.mode != 'no_cut':
            return

        frame_predictions = export_cache.frame_predictions
        base_dataset = self._unwrap_validation_dataset(export_cache.dataset)
        if frame_predictions is None or not isinstance(base_dataset, NoCutDataset):
            return

        lookup_lengths = getattr(base_dataset, '_lookup_len_list', None)
        if not lookup_lengths:
            return

        for batch_offset, sample_index in enumerate(resolved_indices):
            if batch_offset >= int(cpu_probs.shape[0]):
                break
            try:
                frame_index, local_part = index_in_list(int(sample_index), list(lookup_lengths))
            except (IndexError, ValueError):
                continue
            frame_cache = frame_predictions.get(int(frame_index))
            if frame_cache is None:
                continue
            actual_part = int(local_part)
            if frame_cache.part_lookup is not None:
                if actual_part < 0 or actual_part >= len(frame_cache.part_lookup):
                    continue
                actual_part = int(frame_cache.part_lookup[actual_part])
            location, rotation_index, scale_variant, augmentation_variant = self._decode_no_cut_export_item(
                base_dataset,
                actual_part,
            )
            if rotation_index != 0 or scale_variant != 0 or augmentation_variant != 0:
                continue
            if location < 0 or location >= frame_cache.parts_count or location in frame_cache.patches:
                continue
            frame_cache.patches[int(location)] = np.ascontiguousarray(cpu_probs[batch_offset])
            if confidence is not None and batch_offset < int(confidence.shape[0]):
                frame_cache.confidence_patches[int(location)] = np.ascontiguousarray(
                    confidence[batch_offset].detach().cpu().to(dtype=torch.float16).numpy()
                )

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.validation_no_cut_frame import _ValidationExportCache
