"""model_train_and_recognition: create validation export cache."""
from __future__ import annotations
from pathlib import Path
from neuralimage.model.NeuralNetwork.dataset import NoCutDataset
from neuralimage.lib.images import ImagePreparator, SampleCalculator

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import VALIDATION_EXPORT_CACHE_MAX_BYTES

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.validation_no_cut_frame import (
    _ValidationExportCache,
    _ValidationNoCutFrameExportCache,
)


class CreateValidationExportCacheMixin:
    def _create_validation_export_cache(self) -> _ValidationExportCache | None:
        if (
            not self._save_validation_binary_images
            and not (self._advanced_validation_enabled() and self._advanced_validation_full_frame())
        ) or self._val_dataloader is None:
            return None

        dataset = self._resolve_validation_dataset()
        base_dataset = self._unwrap_validation_dataset(dataset)
        if dataset is None or base_dataset is None:
            return None

        if not isinstance(base_dataset, NoCutDataset):
            return None
        if not self._can_cache_no_cut_validation_export(base_dataset):
            return None

        cut_settings = getattr(base_dataset, '_cut_settings', None)
        segment_size = tuple(getattr(cut_settings, 'segment_size', (0, 0)))
        if len(segment_size) != 2:
            return None

        frame_predictions: dict[int, _ValidationNoCutFrameExportCache] = {}
        estimated_bytes = 0
        for frame_index, (image_path, label_path) in enumerate(getattr(base_dataset, 'samples', [])):
            prepared_size = ImagePreparator(Path(image_path), getattr(base_dataset, '_prep_settings')).size
            if len(prepared_size) != 2:
                return None
            base_width, base_height = int(prepared_size[0]), int(prepared_size[1])
            sample_calculator = SampleCalculator((base_height, base_width), cut_settings)
            _ = len(sample_calculator)
            width_steps, height_steps = sample_calculator.size
            parts_count = max(0, int(width_steps * height_steps))
            if parts_count <= 0:
                continue

            estimated_bytes += parts_count * int(segment_size[0]) * int(segment_size[1]) * 2
            if estimated_bytes > VALIDATION_EXPORT_CACHE_MAX_BYTES:
                return None

            frame_length = int(getattr(base_dataset, '_frame_lengths', [parts_count])[frame_index])
            frame_predictions[frame_index] = _ValidationNoCutFrameExportCache(
                frame_index=int(frame_index),
                image_path=Path(image_path),
                label_path=Path(label_path),
                baseim_size=(base_width, base_height),
                overlap=max(0, int(segment_size[0]) - int(getattr(cut_settings, 'step', segment_size[0]))),
                parts_count=int(parts_count),
                part_lookup=self._build_no_cut_frame_part_lookup(
                    base_dataset,
                    frame_index=int(frame_index),
                    parts_count=int(frame_length),
                ),
                patches={},
                confidence_patches={},
            )

        if not frame_predictions:
            return None
        return _ValidationExportCache(
            mode='no_cut',
            dataset=dataset,
            frame_predictions=frame_predictions,
        )
