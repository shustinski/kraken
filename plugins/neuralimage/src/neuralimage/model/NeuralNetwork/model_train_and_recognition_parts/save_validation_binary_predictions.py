"""model_train_and_recognition: save validation binary predictions."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from neuralimage.model.NeuralNetwork.recognition_pipeline import (
    sew as _sew,
)


class SaveValidationBinaryPredictionsMixin:
    def _save_validation_binary_predictions_from_cache(
        self,
        *,
        epoch_dir: Path,
        threshold: float,
        export_cache: _ValidationExportCache | None,
    ) -> int | None:
        if export_cache is None:
            return None

        if export_cache.mode != 'no_cut':
            return None

        frame_predictions = export_cache.frame_predictions or {}
        if not frame_predictions:
            return None
        for frame_cache in frame_predictions.values():
            if len(frame_cache.patches) < int(frame_cache.parts_count):
                return None

        saved_images = 0
        for frame_cache in frame_predictions.values():
            predicted = self._stack_validation_patches(frame_cache, frame_cache.patches)
            if predicted is None:
                continue
            _sew(
                epoch_dir,
                {
                    'name': frame_cache.image_path.name,
                    'baseim_size': frame_cache.baseim_size,
                    'overlap': int(frame_cache.overlap),
                    'predicted_image': predicted,
                },
                jpeg_quality=95,
                threshold=float(threshold),
                postprocess_kernel_size=0,
            )
            saved_images += 1
        return int(saved_images)

    def _describe_validation_sample(self, dataset: Any, sample_index: int, fallback_index: int) -> str:
        describe_fn = getattr(dataset, 'describe_sample', None)
        if callable(describe_fn):
            try:
                return self._sanitize_artifact_name(str(describe_fn(int(sample_index))), fallback='sample')
            except Exception:
                pass
        return f'sample_{int(fallback_index):06d}'

    def _describe_train_preview_sample(self, sample_indices: Any, fallback_batch_index: int) -> str:
        if sample_indices is None:
            return f'batch_{int(fallback_batch_index) + 1:06d}'
        resolved_indices = self._resolve_validation_sample_indices(
            sample_indices,
            fallback_start=int(fallback_batch_index),
            batch_size=1,
        )
        if resolved_indices:
            sample_index = int(resolved_indices[0])
            dataset = getattr(getattr(self, '_train_dataloader', None), 'dataset', None)
            describe_fn = getattr(dataset, 'describe_sample', None)
            if callable(describe_fn):
                try:
                    sample_name = str(describe_fn(sample_index)).strip()
                    if sample_name:
                        return Path(sample_name).name
                except Exception:
                    pass
            return f'sample_{sample_index:06d}'
        return f'batch_{int(fallback_batch_index) + 1:06d}'
