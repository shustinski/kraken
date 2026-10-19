"""model_train_and_recognition: save validation binary predictions 2."""
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
from collections.abc import Callable
from typing import Any, ContextManager
import numpy as np
import torch
from PIL import Image
from neuralimage.model.NeuralNetwork.recognition_pipeline import (
    cut_image_prepare as _cut_image_prepare,
    gpu_predict as _gpu_predict,
    sew as _sew,
)


class SaveValidationBinaryPredictions2Mixin:
    def _save_validation_binary_predictions(
        self,
        *,
        epoch: int,
        device: torch.device,
        autocast_ctx: Callable[[], ContextManager[Any]],
        threshold: float,
        export_cache: _ValidationExportCache | None = None,
    ) -> None:
        from neuralimage.model.NeuralNetwork.model_train_and_recognition import _cut_image_prepare, _gpu_predict, _sew

        if (not self._save_validation_binary_images) or self._val_dataloader is None:
            return

        epoch_dir = self._save_path.parent / 'validation_binary_predictions' / f'epoch_{int(epoch + 1):04d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        saved_images = self._save_validation_binary_predictions_from_cache(
            epoch_dir=epoch_dir,
            threshold=float(threshold),
            export_cache=export_cache,
        )
        if saved_images is not None:
            self._bus.put([
                'logging',
                (
                    f'Validation predictions saved in recognition-style stitched form: {int(saved_images)} file(s) '
                    f'in {epoch_dir}.'
                ),
            ])
            return

        saved_images = 0
        export_items = self._resolve_validation_export_items()
        batch_size = max(1, int(getattr(self._val_dataloader, 'batch_size', 1) or 1))

        if export_items:
            with torch.no_grad():
                for export_item in export_items:
                    predicted = _gpu_predict(
                        _cut_image_prepare(
                            Path(export_item['image_path']),
                            tuple(export_item['segment_shape']),
                            int(export_item['overlap']),
                            use_context_branch=bool(export_item['use_context_branch']),
                            use_cross_attention=bool(export_item.get('use_cross_attention', True)),
                            context_crop_size=export_item['context_crop_size'],
                            context_input_size=export_item['context_input_size'],
                        ),
                        self._model,
                        device,
                        batch_size,
                    )
                    _sew(
                        epoch_dir,
                        predicted,
                        jpeg_quality=95,
                        threshold=float(threshold),
                        postprocess_kernel_size=0,
                    )
                    saved_images += 1
        else:
            dataset = self._resolve_validation_dataset()
            if dataset is None:
                return
            with torch.no_grad():
                for batch in self._val_dataloader:
                    data, _target, sample_indices = self._split_batch(batch)
                    inputs = self._move_batch_input_to_device(data, device)
                    with autocast_ctx():
                        outputs = self._forward_model(inputs)
                    probs = torch.sigmoid(self._sanitize_outputs_for_loss(outputs))
                    binary_predictions = (probs >= float(threshold)).detach().cpu()
                    resolved_indices = self._resolve_validation_sample_indices(
                        sample_indices,
                        fallback_start=saved_images,
                        batch_size=int(binary_predictions.shape[0]),
                    )

                    for batch_offset in range(int(binary_predictions.shape[0])):
                        sample_index = (
                            int(resolved_indices[batch_offset]) if batch_offset < len(resolved_indices) else saved_images
                        )
                        sample_name = self._describe_validation_sample(dataset, sample_index, saved_images)
                        sample_path = epoch_dir / f'{saved_images:06d}_{sample_name}.png'
                        sample_tensor = binary_predictions[batch_offset]
                        if sample_tensor.ndim == 3:
                            sample_tensor = sample_tensor[0]
                        sample_array = sample_tensor.numpy().astype(np.uint8) * 255
                        Image.fromarray(sample_array, mode='L').save(sample_path)
                        saved_images += 1

        self._bus.put([
            'logging',
            (
                f'Validation predictions saved in recognition-style stitched form: {saved_images} file(s) '
                f'in {epoch_dir}.'
            ),
        ])

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.validation_no_cut_frame import _ValidationExportCache
