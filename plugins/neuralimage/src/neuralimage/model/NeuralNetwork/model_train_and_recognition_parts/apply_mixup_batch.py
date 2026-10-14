"""model_train_and_recognition: apply mixup batch."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
import torch


class ApplyMixupBatchMixin:
    def _apply_mixup_to_batch(
        self,
        image: Any,
        label: torch.Tensor | Mapping[str, torch.Tensor],
        sample_indices: Any,
    ) -> tuple[Any, torch.Tensor | Mapping[str, torch.Tensor], Any, torch.Tensor | None]:
        enabled, probability, alpha = self._resolved_mixup_parameters()
        local_image = self._extract_local_image(image)
        batch_size = int(local_image.size(0))
        if isinstance(image, Mapping) and torch.is_tensor(image.get('patch_coords_norm')):
            # A mixed local patch cannot be assigned one physically correct
            # full-frame coordinate. Keep coordinate-aware batches geometrically
            # well-defined and leave mixup to models without spatial metadata.
            return image, label, None, None
        if (not enabled) or batch_size <= 1 or probability <= 0.0 or alpha <= 0.0:
            return image, label, None, None
        if float(np.random.random()) > probability:
            return image, label, None, None

        lambda_value = float(np.random.beta(alpha, alpha))
        lambda_value = float(min(max(lambda_value, 0.0), 1.0))
        permutation = torch.randperm(batch_size, device=local_image.device)
        order = torch.arange(batch_size, device=local_image.device)
        if bool((permutation == order).any()):
            permutation = torch.roll(order, shifts=1)

        lambda_tensor = local_image.new_full((batch_size,), lambda_value)
        lambda_view = lambda_tensor.view(batch_size, 1, 1, 1)
        mixed_image = self._mixup_batch_input(image, permutation, lambda_view)
        if isinstance(label, Mapping):
            mixed_label: torch.Tensor | Mapping[str, torch.Tensor] = {}
            for key, value in label.items():
                if not torch.is_tensor(value) or value.ndim < 1 or int(value.shape[0]) != batch_size:
                    mixed_label[str(key)] = value
                    continue
                if str(key).endswith('__valid'):
                    mixed_label[str(key)] = torch.minimum(value, value[permutation])
                else:
                    mixed_label[str(key)] = (lambda_view * value) + ((1.0 - lambda_view) * value[permutation])
        else:
            mixed_label = (lambda_view * label) + ((1.0 - lambda_view) * label[permutation])
        pair_indices = self._permute_sample_indices(sample_indices, permutation)
        return mixed_image, mixed_label, pair_indices, lambda_tensor

    @staticmethod
    def _mixup_batch_input(data: Any, permutation: torch.Tensor, lambda_view: torch.Tensor) -> Any:
        if isinstance(data, Mapping):
            mixed: dict[str, Any] = {}
            for key, value in data.items():
                if torch.is_tensor(value) and value.ndim >= 4 and int(value.shape[0]) == int(permutation.shape[0]):
                    mixed[str(key)] = (lambda_view * value) + ((1.0 - lambda_view) * value[permutation])
                else:
                    mixed[str(key)] = value
            return mixed
        if torch.is_tensor(data):
            return (lambda_view * data) + ((1.0 - lambda_view) * data[permutation])
        raise TypeError(f'Unsupported batch input type: {type(data)!r}')

    def _apply_cutout_to_batch(self, image: torch.Tensor) -> torch.Tensor:
        enabled, probability, holes, size_ratio = self._resolved_cutout_parameters()
        if (not enabled) or probability <= 0.0 or size_ratio <= 0.0:
            return image

        batch_size, _channels, height, width = image.shape
        max_cutout_height = max(1, min(height, int(round(height * size_ratio))))
        max_cutout_width = max(1, min(width, int(round(width * size_ratio))))
        if max_cutout_height <= 0 or max_cutout_width <= 0:
            return image

        cutout_image = image.clone()
        for sample_index in range(batch_size):
            if float(np.random.random()) > probability:
                continue
            for _ in range(holes):
                cutout_height = (
                    1
                    if max_cutout_height == 1
                    else int(torch.randint(1, max_cutout_height + 1, (1,), device=image.device).item())
                )
                cutout_width = (
                    1
                    if max_cutout_width == 1
                    else int(torch.randint(1, max_cutout_width + 1, (1,), device=image.device).item())
                )
                max_top = max(0, height - cutout_height)
                max_left = max(0, width - cutout_width)
                top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,), device=image.device).item())
                left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,), device=image.device).item())
                fill_color = torch.rand((_channels, 1, 1), device=image.device, dtype=image.dtype)
                cutout_image[sample_index, :, top:top + cutout_height, left:left + cutout_width] = fill_color
        return cutout_image
