"""model_train_and_recognition: inject local batch into."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
import torch.nn.functional as F
from neuralimage.losses.topograph import (
    extract_critical_region_mask,
)
from neuralimage.losses.topograph_viz import render_critical_regions_overlay
from neuralimage.targets.dataset_hooks import extract_primary_label


class InjectLocalBatchIntoMixin:
    @staticmethod
    def _inject_local_batch_into_global_context(
        global_image: torch.Tensor,
        local_image: torch.Tensor,
        patch_coords_norm: torch.Tensor,
    ) -> torch.Tensor:
        if global_image.ndim != 4 or local_image.ndim != 4:
            return global_image
        batch_size = int(local_image.shape[0])
        if int(global_image.shape[0]) == 1 and batch_size > 1:
            global_image = global_image.expand(batch_size, -1, -1, -1)
        if int(global_image.shape[0]) != batch_size:
            return global_image
        if patch_coords_norm.ndim == 1:
            patch_coords_norm = patch_coords_norm.view(1, 4)
        if patch_coords_norm.ndim != 2 or int(patch_coords_norm.shape[-1]) != 4:
            return global_image
        if int(patch_coords_norm.shape[0]) == 1 and batch_size > 1:
            patch_coords_norm = patch_coords_norm.expand(batch_size, -1)
        if int(patch_coords_norm.shape[0]) != batch_size:
            return global_image

        _, _, global_h, global_w = global_image.shape
        updated = global_image.clone()
        coords = torch.clamp(
            patch_coords_norm.detach().to(device='cpu', dtype=torch.float32),
            0.0,
            1.0,
        )
        for sample_index in range(batch_size):
            x0, y0, x1, y1 = [float(value) for value in coords[sample_index].tolist()]
            left = max(0, min(global_w, int(round(x0 * global_w))))
            right = max(0, min(global_w, int(round(x1 * global_w))))
            top = max(0, min(global_h, int(round(y0 * global_h))))
            bottom = max(0, min(global_h, int(round(y1 * global_h))))
            if right <= left:
                right = min(global_w, left + 1)
            if bottom <= top:
                bottom = min(global_h, top + 1)
            if right <= left or bottom <= top:
                continue
            resized_local = F.interpolate(
                local_image[sample_index:sample_index + 1],
                size=(bottom - top, right - left),
                mode='bilinear',
                align_corners=False,
            )
            updated[sample_index:sample_index + 1, :, top:bottom, left:right] = resized_local
        return updated

    def _sync_global_context_with_local_patch(self, image: Any) -> Any:
        if not isinstance(image, Mapping):
            return image
        local_image = image.get('local_image')
        global_image = image.get('global_image')
        patch_coords_norm = image.get('patch_coords_norm')
        if (
            not torch.is_tensor(local_image)
            or not torch.is_tensor(global_image)
            or not torch.is_tensor(patch_coords_norm)
        ):
            return image

        updated_global = self._inject_local_batch_into_global_context(
            global_image,
            local_image,
            patch_coords_norm,
        )
        updated = dict(image)
        updated['global_image'] = updated_global
        context_image = updated.get('context_image')
        if torch.is_tensor(context_image) and tuple(context_image.shape) == tuple(updated_global.shape):
            updated['context_image'] = updated_global
        return updated

    def _apply_cutout_to_input(self, image: Any) -> Any:
        if not isinstance(image, Mapping):
            return self._apply_cutout_to_batch(image)
        local_image = image.get('local_image')
        if not torch.is_tensor(local_image):
            return image
        updated = dict(image)
        updated['local_image'] = self._apply_cutout_to_batch(local_image)
        return self._sync_global_context_with_local_patch(updated)

    def _apply_random_artifacts_to_input(self, image: Any) -> Any:
        if not isinstance(image, Mapping):
            return self._apply_random_artifacts_to_batch(image)
        local_image = image.get('local_image')
        if not torch.is_tensor(local_image):
            return image
        updated = dict(image)
        updated['local_image'] = self._apply_random_artifacts_to_batch(local_image)
        return self._sync_global_context_with_local_patch(updated)

    def _apply_training_batch_augmentations(
        self,
        image: Any,
        label: torch.Tensor | Mapping[str, torch.Tensor],
        sample_indices: Any,
    ) -> tuple[Any, torch.Tensor | Mapping[str, torch.Tensor], Any, torch.Tensor | None]:
        image, label, mixup_pair_indices, mixup_lambdas = self._apply_mixup_to_batch(
            image,
            label,
            sample_indices,
        )
        image = self._apply_cutout_to_input(image)
        image = self._apply_random_artifacts_to_input(image)
        return image, label, mixup_pair_indices, mixup_lambdas

    def _publish_batch_preview(
        self,
        *,
        epoch: int,
        batch_index: int,
        train_size: int,
        data: Any,
        target: Any,
        outputs: Any,
        preview_stride: int,
        sample_indices: Any = None,
    ) -> None:
        if not self._show_batch_preview:
            return
        if not ((batch_index % preview_stride == 0) or (batch_index == train_size - 1)):
            return
        preview_image = self._tensor_to_preview_array(data[0])
        preview_target = extract_primary_label(target) if isinstance(target, Mapping) else target
        preview_label = self._tensor_to_preview_array(preview_target[0])
        primary_outputs = self._sanitize_outputs_for_loss(outputs)
        preview_outputs = self._tensor_to_preview_array(torch.sigmoid(primary_outputs[0].detach()))
        preview_payload: dict[str, Any] = {
            'type': 'train_batch_preview',
            'epoch': int(epoch + 1),
            'batch_index': int(batch_index + 1),
            'sample_name': self._describe_train_preview_sample(sample_indices, batch_index),
            'image': preview_image,
            'label': preview_label,
            'outputs': preview_outputs,
        }
        if self._topograph_is_active() and bool(getattr(self, '_topograph_debug_viz', False)):
            preview_label_tensor = preview_target[0:1]
            if preview_label_tensor.ndim == 3:
                preview_label_tensor = preview_label_tensor.unsqueeze(1)
            with torch.no_grad():
                critical_mask = extract_critical_region_mask(
                    primary_outputs[0:1],
                    preview_label_tensor,
                    use_c=bool(getattr(self, '_topograph_use_c', False)),
                )
            preview_payload['topograph_overlay'] = render_critical_regions_overlay(
                preview_outputs,
                critical_mask.cpu().numpy(),
            )
        self._bus.put(['metrics', preview_payload])
