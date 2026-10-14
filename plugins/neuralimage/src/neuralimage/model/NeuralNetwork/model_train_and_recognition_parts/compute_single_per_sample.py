"""model_train_and_recognition: compute single per sample."""
from __future__ import annotations
from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComputeSinglePerSampleMixin:
    def _compute_single_per_sample_loss(
        self,
        loss_mode: str,
        outputs: torch.Tensor,
        label: torch.Tensor,
        bce_criterion: nn.Module,
        *,
        apply_pixel_mining: bool = False,
    ) -> torch.Tensor:
        if loss_mode in ('ce', 'ce_dice'):
            logits_two_class = torch.cat([-outputs, outputs], dim=1)
            soft_label = label[:, 0, :, :]
            target_probs = torch.stack((1.0 - soft_label, soft_label), dim=1)
            log_probs = F.log_softmax(logits_two_class, dim=1)
            ce_map = -(target_probs * log_probs).sum(dim=1)
            valid_mask = torch.ones_like(soft_label, dtype=ce_map.dtype)
            ce_per_sample = self._reduce_loss_map_per_sample(
                ce_map,
                loss_mode=loss_mode,
                apply_pixel_mining=apply_pixel_mining,
                valid_mask=valid_mask,
            )
            ce_per_sample = torch.nan_to_num(ce_per_sample, nan=1.0, posinf=50.0, neginf=0.0)
            if loss_mode == 'ce':
                return torch.clamp(ce_per_sample, min=0.0, max=50.0)

            probs_fg = torch.softmax(logits_two_class, dim=1)[:, 1, :, :]
            probs_flat = probs_fg.view(probs_fg.shape[0], -1)
            label_flat = soft_label.view(soft_label.shape[0], -1)
            valid_flat = valid_mask.view(valid_mask.shape[0], -1)
            probs_flat = probs_flat * valid_flat
            label_flat = label_flat * valid_flat
            dice_per_sample, _ = self._compute_soft_overlap_losses_per_sample(
                probs_flat,
                label_flat,
            )
            dice_weight = self._resolved_dice_weight()
            mixed = ((1.0 - dice_weight) * ce_per_sample) + (dice_weight * dice_per_sample)
            mixed = torch.nan_to_num(mixed, nan=1.0, posinf=50.0, neginf=0.0)
            return torch.clamp(mixed, min=0.0, max=50.0)

        if loss_mode == 'boundary':
            return self._compute_boundary_loss_per_sample(outputs, label)

        if loss_mode == 'cldice':
            return self._compute_cldice_loss_per_sample(outputs, label)

        if loss_mode == 'focal_tversky':
            return self._compute_focal_tversky_loss_per_sample(outputs, label)

        bce_loss_map = cast(torch.Tensor, bce_criterion(outputs, label))
        pointwise_loss_map = bce_loss_map
        if loss_mode in ('focal_bce', 'focal_dice', 'focal_iou'):
            pointwise_loss_map = self._compute_focal_bce_loss_map(outputs, label, bce_loss_map)
        pointwise_per_sample = self._reduce_loss_map_per_sample(
            pointwise_loss_map,
            loss_mode=loss_mode,
            apply_pixel_mining=apply_pixel_mining,
        )
        pointwise_per_sample = torch.nan_to_num(pointwise_per_sample, nan=1.0, posinf=50.0, neginf=0.0)
        if loss_mode in ('bce', 'focal_bce'):
            return torch.clamp(pointwise_per_sample, min=0.0, max=50.0)

        probs = torch.sigmoid(outputs)
        probs_flat = probs.view(probs.shape[0], -1)
        label_flat = label.view(label.shape[0], -1)
        dice_per_sample, iou_per_sample = self._compute_soft_overlap_losses_per_sample(
            probs_flat,
            label_flat,
        )
        if loss_mode == 'dice':
            return dice_per_sample
        if loss_mode == 'iou':
            return iou_per_sample

        if loss_mode in ('bce_dice', 'focal_dice'):
            dice_weight = self._resolved_dice_weight()
            mixed = ((1.0 - dice_weight) * pointwise_per_sample) + (dice_weight * dice_per_sample)
            mixed = torch.nan_to_num(mixed, nan=1.0, posinf=50.0, neginf=0.0)
            return torch.clamp(mixed, min=0.0, max=50.0)

        iou_weight = self._resolved_iou_weight()
        mixed = ((1.0 - iou_weight) * pointwise_per_sample) + (iou_weight * iou_per_sample)
        mixed = torch.nan_to_num(mixed, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(mixed, min=0.0, max=50.0)
