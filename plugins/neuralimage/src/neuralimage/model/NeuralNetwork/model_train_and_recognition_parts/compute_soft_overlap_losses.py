"""model_train_and_recognition: compute soft overlap losses."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from neuralimage.losses.boundary import compute_boundary_loss, compute_soft_boundary_map
from neuralimage.losses.cldice import compute_cldice_loss, soft_skeletonize

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import (
    BOUNDARY_LOSS_KERNEL_SIZE,
    CLDICE_SKELETON_ITERATIONS,
    FOCAL_LOSS_GAMMA,
    FOCAL_TVERSKY_ALPHA,
    FOCAL_TVERSKY_BETA,
)


class ComputeSoftOverlapLossesMixin:
    @staticmethod
    def _compute_soft_boundary_map(
        mask: torch.Tensor,
        *,
        kernel_size: int = BOUNDARY_LOSS_KERNEL_SIZE,
    ) -> torch.Tensor:
        return compute_soft_boundary_map(mask, kernel_size=kernel_size)

    def _compute_boundary_loss_per_sample(
        self,
        outputs: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        return compute_boundary_loss(outputs, label, kernel_size=BOUNDARY_LOSS_KERNEL_SIZE)

    @staticmethod
    def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
        vertical = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
        horizontal = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
        eroded = torch.minimum(vertical, horizontal)
        eroded = torch.nan_to_num(eroded, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(eroded, min=0.0, max=1.0)

    @staticmethod
    def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        dilated = torch.nan_to_num(dilated, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(dilated, min=0.0, max=1.0)

    def _soft_open(self, mask: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(mask))

    def _soft_skeletonize(
        self,
        mask: torch.Tensor,
        *,
        iterations: int = CLDICE_SKELETON_ITERATIONS,
    ) -> torch.Tensor:
        return soft_skeletonize(mask, iterations=iterations)

    def _compute_cldice_loss_per_sample(
        self,
        outputs: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        supervision = self._resolved_supervision_targets()
        iterations = int(getattr(getattr(supervision, 'basic', None), 'cldice_iterations', CLDICE_SKELETON_ITERATIONS))
        return compute_cldice_loss(outputs, label, iterations=max(0, iterations))

    @staticmethod
    def _compute_focal_tversky_loss_per_sample(
        outputs: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(outputs)
        probs_flat = probs.view(probs.shape[0], -1)
        label_flat = label.view(label.shape[0], -1)
        eps = 1e-6
        true_positive = (probs_flat * label_flat).sum(dim=1)
        false_positive = (probs_flat * (1.0 - label_flat)).sum(dim=1)
        false_negative = ((1.0 - probs_flat) * label_flat).sum(dim=1)
        tversky = (true_positive + eps) / (
            true_positive
            + (FOCAL_TVERSKY_ALPHA * false_positive)
            + (FOCAL_TVERSKY_BETA * false_negative)
            + eps
        )
        focal_tversky_loss = torch.pow(1.0 - tversky, FOCAL_LOSS_GAMMA)
        empty_target_mask = label_flat.sum(dim=1) <= eps
        if bool(empty_target_mask.any()):
            pixel_count = max(1, int(probs_flat.shape[1]))
            empty_target_penalty = false_positive / float(pixel_count)
            empty_target_penalty = torch.clamp(empty_target_penalty, min=0.0, max=1.0)
            focal_tversky_loss = torch.where(empty_target_mask, empty_target_penalty, focal_tversky_loss)
        focal_tversky_loss = torch.nan_to_num(focal_tversky_loss, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(focal_tversky_loss, min=0.0, max=50.0)

    @staticmethod
    def _compute_soft_overlap_losses_per_sample(
        probs_flat: torch.Tensor,
        label_flat: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        intersection = (probs_flat * label_flat).sum(dim=1)
        pred_mass = probs_flat.sum(dim=1)
        target_mass = label_flat.sum(dim=1)

        dice_denom = pred_mass + target_mass
        dice = (2.0 * intersection + eps) / (dice_denom + eps)
        dice_per_sample = 1.0 - dice

        union = pred_mass + target_mass - intersection
        iou = (intersection + eps) / (union + eps)
        iou_per_sample = 1.0 - iou

        empty_target_mask = target_mass <= eps
        if bool(empty_target_mask.any()):
            pixel_count = max(1, int(probs_flat.shape[1]))
            empty_target_penalty = pred_mass / float(pixel_count)
            empty_target_penalty = torch.clamp(empty_target_penalty, min=0.0, max=1.0)
            dice_per_sample = torch.where(empty_target_mask, empty_target_penalty, dice_per_sample)
            iou_per_sample = torch.where(empty_target_mask, empty_target_penalty, iou_per_sample)

        dice_per_sample = torch.nan_to_num(dice_per_sample, nan=1.0, posinf=50.0, neginf=0.0)
        iou_per_sample = torch.nan_to_num(iou_per_sample, nan=1.0, posinf=50.0, neginf=0.0)
        return (
            torch.clamp(dice_per_sample, min=0.0, max=50.0),
            torch.clamp(iou_per_sample, min=0.0, max=50.0),
        )
