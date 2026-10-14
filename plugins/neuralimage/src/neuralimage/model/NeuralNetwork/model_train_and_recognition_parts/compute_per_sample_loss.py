"""model_train_and_recognition: compute per sample loss."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralimage.losses.distance_boundary import compute_distance_boundary_loss


class ComputePerSampleLossMixin:
    def _compute_per_sample_loss_for_logits(
        self,
        outputs: torch.Tensor,
        label: torch.Tensor,
        bce_criterion: nn.Module,
        *,
        apply_pixel_mining: bool = False,
    ) -> torch.Tensor:
        outputs = self._sanitize_outputs_for_loss(outputs)
        label = self._sanitize_labels_for_loss(label)
        combined_loss: torch.Tensor | None = None
        for loss_mode, coefficient in self._resolved_loss_term_weights().items():
            single_loss = self._compute_single_per_sample_loss(
                loss_mode,
                outputs,
                label,
                bce_criterion,
                apply_pixel_mining=apply_pixel_mining,
            )
            weighted_loss = single_loss * float(coefficient)
            combined_loss = weighted_loss if combined_loss is None else (combined_loss + weighted_loss)
        if combined_loss is None:
            fallback_loss = self._compute_single_per_sample_loss(
                self._resolved_loss_function(),
                outputs,
                label,
                bce_criterion,
                apply_pixel_mining=apply_pixel_mining,
            )
            return torch.clamp(torch.nan_to_num(fallback_loss, nan=1.0, posinf=50.0, neginf=0.0), min=0.0, max=50.0)
        combined_loss = torch.nan_to_num(combined_loss, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(combined_loss, min=0.0, max=50.0)

    def _compute_per_sample_loss(
        self,
        outputs: Any,
        label: torch.Tensor | Mapping[str, torch.Tensor],
        bce_criterion: nn.Module,
        *,
        apply_pixel_mining: bool = False,
    ) -> torch.Tensor:
        raw_label = label
        label = self._sanitize_labels_for_loss(label)
        supervised_outputs = self._iter_supervised_output_tensors(outputs)
        primary_output = self._resize_output_like_label(supervised_outputs[0], label)
        if len(supervised_outputs) == 1:
            combined_loss = self._compute_per_sample_loss_for_logits(
                supervised_outputs[0],
                label,
                bce_criterion,
                apply_pixel_mining=apply_pixel_mining,
            )
        else:
            combined_loss: torch.Tensor | None = None
            for output_weight, output_tensor in zip(
                self._deep_supervision_weights(len(supervised_outputs)),
                supervised_outputs,
            ):
                resized_output = self._resize_output_like_label(output_tensor, label)
                output_loss = self._compute_per_sample_loss_for_logits(
                    resized_output,
                    label,
                    bce_criterion,
                    apply_pixel_mining=apply_pixel_mining,
                )
                weighted_loss = output_loss * float(output_weight)
                combined_loss = weighted_loss if combined_loss is None else (combined_loss + weighted_loss)

            if combined_loss is None:
                raise RuntimeError('Deep supervision loss aggregation produced no supervised outputs.')
            combined_loss = torch.nan_to_num(combined_loss, nan=1.0, posinf=50.0, neginf=0.0)
            combined_loss = torch.clamp(combined_loss, min=0.0, max=50.0)

        learned_terms: dict[str, torch.Tensor] | None = (
            {'mask': combined_loss}
            if self._uses_homoscedastic_loss_weighting() and getattr(self, '_loss_weighter', None) is not None
            else None
        )
        confidence_output = self._extract_confidence_output_tensor(outputs)
        if confidence_output is not None:
            confidence_output = self._resize_output_like_label(confidence_output, label)
            confidence_target = 1.0 - torch.abs(torch.sigmoid(primary_output).detach() - label)
            confidence_target = torch.clamp(
                torch.nan_to_num(confidence_target, nan=0.0, posinf=1.0, neginf=0.0),
                min=0.0,
                max=1.0,
            )
            confidence_loss_map = F.binary_cross_entropy_with_logits(
                confidence_output,
                confidence_target,
                reduction='none',
            )
            confidence_loss_per_sample = self._reduce_loss_map_per_sample(
                confidence_loss_map,
                loss_mode='confidence',
                apply_pixel_mining=False,
            )
            confidence_loss_per_sample = torch.nan_to_num(
                confidence_loss_per_sample,
                nan=1.0,
                posinf=50.0,
                neginf=0.0,
            )
            confidence_loss_per_sample = torch.clamp(confidence_loss_per_sample, min=0.0, max=50.0)
            if learned_terms is not None:
                learned_terms['confidence'] = confidence_loss_per_sample
            else:
                combined_loss = torch.clamp(
                    combined_loss + (confidence_loss_per_sample * self._resolved_confidence_loss_weight()),
                    min=0.0,
                    max=50.0,
                )

        if learned_terms is not None:
            learned_terms.update(self._compute_auxiliary_supervision_terms(outputs, raw_label))
        else:
            auxiliary_loss = self._compute_auxiliary_supervision_loss(outputs, raw_label)
            if auxiliary_loss is not None:
                combined_loss = torch.clamp(combined_loss + auxiliary_loss, min=0.0, max=50.0)
        if isinstance(raw_label, Mapping):
            supervision = self._resolved_supervision_targets()
            distance_weight = max(0.0, float(getattr(supervision, 'distance_boundary_weight', 0.0)))
            sdf_target = raw_label.get('sdf')
            if distance_weight > 0.0 and torch.is_tensor(sdf_target):
                distance_loss = compute_distance_boundary_loss(
                    primary_output,
                    sdf_target,
                    valid_mask=raw_label.get('sdf__valid'),
                )
                if learned_terms is not None:
                    learned_terms['distance_boundary'] = distance_loss
                else:
                    combined_loss = torch.clamp(combined_loss + distance_weight * distance_loss, min=0.0, max=50.0)
        if learned_terms is not None:
            combined_loss = getattr(self, '_loss_weighter')(learned_terms)
            combined_loss = torch.clamp(
                torch.nan_to_num(combined_loss, nan=1.0, posinf=50.0, neginf=0.0),
                min=0.0,
                max=50.0,
            )
        topograph_loss = self._compute_topograph_component_loss(primary_output, label)
        if topograph_loss is not None:
            combined_loss = torch.clamp(
                combined_loss + (float(getattr(self, '_topograph_loss_weight', 0.0)) * topograph_loss),
                min=0.0,
                max=50.0,
            )
        self._update_batch_loss_components(
            outputs=primary_output,
            label=label,
            bce_criterion=bce_criterion,
            combined_loss=combined_loss,
            topograph_loss=topograph_loss,
            apply_pixel_mining=apply_pixel_mining,
        )
        return combined_loss
