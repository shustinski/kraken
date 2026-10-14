"""model_train_and_recognition: compute auxiliary supervision loss."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neuralimage.losses.composite import (
    AuxiliaryHeadLoss,
    HomoscedasticLossWeighter,
    compute_auxiliary_head_loss,
    resolve_auxiliary_head_weights,
)
from neuralimage.model.NeuralNetwork.blocks import extract_confidence_output, extract_mask_outputs
from neuralimage.targets.dataset_hooks import extract_primary_label

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import (
    CONFIDENCE_LOSS_WEIGHT,
    DEEP_SUPERVISION_DECAY,
)


class ComputeAuxiliarySupervisionLossMixin:
    @staticmethod
    def _deep_supervision_weights(count: int) -> tuple[float, ...]:
        resolved_count = max(1, int(count))
        weights = [DEEP_SUPERVISION_DECAY ** index for index in range(resolved_count)]
        total = sum(weights)
        if total <= 0.0:
            return tuple(1.0 / float(resolved_count) for _ in range(resolved_count))
        return tuple(float(weight / total) for weight in weights)

    def _iter_supervised_output_tensors(self, outputs: Any) -> tuple[torch.Tensor, ...]:
        outputs = extract_mask_outputs(outputs)
        if torch.is_tensor(outputs):
            return (self._sanitize_outputs_for_loss(outputs),)
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise TypeError('Model must return a tensor or a non-empty sequence of tensors.')

        sanitized_outputs: list[torch.Tensor] = []
        for output in outputs:
            if not torch.is_tensor(output):
                raise TypeError('All deep supervision outputs must be tensors.')
            sanitized_outputs.append(self._sanitize_outputs_for_loss(output))
        return tuple(sanitized_outputs)

    def _resize_output_like_label(self, outputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if outputs.shape[-2:] == label.shape[-2:]:
            return outputs
        return F.interpolate(outputs, size=label.shape[-2:], mode='bilinear', align_corners=False)

    def _sanitize_outputs_for_loss(self, outputs: Any) -> torch.Tensor:
        outputs = self._extract_primary_output_tensor(outputs)
        sanitized = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        sanitized = torch.where(torch.isposinf(sanitized), torch.full_like(sanitized, 20.0), sanitized)
        sanitized = torch.where(torch.isneginf(sanitized), torch.full_like(sanitized, -20.0), sanitized)
        return torch.clamp(sanitized, min=-20.0, max=20.0)

    def _extract_confidence_output_tensor(self, outputs: Any) -> torch.Tensor | None:
        confidence = extract_confidence_output(outputs)
        if confidence is None:
            return None
        sanitized = torch.where(torch.isnan(confidence), torch.zeros_like(confidence), confidence)
        sanitized = torch.where(torch.isposinf(sanitized), torch.full_like(sanitized, 20.0), sanitized)
        sanitized = torch.where(torch.isneginf(sanitized), torch.full_like(sanitized, -20.0), sanitized)
        return torch.clamp(sanitized, min=-20.0, max=20.0)

    @staticmethod
    def _sanitize_labels_for_loss(label: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(label, Mapping):
            label = extract_primary_label(label)
        sanitized = torch.nan_to_num(label, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(sanitized, min=0.0, max=1.0)

    @staticmethod
    def _resolve_dataset_attribute(dataloader: DataLoader | None, attribute_name: str) -> Any:
        if dataloader is None:
            return None
        dataset = getattr(dataloader, 'dataset', None)
        visited: set[int] = set()
        while dataset is not None and id(dataset) not in visited:
            visited.add(id(dataset))
            value = getattr(dataset, attribute_name, None)
            if value is not None:
                return value
            dataset = getattr(dataset, '_base_dataset', None)
            if dataset is None and hasattr(getattr(dataloader, 'dataset', None), '_datasets'):
                for nested in getattr(dataloader.dataset, '_datasets', []):
                    value = getattr(nested, attribute_name, None)
                    if value is not None:
                        return value
                break
        return None

    def _resolved_supervision_targets(self) -> Any:
        return self._resolve_dataset_attribute(getattr(self, '_train_dataloader', None), '_supervision_targets')

    def _advanced_validation_enabled(self) -> bool:
        loader = getattr(self, '_val_dataloader', None) or getattr(self, '_train_dataloader', None)
        value = self._resolve_dataset_attribute(loader, '_advanced_validation')
        if value is None:
            return True
        return bool(value)

    def _advanced_validation_options(self) -> tuple[int, bool, int]:
        loader = getattr(self, '_val_dataloader', None) or getattr(self, '_train_dataloader', None)
        tolerance = self._resolve_dataset_attribute(loader, '_advanced_validation_boundary_tolerance')
        include_hd95 = self._resolve_dataset_attribute(loader, '_advanced_validation_include_hd95')
        confidence_bins = self._resolve_dataset_attribute(loader, '_advanced_validation_confidence_bins')
        return (
            max(0, int(tolerance if tolerance is not None else 2)),
            bool(True if include_hd95 is None else include_hd95),
            max(2, int(confidence_bins if confidence_bins is not None else 10)),
        )

    def _advanced_validation_full_frame(self) -> bool:
        loader = getattr(self, '_val_dataloader', None) or getattr(self, '_train_dataloader', None)
        value = self._resolve_dataset_attribute(
            loader,
            '_advanced_validation_full_frame',
        )
        return bool(True if value is None else value)

    def _resolved_auxiliary_head_weights(self) -> dict[str, float]:
        supervision = self._resolved_supervision_targets()
        if supervision is None or not getattr(supervision, 'any_enabled', lambda: False)():
            return {}
        enabled = supervision.enabled_targets()
        return resolve_auxiliary_head_weights(
            enabled,
            getattr(supervision, 'auxiliary_head_weights', None),
        )

    def _resolved_confidence_loss_weight(self) -> float:
        uncertainty = self._resolve_dataset_attribute(getattr(self, '_train_dataloader', None), '_uncertainty')
        return max(0.0, float(getattr(uncertainty, 'confidence_loss_weight', CONFIDENCE_LOSS_WEIGHT)))

    def _compute_auxiliary_supervision_loss(
        self,
        outputs: Any,
        label: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if not isinstance(label, Mapping) or not isinstance(outputs, dict):
            return None
        head_weights = self._resolved_auxiliary_head_weights()
        if not head_weights:
            return None
        target_tensors = {
            str(key): value
            for key, value in label.items()
            if str(key) != 'mask' and torch.is_tensor(value)
        }
        output_tensors = {
            str(key): value
            for key, value in outputs.items()
            if str(key) not in {'mask', 'confidence'} and torch.is_tensor(value)
        }
        return compute_auxiliary_head_loss(output_tensors, target_tensors, head_weights=head_weights)

    def _compute_auxiliary_supervision_terms(
        self,
        outputs: Any,
        label: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(label, Mapping) or not isinstance(outputs, dict):
            return {}
        terms: dict[str, torch.Tensor] = {}
        for head_name in self._resolved_auxiliary_head_weights():
            prediction = outputs.get(head_name)
            target = label.get(head_name)
            if not torch.is_tensor(prediction) or not torch.is_tensor(target):
                continue
            terms[head_name] = AuxiliaryHeadLoss.compute_single_head(
                prediction,
                target,
                head_name=head_name,
                valid_mask=label.get(f'{head_name}__valid'),
            )
        return terms

    def _uses_homoscedastic_loss_weighting(self) -> bool:
        return getattr(self, '_loss_weighting_strategy', 'static') == 'homoscedastic_uncertainty'

    def _prepare_loss_weighter(self, device: torch.device) -> None:
        if not self._uses_homoscedastic_loss_weighting():
            self._loss_weighter = None
            return
        term_names = ['mask', *self._resolved_auxiliary_head_weights().keys()]
        uncertainty = self._resolve_dataset_attribute(self._train_dataloader, '_uncertainty')
        if bool(getattr(uncertainty, 'uses_confidence_head', lambda: False)()):
            term_names.append('confidence')
        supervision = self._resolved_supervision_targets()
        if float(getattr(supervision, 'distance_boundary_weight', 0.0)) > 0.0:
            term_names.append('distance_boundary')
        self._loss_weighter = HomoscedasticLossWeighter(
            tuple(dict.fromkeys(term_names)),
            mask_weight_floor=float(self._mask_loss_weight_floor),
        ).to(device)

    def _resolved_cutout_parameters(self) -> tuple[bool, float, int, float]:
        params = getattr(self, '_cutout_params', None)
        enabled = bool(getattr(params, 'enabled', False))
        probability = float(getattr(params, 'probability', 1.0))
        holes = max(1, int(getattr(params, 'holes', 1)))
        size_ratio = float(getattr(params, 'size_ratio', 0.25))
        probability = float(min(max(probability, 0.0), 1.0))
        size_ratio = float(min(max(size_ratio, 0.0), 1.0))
        return enabled, probability, holes, size_ratio
