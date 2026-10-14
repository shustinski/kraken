"""model_train_and_recognition: compute binary metrics."""
from __future__ import annotations
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from neuralimage.model.NeuralNetwork.dataset import NoCutDataset
from neuralimage.lib.images import _build_enabled_transform_variants


class ComputeBinaryMetricsMixin:
    @staticmethod
    def _compute_binary_metrics(
        *,
        true_positive: float,
        false_positive: float,
        false_negative: float,
        correct: int,
        total: int,
    ) -> dict[str, float]:
        accuracy = (float(correct) / float(total)) if total else 0.0
        iou_denom = true_positive + false_positive + false_negative
        dice_denom = (2.0 * true_positive) + false_positive + false_negative
        if iou_denom == 0.0:
            return {
                'accuracy': float(accuracy),
                'iou': 1.0,
                'dice': 1.0,
                'f1': 1.0,
            }

        iou = true_positive / iou_denom
        dice = (2.0 * true_positive) / dice_denom if dice_denom > 0.0 else 0.0
        precision_denom = true_positive + false_positive
        recall_denom = true_positive + false_negative
        precision = true_positive / precision_denom if precision_denom > 0.0 else 0.0
        recall = true_positive / recall_denom if recall_denom > 0.0 else 0.0
        f1_denom = precision + recall
        f1 = (2.0 * precision * recall) / f1_denom if f1_denom > 0.0 else 0.0
        return {
            'accuracy': float(accuracy),
            'iou': float(iou),
            'dice': float(dice),
            'f1': float(f1),
        }

    @staticmethod
    def _pick_best_validation_threshold(
        threshold_metrics: dict[float, dict[str, float]],
    ) -> tuple[float, dict[str, float]]:
        if not threshold_metrics:
            return 0.5, {'accuracy': 0.0, 'iou': 0.0, 'dice': 0.0, 'f1': 0.0}

        def _sort_key(item: tuple[float, dict[str, float]]) -> tuple[float, float, float]:
            threshold, metrics = item
            return (
                float(metrics.get('dice', 0.0)),
                float(metrics.get('iou', 0.0)),
                -abs(float(threshold) - 0.5),
            )

        best_threshold, best_metrics = max(threshold_metrics.items(), key=_sort_key)
        return float(best_threshold), dict(best_metrics)

    def _resolve_validation_dataset(self) -> Any:
        if self._val_dataloader is None:
            return None
        return getattr(self._val_dataloader, 'dataset', None)

    @staticmethod
    def _unwrap_validation_dataset(dataset: Any) -> Any:
        current = dataset
        seen_ids: set[int] = set()
        while current is not None and id(current) not in seen_ids:
            seen_ids.add(id(current))
            base_dataset = getattr(current, '_base_dataset', None)
            if base_dataset is None:
                return current
            current = base_dataset
        return current

    def _resolve_validation_export_items(self) -> list[dict[str, Any]]:
        dataset = self._resolve_validation_dataset()
        base_dataset = self._unwrap_validation_dataset(dataset)
        if isinstance(base_dataset, NoCutDataset):
            cut_settings = getattr(base_dataset, '_cut_settings', None)
            segment_size = tuple(getattr(cut_settings, 'segment_size', (0, 0)))
            if len(segment_size) != 2:
                return []
            step = int(getattr(cut_settings, 'step', segment_size[0]))
            overlap = max(0, int(segment_size[0]) - step)
            channels = int(getattr(base_dataset, 'colors', 1))
            return [
                {
                    'image_path': Path(image_path),
                    'segment_shape': (channels, int(segment_size[0]), int(segment_size[1])),
                    'overlap': overlap,
                    'use_context_branch': bool(getattr(base_dataset, '_use_context_branch', False)),
                    'use_cross_attention': bool(getattr(base_dataset, '_use_cross_attention', True)),
                    'context_crop_size': getattr(base_dataset, '_context_crop_size', None),
                    'context_input_size': getattr(base_dataset, '_context_input_size', None),
                }
                for image_path, _label_path in getattr(base_dataset, 'samples', [])
            ]

        return []

    @staticmethod
    def _resolve_validation_sample_indices(
        sample_indices: Any,
        fallback_start: int | None = None,
        batch_size: int | None = None,
    ) -> list[int]:
        resolved_fallback_start = 0 if fallback_start is None else int(fallback_start)
        resolved_batch_size = 1 if batch_size is None else max(1, int(batch_size))
        if torch.is_tensor(sample_indices):
            resolved = sample_indices.detach().cpu().tolist()
            if isinstance(resolved, list):
                return [int(index) for index in resolved]
            return [int(resolved)]
        if isinstance(sample_indices, np.ndarray):
            resolved = sample_indices.tolist()
            if isinstance(resolved, list):
                return [int(index) for index in resolved]
            return [int(resolved)]
        if sample_indices is None:
            return list(range(resolved_fallback_start, resolved_fallback_start + resolved_batch_size))
        if isinstance(sample_indices, (int, np.integer)):
            return [int(sample_indices)]
        return [int(index) for index in sample_indices]

    @staticmethod
    def _can_cache_no_cut_validation_export(base_dataset: NoCutDataset) -> bool:
        cut_settings = getattr(base_dataset, '_cut_settings', None)
        if bool(getattr(cut_settings, 'random_crop', False)):
            return False
        if bool(getattr(base_dataset, '_skip_uniform_labels', False)):
            return False
        if bool(getattr(base_dataset, '_rare_patch_oversampling_enabled', False)):
            return False
        return True

    @staticmethod
    def _decode_no_cut_export_item(base_dataset: NoCutDataset, item: int) -> tuple[int, int, int, int]:
        cut_settings = getattr(base_dataset, '_cut_settings', None)
        loc = int(item)
        augmentation_variant = 0
        if bool(getattr(cut_settings, 'additional_augmentation', False)):
            loc, augmentation_variant = divmod(loc, 2)
        scale_variant = 0
        if bool(getattr(cut_settings, 'scale_augmentation', False)):
            loc, scale_variant = divmod(loc, 2)
        transform_variants = _build_enabled_transform_variants(
            enable_rotate_180=bool(getattr(cut_settings, 'vertical_rotation', False)),
            enable_rotate_90=bool(getattr(cut_settings, 'horizontal_rotation', False)),
            enable_flip_x=bool(getattr(cut_settings, 'flip_x', False)),
            enable_flip_y=bool(getattr(cut_settings, 'flip_y', False)),
            square_patch=bool(getattr(base_dataset, '_sample_x', 0) == getattr(base_dataset, '_sample_y', 0)),
        )
        transform_count = max(1, len(transform_variants))
        location, transform_index = divmod(int(loc), transform_count)
        return int(location), int(transform_index), int(scale_variant), int(augmentation_variant)

    @staticmethod
    def _build_no_cut_frame_part_lookup(
        base_dataset: NoCutDataset,
        *,
        frame_index: int,
        parts_count: int,
    ) -> list[int] | None:
        if not bool(getattr(base_dataset, 'shuffle_patches_in_frame', False)):
            return None
        parts = list(range(int(parts_count)))
        randomizer = random.Random(int(getattr(base_dataset, '_frame_seed')(frame_index)))
        randomizer.shuffle(parts)
        return [int(item) for item in parts]
