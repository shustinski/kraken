"""model_train_and_recognition: resolve random artifact prewarm."""
from __future__ import annotations
import os
import math
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import (
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    _RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC,
    _RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.random_artifact_bank import _RandomArtifactBank


class ResolveRandomArtifactPrewarmMixin:
    def _resolved_random_artifact_parameters(self) -> tuple[bool, float, int, float, tuple[str, ...]]:
        params = getattr(self, '_random_artifacts_params', None)
        enabled = bool(getattr(params, 'enabled', False))
        probability = float(getattr(params, 'probability', 1.0))
        count = max(1, int(getattr(params, 'count', 1)))
        size_ratio = float(getattr(params, 'size_ratio', 0.25))
        artifact_types = (
            tuple(params.enabled_types())
            if params is not None and hasattr(params, 'enabled_types')
            else ('dust', 'resist_residue', 'etch_residue', 'particle_cluster', 'flake')
        )
        probability = float(min(max(probability, 0.0), 1.0))
        size_ratio = float(min(max(size_ratio, 0.0), 1.0))
        return enabled, probability, count, size_ratio, artifact_types

    @staticmethod
    def _random_artifact_size_bounds(
        *,
        image_height: int,
        image_width: int,
        size_ratio: float,
    ) -> tuple[int, int, int, int]:
        max_artifact_height = max(1, min(int(image_height), int(round(int(image_height) * float(size_ratio)))))
        max_artifact_width = max(1, min(int(image_width), int(round(int(image_width) * float(size_ratio)))))
        min_artifact_height = 1 if max_artifact_height <= 2 else max(2, int(round(max_artifact_height * 0.35)))
        min_artifact_width = 1 if max_artifact_width <= 2 else max(2, int(round(max_artifact_width * 0.35)))
        min_artifact_height = min(max_artifact_height, min_artifact_height)
        min_artifact_width = min(max_artifact_width, min_artifact_width)
        return (
            int(min_artifact_height),
            int(max_artifact_height),
            int(min_artifact_width),
            int(max_artifact_width),
        )

    def _resolve_random_artifact_prewarm_buckets(
        self,
        *,
        image_height: int,
        image_width: int,
        size_ratio: float,
    ) -> list[tuple[int, int]]:
        min_h, max_h, min_w, max_w = self._random_artifact_size_bounds(
            image_height=image_height,
            image_width=image_width,
            size_ratio=size_ratio,
        )
        levels_h = [min_h, max_h]
        levels_w = [min_w, max_w]
        if max_h > min_h:
            levels_h.extend([
                int(round(min_h + ((max_h - min_h) / 3.0))),
                int(round(min_h + (((max_h - min_h) * 2.0) / 3.0))),
            ])
        if max_w > min_w:
            levels_w.extend([
                int(round(min_w + ((max_w - min_w) / 3.0))),
                int(round(min_w + (((max_w - min_w) * 2.0) / 3.0))),
            ])
        buckets: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for bucket in zip(sorted(set(levels_h)), sorted(set(levels_w))):
            normalized = (
                max(1, int(bucket[0])),
                max(1, int(bucket[1])),
            )
            if normalized in seen:
                continue
            seen.add(normalized)
            buckets.append(normalized)
        return buckets

    def _ensure_random_artifact_bank(
        self,
        *,
        image: torch.Tensor,
        artifact_types: tuple[str, ...],
    ) -> _RandomArtifactBank | None:
        if not artifact_types:
            return None
        if not hasattr(self, '_random_artifact_bank'):
            return None
        channels = int(image.shape[1])
        bank = getattr(self, '_random_artifact_bank', None)
        if bank is not None and bank.matches(channels=channels, artifact_types=artifact_types):
            return bank
        self._stop_random_artifact_bank()
        _enabled, _probability, count, size_ratio, _artifact_types = self._resolved_random_artifact_parameters()
        bank = _RandomArtifactBank(
            channels=channels,
            artifact_types=artifact_types,
            target_per_bucket=max(_RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET, int(count) * 2),
        )
        prewarm_buckets = self._resolve_random_artifact_prewarm_buckets(
            image_height=int(image.shape[2]),
            image_width=int(image.shape[3]),
            size_ratio=size_ratio,
        )
        bank.start(prewarm_buckets=prewarm_buckets)
        bank.wait_until_ready(timeout=_RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC)
        self._random_artifact_bank = bank
        return bank

    def _stop_random_artifact_bank(self) -> None:
        bank = getattr(self, '_random_artifact_bank', None)
        self._random_artifact_bank = None
        if bank is not None:
            bank.stop()

    def _resolved_mixup_parameters(self) -> tuple[bool, float, float]:
        params = getattr(self, '_mixup_params', None)
        enabled = bool(getattr(params, 'enabled', False))
        probability = float(getattr(params, 'probability', 1.0))
        alpha = float(getattr(params, 'alpha', 0.2))
        probability = float(min(max(probability, 0.0), 1.0))
        alpha = float(max(alpha, 0.0))
        return enabled, probability, alpha

    def _has_training_batch_augmentations(self) -> bool:
        mixup_enabled, mixup_probability, mixup_alpha = self._resolved_mixup_parameters()
        if mixup_enabled and mixup_probability > 0.0 and mixup_alpha > 0.0:
            return True
        cutout_enabled, cutout_probability, _cutout_holes, cutout_size_ratio = self._resolved_cutout_parameters()
        if cutout_enabled and cutout_probability > 0.0 and cutout_size_ratio > 0.0:
            return True
        artifacts_enabled, artifacts_probability, _artifacts_count, artifacts_size_ratio, artifact_types = (
            self._resolved_random_artifact_parameters()
        )
        return bool(
            artifacts_enabled
            and artifacts_probability > 0.0
            and artifacts_size_ratio > 0.0
            and artifact_types
        )

    @staticmethod
    def _resolved_ignore_index() -> int:
        raw = str(os.getenv('NEURALIMAGE_IGNORE_INDEX', '-100')).strip()
        try:
            return int(raw)
        except ValueError:
            return -100

    def _reduce_loss_map_per_sample(
        self,
        loss_map: torch.Tensor,
        *,
        loss_mode: str,
        apply_pixel_mining: bool,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_flat = loss_map.view(loss_map.shape[0], -1)
        valid_flat = valid_mask.view(valid_mask.shape[0], -1).to(dtype=torch.bool) if valid_mask is not None else None
        use_pixel_mining = (
            apply_pixel_mining
            and self._hard_pixel_mining_enabled()
            and self._loss_supports_hard_pixel_mining(loss_mode)
        )
        keep_ratio = self._resolved_hard_pixel_keep_ratio()
        reduced_losses: list[torch.Tensor] = []
        for batch_index in range(loss_flat.shape[0]):
            sample_losses = loss_flat[batch_index]
            if valid_flat is not None:
                sample_losses = sample_losses[valid_flat[batch_index]]
            if sample_losses.numel() == 0:
                reduced_losses.append(loss_flat.new_tensor(0.0))
                continue
            if use_pixel_mining and keep_ratio < 1.0:
                keep_count = max(1, int(math.ceil(sample_losses.numel() * keep_ratio)))
                if keep_count < sample_losses.numel():
                    sample_losses = torch.topk(sample_losses, keep_count, largest=True, sorted=False).values
            reduced_losses.append(sample_losses.mean())
        return torch.stack(reduced_losses, dim=0)

    @staticmethod
    def _compute_focal_bce_loss_map(
        outputs: torch.Tensor,
        label: torch.Tensor,
        bce_loss_map: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(outputs)
        pt = (probs * label) + ((1.0 - probs) * (1.0 - label))
        pt = torch.clamp(pt, min=0.0, max=1.0)
        alpha_t = (FOCAL_LOSS_ALPHA * label) + ((1.0 - FOCAL_LOSS_ALPHA) * (1.0 - label))
        focal_factor = torch.pow(1.0 - pt, FOCAL_LOSS_GAMMA)
        focal_map = alpha_t * focal_factor * bce_loss_map
        focal_map = torch.nan_to_num(focal_map, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(focal_map, min=0.0, max=50.0)
