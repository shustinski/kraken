from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceTrainingConfig:
    """Training-only confidence-head supervision."""

    enabled: bool = False
    loss_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.loss_weight < 0.0:
            raise ValueError('Confidence loss_weight cannot be negative.')


@dataclass(frozen=True)
class InferenceUncertaintyConfig:
    """Inference-only uncertainty estimation and confidence export."""

    enabled: bool = False
    method: str = 'confidence_head'
    mc_dropout_samples: int = 8
    mc_dropout_rate: float = 0.1
    tta_flips: bool = True
    tta_rotations: bool = False
    export_confidence_map: bool = True

    def __post_init__(self) -> None:
        if self.method not in {'confidence_head', 'mc_dropout', 'tta_variance', 'combined', 'auto'}:
            raise ValueError('Unsupported uncertainty method.')
        if self.mc_dropout_samples < 2:
            raise ValueError('mc_dropout_samples must be at least 2.')
        if not 0.0 <= self.mc_dropout_rate < 1.0:
            raise ValueError('mc_dropout_rate must be in [0, 1).')


@dataclass
class UncertaintyConfig:
    """Runtime compatibility facade for legacy training and inference consumers."""

    enabled: bool = False
    method: str = 'confidence_head'
    mc_dropout_samples: int = 8
    mc_dropout_rate: float = 0.1
    tta_flips: bool = True
    tta_rotations: bool = False
    export_confidence_map: bool = True
    confidence_loss_weight: float = 0.1
    confidence_training_enabled: bool | None = None

    def __post_init__(self) -> None:
        if self.method not in {'confidence_head', 'mc_dropout', 'tta_variance', 'combined', 'auto'}:
            raise ValueError('Unsupported uncertainty method.')
        if self.mc_dropout_samples < 2:
            raise ValueError('mc_dropout_samples must be at least 2.')
        if not 0.0 <= self.mc_dropout_rate < 1.0:
            raise ValueError('mc_dropout_rate must be in [0, 1).')
        if self.confidence_loss_weight < 0.0:
            raise ValueError('confidence_loss_weight cannot be negative.')

    def uses_confidence_head(self) -> bool:
        if self.confidence_training_enabled is not None:
            return self.confidence_training_enabled
        return self.enabled and self.method in {'confidence_head', 'auto', 'combined'}

    def uses_mc_dropout(self) -> bool:
        return self.enabled and self.method in {'mc_dropout', 'combined'}

    def uses_tta_variance(self) -> bool:
        return self.enabled and self.method in {'tta_variance', 'combined'}


def build_confidence_training_config(raw: Mapping[str, Any] | None) -> ConfidenceTrainingConfig:
    if not isinstance(raw, Mapping):
        return ConfidenceTrainingConfig()
    return ConfidenceTrainingConfig(
        enabled=bool(raw.get('enabled', False)),
        loss_weight=float(raw.get('loss_weight', raw.get('confidence_loss_weight', 0.1))),
    )


def build_inference_uncertainty_config(raw: Mapping[str, Any] | None) -> InferenceUncertaintyConfig:
    if not isinstance(raw, Mapping):
        return InferenceUncertaintyConfig()
    return InferenceUncertaintyConfig(
        enabled=bool(raw.get('enabled', False)),
        method=str(raw.get('method', 'confidence_head')).strip().lower(),
        mc_dropout_samples=int(raw.get('mc_dropout_samples', 8)),
        mc_dropout_rate=float(raw.get('mc_dropout_rate', 0.1)),
        tta_flips=bool(raw.get('tta_flips', True)),
        tta_rotations=bool(raw.get('tta_rotations', False)),
        export_confidence_map=bool(raw.get('export_confidence_map', True)),
    )


def combine_uncertainty_config(
    training: ConfidenceTrainingConfig,
    inference: InferenceUncertaintyConfig,
) -> UncertaintyConfig:
    return UncertaintyConfig(
        enabled=inference.enabled,
        method=inference.method,
        mc_dropout_samples=inference.mc_dropout_samples,
        mc_dropout_rate=inference.mc_dropout_rate,
        tta_flips=inference.tta_flips,
        tta_rotations=inference.tta_rotations,
        export_confidence_map=inference.export_confidence_map,
        confidence_loss_weight=training.loss_weight,
        confidence_training_enabled=training.enabled,
    )


def migrate_legacy_uncertainty_config(
    raw: Mapping[str, Any] | None,
) -> tuple[ConfidenceTrainingConfig, InferenceUncertaintyConfig]:
    legacy = build_uncertainty_config(raw)
    training = ConfidenceTrainingConfig(
        enabled=legacy.enabled and legacy.method in {'confidence_head', 'combined', 'auto'},
        loss_weight=legacy.confidence_loss_weight,
    )
    inference = InferenceUncertaintyConfig(
        enabled=legacy.enabled,
        method=legacy.method,
        mc_dropout_samples=legacy.mc_dropout_samples,
        mc_dropout_rate=legacy.mc_dropout_rate,
        tta_flips=legacy.tta_flips,
        tta_rotations=legacy.tta_rotations,
        export_confidence_map=legacy.export_confidence_map,
    )
    return training, inference


def build_uncertainty_config(raw: Mapping[str, Any] | None) -> UncertaintyConfig:
    if not isinstance(raw, Mapping):
        return UncertaintyConfig()
    return UncertaintyConfig(
        enabled=bool(raw.get('enabled', False)),
        method=str(raw.get('method', 'confidence_head')).strip().lower(),
        mc_dropout_samples=int(raw.get('mc_dropout_samples', 8)),
        mc_dropout_rate=float(raw.get('mc_dropout_rate', 0.1)),
        tta_flips=bool(raw.get('tta_flips', True)),
        tta_rotations=bool(raw.get('tta_rotations', False)),
        export_confidence_map=bool(raw.get('export_confidence_map', True)),
        confidence_loss_weight=float(raw.get('confidence_loss_weight', 0.1)),
        confidence_training_enabled=(
            bool(raw.get('confidence_training_enabled'))
            if 'confidence_training_enabled' in raw
            else None
        ),
    )
