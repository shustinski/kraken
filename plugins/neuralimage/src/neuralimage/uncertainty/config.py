from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class UncertaintyConfig:
    enabled: bool = False
    method: str = 'confidence_head'
    mc_dropout_samples: int = 8
    mc_dropout_rate: float = 0.1
    tta_flips: bool = True
    tta_rotations: bool = False
    export_confidence_map: bool = True
    confidence_loss_weight: float = 0.1

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
        return self.enabled and self.method in {'confidence_head', 'auto', 'combined'}

    def uses_mc_dropout(self) -> bool:
        return self.enabled and self.method in {'mc_dropout', 'combined'}

    def uses_tta_variance(self) -> bool:
        return self.enabled and self.method in {'tta_variance', 'combined'}


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
    )
