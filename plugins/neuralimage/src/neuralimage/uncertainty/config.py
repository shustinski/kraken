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

    def uses_confidence_head(self) -> bool:
        return self.enabled and self.method in {'confidence_head', 'auto'}

    def uses_mc_dropout(self) -> bool:
        return self.enabled and self.method == 'mc_dropout'

    def uses_tta_variance(self) -> bool:
        return self.enabled and self.method == 'tta_variance'


def build_uncertainty_config(raw: Mapping[str, Any] | None) -> UncertaintyConfig:
    if not isinstance(raw, Mapping):
        return UncertaintyConfig()
    return UncertaintyConfig(
        enabled=bool(raw.get('enabled', False)),
        method=str(raw.get('method', 'confidence_head')),
        mc_dropout_samples=int(raw.get('mc_dropout_samples', 8)),
        mc_dropout_rate=float(raw.get('mc_dropout_rate', 0.1)),
        tta_flips=bool(raw.get('tta_flips', True)),
        tta_rotations=bool(raw.get('tta_rotations', False)),
        export_confidence_map=bool(raw.get('export_confidence_map', True)),
    )
