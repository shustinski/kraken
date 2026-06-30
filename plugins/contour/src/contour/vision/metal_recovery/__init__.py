from __future__ import annotations

from .detector import MetalDetectionResult, MetalPolygonRecord, MetalRecoveryConfig, detect_metalization
from .settings_bridge import metal_recovery_config_from_settings

__all__ = [
    "MetalDetectionResult",
    "MetalPolygonRecord",
    "MetalRecoveryConfig",
    "detect_metalization",
    "metal_recovery_config_from_settings",
]
