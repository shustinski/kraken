from __future__ import annotations

from .detector import MetalDetectionResult, MetalPolygonRecord, MetalRecoveryConfig, detect_metalization
from .settings_bridge import metal_recovery_config_from_settings
from .strategy_contracts import (
    StrategyConfigurationError,
    StrategySegmentation,
    StrategyUnavailableError,
)
from .strategy_registry import (
    IMPLEMENTED_NEW_STRATEGIES,
    MetalStrategyConfigs,
    ParameterSpec,
    StrategySpec,
    strategy_spec,
    visible_strategy_specs,
)

__all__ = [
    "IMPLEMENTED_NEW_STRATEGIES",
    "MetalDetectionResult",
    "MetalPolygonRecord",
    "MetalRecoveryConfig",
    "MetalStrategyConfigs",
    "ParameterSpec",
    "StrategyConfigurationError",
    "StrategySegmentation",
    "StrategySpec",
    "StrategyUnavailableError",
    "detect_metalization",
    "metal_recovery_config_from_settings",
    "strategy_spec",
    "visible_strategy_specs",
]
