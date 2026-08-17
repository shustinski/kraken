from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralimage.active_learning.config import ActiveLearningConfig, build_active_learning_config
from neuralimage.augmentations.sem_config import SemAugmentationConfig, build_sem_augmentation_config
from neuralimage.preprocessing.config import PreprocessingConfig, build_preprocessing_config
from neuralimage.targets.config import SupervisionTargetsParameters, build_supervision_targets_parameters
from neuralimage.uncertainty.config import UncertaintyConfig, build_uncertainty_config


CONFIG_VERSION = 1


@dataclass(frozen=True)
class HeadsConfig:
    enabled: tuple[str, ...] = ()


@dataclass(frozen=True)
class LossesConfig:
    weighting_strategy: str = 'static'
    mask_weight_floor: float = 0.25


@dataclass(frozen=True)
class HardMiningConfig:
    mode: str = 'off'
    geometry_weight: float = 0.5
    loss_weight: float = 0.5
    exploration_floor: float = 0.1
    ema_alpha: float = 0.1
    score_clip: float = 5.0
    refresh_epochs: int = 1
    offline_manifest: Path | None = None


@dataclass(frozen=True)
class ContextConfig:
    enabled: bool = False
    fusion_type: str = 'concat'
    cross_attention: bool = True
    attention_dim: int = 128
    attention_heads: int = 4
    max_global_tokens: int = 1024


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = True
    full_frame: bool = True
    boundary_tolerance: int = 2
    include_hd95: bool = True
    confidence_bins: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    topology_first: bool = True
    dataset_manifest: Path | None = None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass(frozen=True)
class SemSegmentationConfig:
    version: int = CONFIG_VERSION
    preset: str = 'legacy_v1'
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: SemAugmentationConfig = field(default_factory=SemAugmentationConfig)
    targets: SupervisionTargetsParameters = field(default_factory=SupervisionTargetsParameters)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    losses: LossesConfig = field(default_factory=LossesConfig)
    hard_mining: HardMiningConfig = field(default_factory=HardMiningConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self) -> None:
        if self.version != CONFIG_VERSION:
            raise ValueError(f'Unsupported SEM segmentation config version: {self.version}.')
        enabled_targets = set(self.targets.enabled_targets())
        enabled_heads = set(self.heads.enabled)
        supported_heads = {
            'boundary', 'skeleton', 'sdf', 'distance_transform', 'thickness',
            'vertex', 'corner', 'endpoint', 'junction', 'orientation', 'tangent',
            'curvature', 'topology',
        }
        unknown_heads = sorted(enabled_heads - supported_heads)
        if unknown_heads:
            raise ValueError(f'Unsupported auxiliary heads: {unknown_heads}.')
        if enabled_heads != enabled_targets:
            missing = sorted(enabled_targets - enabled_heads)
            extra = sorted(enabled_heads - enabled_targets)
            raise ValueError(f'Heads and targets must match; missing_heads={missing}, missing_targets={extra}.')
        if self.targets.distance_boundary_weight > 0.0 and 'sdf' not in enabled_targets:
            raise ValueError('distance_boundary_weight requires the SDF target and head.')
        if self.losses.weighting_strategy not in {'static', 'homoscedastic_uncertainty'}:
            raise ValueError('Unsupported loss weighting strategy.')
        if not 0.0 < self.losses.mask_weight_floor <= 1.0:
            raise ValueError('Loss mask_weight_floor must be in (0, 1].')
        if self.hard_mining.mode not in {'off', 'online', 'offline', 'online_and_offline'}:
            raise ValueError('Unsupported hard-mining mode.')
        if not 0.0 <= self.hard_mining.exploration_floor <= 1.0:
            raise ValueError('Hard-mining exploration_floor must be in [0, 1].')
        if self.hard_mining.geometry_weight < 0.0 or self.hard_mining.loss_weight < 0.0:
            raise ValueError('Hard-mining weights cannot be negative.')
        if self.hard_mining.mode != 'off' and self.hard_mining.geometry_weight + self.hard_mining.loss_weight <= 0.0:
            raise ValueError('Enabled hard mining requires a positive geometry or loss weight.')
        if not 0.0 < self.hard_mining.ema_alpha <= 1.0 or self.hard_mining.score_clip <= 0.0:
            raise ValueError('Hard-mining EMA alpha must be in (0, 1] and score_clip must be positive.')
        if self.hard_mining.refresh_epochs <= 0:
            raise ValueError('Hard-mining refresh_epochs must be positive.')
        if self.context.fusion_type not in {'concat', 'add'}:
            raise ValueError('Context fusion_type must be concat or add.')
        if self.context.attention_dim <= 0 or self.context.attention_heads <= 0:
            raise ValueError('Context attention dimensions must be positive.')
        if self.context.attention_dim % self.context.attention_heads != 0:
            raise ValueError('Context attention_dim must be divisible by attention_heads.')
        if self.context.max_global_tokens <= 0:
            raise ValueError('Context max_global_tokens must be positive.')
        if self.validation.boundary_tolerance < 0:
            raise ValueError('Boundary tolerance cannot be negative.')
        if self.validation.confidence_bins < 2:
            raise ValueError('Validation confidence_bins must be at least 2.')
        if self.uncertainty.method not in {
            'confidence_head', 'mc_dropout', 'tta_variance', 'combined', 'auto'
        }:
            raise ValueError('Unsupported uncertainty method.')
        if self.uncertainty.mc_dropout_samples < 2 or not 0.0 <= self.uncertainty.mc_dropout_rate < 1.0:
            raise ValueError('MC Dropout requires at least two samples and a rate in [0, 1).')
        for name in (
            'low_confidence_threshold', 'high_entropy_threshold',
            'instability_threshold', 'disagreement_threshold',
        ):
            if not 0.0 <= float(getattr(self.active_learning, name)) <= 1.0:
                raise ValueError(f'Active Learning {name} must be in [0, 1].')
        if self.active_learning.max_exports_per_run <= 0:
            raise ValueError('Active Learning max_exports_per_run must be positive.')
        basic = self.targets.basic
        if basic.boundary_kernel_size <= 0 or basic.boundary_kernel_size % 2 == 0:
            raise ValueError('Boundary kernel size must be a positive odd integer.')
        if min(basic.sdf_clip, basic.distance_clip, basic.thickness_max) <= 0.0:
            raise ValueError('Distance and thickness scaling values must be positive.')
        if basic.skeleton_iterations < 0 or basic.border_ignore < 0 or basic.cldice_iterations <= 0:
            raise ValueError('Basic target iteration and border settings are invalid.')
        geometry = self.targets.geometry
        if geometry.corner_sigma <= 0.0 or geometry.orientation_radius <= 0 or geometry.border_ignore < 0:
            raise ValueError('Geometry radius, sigma and border settings are invalid.')
        if not self.experiment.seeds:
            raise ValueError('At least one experiment seed is required.')

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _mapping(raw: Any, key: str) -> Mapping[str, Any]:
    value = raw.get(key, {}) if isinstance(raw, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def build_sem_segmentation_config(raw: Mapping[str, Any] | None) -> SemSegmentationConfig:
    if not isinstance(raw, Mapping):
        return SemSegmentationConfig()
    targets = build_supervision_targets_parameters(_mapping(raw, 'targets'))
    heads_raw = _mapping(raw, 'heads')
    head_values = heads_raw.get('enabled', targets.enabled_targets())
    enabled_heads = tuple(str(value) for value in head_values) if isinstance(head_values, (list, tuple)) else ()
    losses_raw = _mapping(raw, 'losses')
    hard_raw = _mapping(raw, 'hard_mining')
    context_raw = _mapping(raw, 'context')
    validation_raw = _mapping(raw, 'validation')
    experiment_raw = _mapping(raw, 'experiment')
    manifest = hard_raw.get('offline_manifest')
    dataset_manifest = experiment_raw.get('dataset_manifest')
    return SemSegmentationConfig(
        version=int(raw.get('version', CONFIG_VERSION)),
        preset=str(raw.get('preset', 'custom')),
        preprocessing=build_preprocessing_config(_mapping(raw, 'preprocessing')),
        augmentation=build_sem_augmentation_config(_mapping(raw, 'augmentation')),
        targets=targets,
        heads=HeadsConfig(enabled=enabled_heads),
        losses=LossesConfig(
            weighting_strategy=str(losses_raw.get('weighting_strategy', 'static')),
            mask_weight_floor=float(losses_raw.get('mask_weight_floor', 0.25)),
        ),
        hard_mining=HardMiningConfig(
            mode=str(hard_raw.get('mode', 'off')),
            geometry_weight=float(hard_raw.get('geometry_weight', 0.5)),
            loss_weight=float(hard_raw.get('loss_weight', 0.5)),
            exploration_floor=float(hard_raw.get('exploration_floor', 0.1)),
            ema_alpha=float(hard_raw.get('ema_alpha', 0.1)),
            score_clip=float(hard_raw.get('score_clip', 5.0)),
            refresh_epochs=int(hard_raw.get('refresh_epochs', 1)),
            offline_manifest=Path(manifest) if manifest else None,
        ),
        context=ContextConfig(
            enabled=bool(context_raw.get('enabled', False)),
            fusion_type=str(context_raw.get('fusion_type', 'concat')),
            cross_attention=bool(context_raw.get('cross_attention', True)),
            attention_dim=int(context_raw.get('attention_dim', 128)),
            attention_heads=int(context_raw.get('attention_heads', 4)),
            max_global_tokens=int(context_raw.get('max_global_tokens', 1024)),
        ),
        uncertainty=build_uncertainty_config(_mapping(raw, 'uncertainty')),
        active_learning=build_active_learning_config(_mapping(raw, 'active_learning')),
        validation=ValidationConfig(
            enabled=bool(validation_raw.get('enabled', True)),
            full_frame=bool(validation_raw.get('full_frame', True)),
            boundary_tolerance=int(validation_raw.get('boundary_tolerance', 2)),
            include_hd95=bool(validation_raw.get('include_hd95', True)),
            confidence_bins=int(validation_raw.get('confidence_bins', 10)),
        ),
        experiment=ExperimentConfig(
            seeds=tuple(int(value) for value in experiment_raw.get('seeds', (17, 29, 43))),
            topology_first=bool(experiment_raw.get('topology_first', True)),
            dataset_manifest=Path(dataset_manifest) if dataset_manifest else None,
        ),
    )


def available_sem_presets() -> tuple[str, ...]:
    return ('legacy_v1', 'sem_topology_experimental_v1')


def get_sem_preset(name: str) -> SemSegmentationConfig:
    normalized = str(name).strip().lower()
    if normalized == 'legacy_v1':
        return SemSegmentationConfig()
    if normalized == 'sem_topology_recommended_v1':
        raise ValueError('The recommended preset is unavailable until three-seed real-SEM ablations pass.')
    if normalized != 'sem_topology_experimental_v1':
        raise KeyError(f'Unknown SEM segmentation preset: {name!r}.')
    return build_sem_segmentation_config(
        {
            'version': CONFIG_VERSION,
            'preset': normalized,
            'preprocessing': {
                'percentile_normalization': True,
                'percentile_low': 0.5,
                'percentile_high': 99.5,
                'scan_line_suppression': True,
                'scan_line_strength': 0.35,
            },
            'augmentation': {'enabled': True, 'plan': 'sem_v2'},
            'targets': {
                'basic': {'boundary': True, 'skeleton': True, 'sdf': True},
                'auxiliary_head_weights': {'boundary': 0.2, 'skeleton': 0.3, 'sdf': 0.15},
                'distance_boundary_weight': 0.1,
                'cache_enabled': True,
            },
            'heads': {'enabled': ['boundary', 'skeleton', 'sdf']},
            'losses': {'weighting_strategy': 'static', 'mask_weight_floor': 0.4},
            'hard_mining': {'mode': 'online', 'exploration_floor': 0.15},
            'uncertainty': {'enabled': True, 'method': 'confidence_head', 'confidence_loss_weight': 0.1},
            'validation': {'enabled': True, 'full_frame': True, 'boundary_tolerance': 2, 'include_hd95': True},
            'experiment': {'seeds': [17, 29, 43], 'topology_first': True},
        }
    )
