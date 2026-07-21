from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


BASIC_TARGET_NAMES: tuple[str, ...] = (
    'boundary',
    'skeleton',
    'sdf',
    'distance_transform',
    'thickness',
)

GEOMETRY_TARGET_NAMES: tuple[str, ...] = (
    'vertex',
    'corner',
    'endpoint',
    'junction',
    'orientation',
    'tangent',
    'curvature',
    'topology',
)


@dataclass
class SupervisionTargetConfig:
    """Configuration for basic auxiliary supervision targets derived from binary masks."""

    boundary: bool = False
    skeleton: bool = False
    sdf: bool = False
    distance_transform: bool = False
    thickness: bool = False
    boundary_kernel_size: int = 3
    skeleton_iterations: int = 10
    sdf_clip: float = 32.0
    thickness_max: float = 64.0

    def enabled_basic_targets(self) -> tuple[str, ...]:
        enabled: list[str] = []
        for name in BASIC_TARGET_NAMES:
            if bool(getattr(self, name, False)):
                enabled.append(name)
        return tuple(enabled)

    def any_enabled(self) -> bool:
        return bool(self.enabled_basic_targets())


@dataclass
class GeometrySupervisionConfig:
    """Geometry-aware auxiliary targets for polygon-derived IC layout masks."""

    vertex: bool = False
    corner: bool = False
    endpoint: bool = False
    junction: bool = False
    orientation: bool = False
    tangent: bool = False
    curvature: bool = False
    topology: bool = False
    corner_sigma: float = 1.5
    junction_min_degree: int = 3
    orientation_bins: int = 36

    def enabled_geometry_targets(self) -> tuple[str, ...]:
        enabled: list[str] = []
        for name in GEOMETRY_TARGET_NAMES:
            if bool(getattr(self, name, False)):
                enabled.append(name)
        return tuple(enabled)

    def any_enabled(self) -> bool:
        return bool(self.enabled_geometry_targets())


@dataclass
class SupervisionTargetsParameters:
    """Combined supervision target configuration."""

    basic: SupervisionTargetConfig = field(default_factory=SupervisionTargetConfig)
    geometry: GeometrySupervisionConfig = field(default_factory=GeometrySupervisionConfig)
    auxiliary_head_weights: dict[str, float] = field(default_factory=dict)

    def enabled_targets(self) -> tuple[str, ...]:
        return self.basic.enabled_basic_targets() + self.geometry.enabled_geometry_targets()

    def any_enabled(self) -> bool:
        return self.basic.any_enabled() or self.geometry.any_enabled()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def build_supervision_targets_parameters(raw: Mapping[str, Any] | None) -> SupervisionTargetsParameters:
    if not isinstance(raw, Mapping):
        return SupervisionTargetsParameters()

    basic_raw = raw.get('basic', raw)
    geometry_raw = raw.get('geometry', {})
    weights_raw = raw.get('auxiliary_head_weights', raw.get('head_weights', {}))

    basic = SupervisionTargetConfig(
        boundary=_coerce_bool(basic_raw.get('boundary')),
        skeleton=_coerce_bool(basic_raw.get('skeleton')),
        sdf=_coerce_bool(basic_raw.get('sdf')),
        distance_transform=_coerce_bool(basic_raw.get('distance_transform')),
        thickness=_coerce_bool(basic_raw.get('thickness')),
        boundary_kernel_size=int(basic_raw.get('boundary_kernel_size', 3)),
        skeleton_iterations=int(basic_raw.get('skeleton_iterations', 10)),
        sdf_clip=float(basic_raw.get('sdf_clip', 32.0)),
        thickness_max=float(basic_raw.get('thickness_max', 64.0)),
    )
    geometry = GeometrySupervisionConfig(
        vertex=_coerce_bool(geometry_raw.get('vertex')),
        corner=_coerce_bool(geometry_raw.get('corner')),
        endpoint=_coerce_bool(geometry_raw.get('endpoint')),
        junction=_coerce_bool(geometry_raw.get('junction')),
        orientation=_coerce_bool(geometry_raw.get('orientation')),
        tangent=_coerce_bool(geometry_raw.get('tangent')),
        curvature=_coerce_bool(geometry_raw.get('curvature')),
        topology=_coerce_bool(geometry_raw.get('topology')),
        corner_sigma=float(geometry_raw.get('corner_sigma', 1.5)),
        junction_min_degree=int(geometry_raw.get('junction_min_degree', 3)),
        orientation_bins=int(geometry_raw.get('orientation_bins', 36)),
    )
    auxiliary_head_weights: dict[str, float] = {}
    if isinstance(weights_raw, Mapping):
        for key, value in weights_raw.items():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if weight > 0.0:
                auxiliary_head_weights[str(key)] = weight
    return SupervisionTargetsParameters(
        basic=basic,
        geometry=geometry,
        auxiliary_head_weights=auxiliary_head_weights,
    )
