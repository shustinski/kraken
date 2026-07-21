from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from neuralimage.targets.basic import (
    generate_boundary_map,
    generate_distance_transform,
    generate_local_thickness_map,
    generate_signed_distance_field,
    generate_skeleton_map,
)
from neuralimage.targets.config import GeometrySupervisionConfig, SupervisionTargetConfig
from neuralimage.targets.geometry import (
    generate_corner_heatmap,
    generate_curvature_map,
    generate_endpoint_map,
    generate_junction_map,
    generate_orientation_field,
    generate_tangent_field,
    generate_topology_preservation_map,
    generate_vertex_map,
)


class TargetGeneratorRegistry:
    """Registry of supervision target generators keyed by target name."""

    _generators: dict[str, Callable[..., np.ndarray]] = {}

    @classmethod
    def register(cls, name: str, generator: Callable[..., np.ndarray]) -> None:
        cls._generators[str(name)] = generator

    @classmethod
    def get(cls, name: str) -> Callable[..., np.ndarray] | None:
        return cls._generators.get(str(name))

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._generators))


def _register_builtin_generators() -> None:
    TargetGeneratorRegistry.register('boundary', generate_boundary_map)
    TargetGeneratorRegistry.register('skeleton', generate_skeleton_map)
    TargetGeneratorRegistry.register('sdf', generate_signed_distance_field)
    TargetGeneratorRegistry.register('distance_transform', generate_distance_transform)
    TargetGeneratorRegistry.register('thickness', generate_local_thickness_map)
    TargetGeneratorRegistry.register('vertex', generate_vertex_map)
    TargetGeneratorRegistry.register('corner', generate_corner_heatmap)
    TargetGeneratorRegistry.register('endpoint', generate_endpoint_map)
    TargetGeneratorRegistry.register('junction', generate_junction_map)
    TargetGeneratorRegistry.register('orientation', generate_orientation_field)
    TargetGeneratorRegistry.register('tangent', generate_tangent_field)
    TargetGeneratorRegistry.register('curvature', generate_curvature_map)
    TargetGeneratorRegistry.register('topology', generate_topology_preservation_map)


_register_builtin_generators()


def _generator_kwargs(
    target_name: str,
    *,
    basic_config: SupervisionTargetConfig,
    geometry_config: GeometrySupervisionConfig,
) -> dict[str, Any]:
    if target_name == 'boundary':
        return {'kernel_size': basic_config.boundary_kernel_size}
    if target_name == 'skeleton':
        return {'iterations': basic_config.skeleton_iterations}
    if target_name == 'sdf':
        return {'clip': basic_config.sdf_clip}
    if target_name == 'thickness':
        return {'max_thickness': basic_config.thickness_max}
    if target_name == 'corner':
        return {'sigma': geometry_config.corner_sigma}
    if target_name == 'junction':
        return {'min_degree': geometry_config.junction_min_degree}
    if target_name == 'orientation':
        return {'bins': geometry_config.orientation_bins}
    return {}


def generate_supervision_targets(
    mask: np.ndarray,
    *,
    basic_config: SupervisionTargetConfig | None = None,
    geometry_config: GeometrySupervisionConfig | None = None,
    enabled_targets: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    basic = basic_config or SupervisionTargetConfig()
    geometry = geometry_config or GeometrySupervisionConfig()
    if enabled_targets is None:
        enabled_targets = basic.enabled_basic_targets() + geometry.enabled_geometry_targets()

    targets: dict[str, np.ndarray] = {'mask': np.asarray(mask, dtype=np.float32)}
    for target_name in enabled_targets:
        generator = TargetGeneratorRegistry.get(target_name)
        if generator is None:
            continue
        kwargs = _generator_kwargs(target_name, basic_config=basic, geometry_config=geometry)
        generated = generator(mask, **kwargs)
        targets[target_name] = np.asarray(generated, dtype=np.float32)
    return targets


def stack_targets_for_batch(targets: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(array, dtype=np.float32) for name, array in targets.items()}
