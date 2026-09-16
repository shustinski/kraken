from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict
import hashlib
import json
import threading
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

_TARGET_CACHE: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
_TARGET_CACHE_LOCK = threading.Lock()


def _target_cache_key(
    mask: np.ndarray,
    basic: SupervisionTargetConfig,
    geometry: GeometrySupervisionConfig,
    enabled_targets: tuple[str, ...],
) -> str:
    binary = np.ascontiguousarray(np.asarray(mask))
    digest = hashlib.sha256(binary.view(np.uint8)).hexdigest()
    config = json.dumps(
        {'basic': asdict(basic), 'geometry': asdict(geometry), 'targets': enabled_targets},
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(f'{digest}:{binary.shape}:{binary.dtype}:{config}'.encode('utf-8')).hexdigest()


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
    if target_name == 'distance_transform':
        return {'clip': basic_config.distance_clip}
    if target_name == 'thickness':
        return {'max_thickness': basic_config.thickness_max}
    if target_name == 'vertex':
        return {'border_ignore': geometry_config.border_ignore}
    if target_name == 'corner':
        return {'sigma': geometry_config.corner_sigma, 'border_ignore': geometry_config.border_ignore}
    if target_name == 'endpoint':
        return {'border_ignore': geometry_config.border_ignore}
    if target_name == 'junction':
        return {
            'min_degree': geometry_config.junction_min_degree,
            'border_ignore': geometry_config.border_ignore,
        }
    if target_name == 'orientation':
        return {'bins': geometry_config.orientation_bins, 'radius': geometry_config.orientation_radius}
    if target_name in {'tangent', 'curvature'}:
        return {'radius': geometry_config.orientation_radius}
    return {}


def _target_validity(
    target_name: str,
    mask: np.ndarray,
    generated: np.ndarray,
    *,
    basic_config: SupervisionTargetConfig,
    geometry_config: GeometrySupervisionConfig,
) -> np.ndarray:
    binary = (np.asarray(mask) >= 0.5).astype(np.float32)
    if binary.ndim == 3:
        binary = np.squeeze(binary)
    channels = int(generated.shape[-1]) if generated.ndim == 3 else 1
    valid = np.ones((*binary.shape, channels), dtype=np.float32) if channels > 1 else np.ones_like(binary)
    if target_name in {'thickness', 'orientation'}:
        valid = binary[..., None] if channels > 1 else binary
    elif target_name in {'tangent', 'curvature'}:
        skeleton = (generate_skeleton_map(binary) > 0.5).astype(np.float32)
        valid = skeleton[..., None] if channels > 1 else skeleton
    border = (
        geometry_config.border_ignore
        if target_name in {'vertex', 'corner', 'endpoint', 'junction', 'orientation', 'tangent', 'curvature', 'topology'}
        else basic_config.border_ignore
    )
    border = min(max(0, int(border)), max(0, min(binary.shape) // 2))
    if border:
        valid[:border, ...] = valid[-border:, ...] = 0.0
        valid[:, :border, ...] = valid[:, -border:, ...] = 0.0
    return valid.astype(np.float32)


def generate_supervision_targets(
    mask: np.ndarray,
    *,
    basic_config: SupervisionTargetConfig | None = None,
    geometry_config: GeometrySupervisionConfig | None = None,
    enabled_targets: tuple[str, ...] | None = None,
    cache: bool = False,
    cache_size: int = 256,
) -> dict[str, np.ndarray]:
    basic = basic_config or SupervisionTargetConfig()
    geometry = geometry_config or GeometrySupervisionConfig()
    if enabled_targets is None:
        enabled_targets = basic.enabled_basic_targets() + geometry.enabled_geometry_targets()
    cache_key = _target_cache_key(mask, basic, geometry, enabled_targets) if cache else None
    if cache_key is not None:
        with _TARGET_CACHE_LOCK:
            cached = _TARGET_CACHE.get(cache_key)
            if cached is not None:
                _TARGET_CACHE.move_to_end(cache_key)
                return {name: value.copy() for name, value in cached.items()}

    targets: dict[str, np.ndarray] = {'mask': np.asarray(mask, dtype=np.float32)}
    for target_name in enabled_targets:
        generator = TargetGeneratorRegistry.get(target_name)
        if generator is None:
            continue
        kwargs = _generator_kwargs(target_name, basic_config=basic, geometry_config=geometry)
        generated = generator(mask, **kwargs)
        targets[target_name] = np.asarray(generated, dtype=np.float32)
        targets[f'{target_name}__valid'] = _target_validity(
            target_name,
            mask,
            targets[target_name],
            basic_config=basic,
            geometry_config=geometry,
        )
    if cache_key is not None:
        with _TARGET_CACHE_LOCK:
            _TARGET_CACHE[cache_key] = {name: value.copy() for name, value in targets.items()}
            _TARGET_CACHE.move_to_end(cache_key)
            while len(_TARGET_CACHE) > max(1, int(cache_size)):
                _TARGET_CACHE.popitem(last=False)
    return targets


def stack_targets_for_batch(targets: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(array, dtype=np.float32) for name, array in targets.items()}
