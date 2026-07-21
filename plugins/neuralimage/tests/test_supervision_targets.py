import numpy as np

from neuralimage.targets.basic import (
    generate_boundary_map,
    generate_distance_transform,
    generate_local_thickness_map,
    generate_signed_distance_field,
    generate_skeleton_map,
)
from neuralimage.targets.registry import generate_supervision_targets
from neuralimage.targets.config import SupervisionTargetConfig, GeometrySupervisionConfig


def _square_mask(size: int = 32) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    mask[8:24, 8:24] = 1.0
    return mask


def test_boundary_map_nonzero_on_edges():
    boundary = generate_boundary_map(_square_mask())
    assert boundary.sum() > 0.0
    assert boundary.max() <= 1.0


def test_skeleton_map_inside_mask():
    skeleton = generate_skeleton_map(_square_mask())
    assert skeleton.shape == (32, 32)
    assert skeleton.sum() > 0.0


def test_signed_distance_field_signs():
    sdf = generate_signed_distance_field(_square_mask())
    assert sdf.min() < 0.0
    assert sdf.max() > 0.0


def test_distance_transform_normalized():
    distance = generate_distance_transform(_square_mask())
    assert distance.max() <= 1.0 + 1e-6
    assert distance[16, 16] > distance[0, 0]


def test_thickness_map_peak_on_centerline():
    thickness = generate_local_thickness_map(_square_mask())
    assert thickness[16, 16] >= thickness[0, 0]


def test_generate_supervision_targets_respects_config():
    config = SupervisionTargetConfig(boundary=True, skeleton=True, sdf=True)
    targets = generate_supervision_targets(_square_mask(), basic_config=config)
    assert set(targets) >= {'mask', 'boundary', 'skeleton', 'sdf'}


def test_geometry_targets_optional():
    geometry = GeometrySupervisionConfig(corner=True, junction=True)
    targets = generate_supervision_targets(
        _square_mask(),
        geometry_config=geometry,
        enabled_targets=geometry.enabled_geometry_targets(),
    )
    assert 'corner' in targets
    assert 'junction' in targets
