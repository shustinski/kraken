import numpy as np
from skimage.morphology import skeletonize, thin

from neuralimage.targets.basic import (
    generate_boundary_map,
    generate_distance_transform,
    generate_local_thickness_map,
    generate_signed_distance_field,
    generate_skeleton_map,
)
from neuralimage.targets.registry import generate_supervision_targets
from neuralimage.targets.config import SupervisionTargetConfig, GeometrySupervisionConfig
from neuralimage.targets.geometry import generate_curvature_map


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


def test_skeleton_map_delegates_to_library_zhang_and_bounded_thinning():
    mask = _square_mask()
    assert np.array_equal(
        generate_skeleton_map(mask).astype(bool),
        skeletonize(mask.astype(bool), method='zhang'),
    )
    assert np.array_equal(
        generate_skeleton_map(mask, iterations=2).astype(bool),
        thin(mask.astype(bool), max_num_iter=2),
    )


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


def test_thinning_preserves_one_pixel_wire_and_disconnected_components():
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[8, 5:20] = 1.0
    mask[28, 20:35] = 1.0
    skeleton = generate_skeleton_map(mask)
    assert np.array_equal(skeleton, mask)


def test_distance_scaling_is_patch_size_independent():
    small = np.zeros((32, 32), dtype=np.float32)
    large = np.zeros((64, 64), dtype=np.float32)
    small[12:20, 4:28] = 1.0
    large[28:36, 12:52] = 1.0
    assert np.isclose(
        generate_distance_transform(small, clip=32.0).max(),
        generate_distance_transform(large, clip=32.0).max(),
    )


def test_geometry_validity_ignores_crop_edges_and_orientation_is_axial():
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[14:18, :] = 1.0
    geometry = GeometrySupervisionConfig(endpoint=True, orientation=True, border_ignore=3)
    targets = generate_supervision_targets(mask, geometry_config=geometry)
    assert targets['endpoint'].sum() == 0.0
    assert not targets['orientation__valid'][:, :3].any()
    assert np.allclose(targets['orientation'][16, 16], (1.0, 0.0), atol=0.15)


def test_target_cache_returns_independent_arrays():
    config = SupervisionTargetConfig(skeleton=True)
    first = generate_supervision_targets(_square_mask(), basic_config=config, cache=True, cache_size=2)
    first['skeleton'][:] = 0.0
    second = generate_supervision_targets(_square_mask(), basic_config=config, cache=True, cache_size=2)
    assert second['skeleton'].sum() > 0.0


def test_curvature_is_measured_along_traced_skeleton_paths():
    straight = np.zeros((32, 32), dtype=np.float32)
    straight[16, 5:27] = 1.0
    corner = np.zeros((32, 32), dtype=np.float32)
    corner[8:17, 16] = 1.0
    corner[16, 16:27] = 1.0

    assert float(generate_curvature_map(straight, radius=4).max()) == 0.0
    assert float(generate_curvature_map(corner, radius=4).max()) > 0.1
