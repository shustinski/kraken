import warnings

import numpy as np
import pytest
import torch

import lib.random_artifacts as random_artifacts
from lib.random_artifacts import (
    ARTIFACT_TYPES,
    _sample_artifact_parameters,
    generate_random_artifact_patch,
)


def test_sample_artifact_parameters_support_every_declared_type():
    np.random.seed(0)

    for artifact_type in ARTIFACT_TYPES:
        params = _sample_artifact_parameters(artifact_type, 64.0)
        assert params['n_blobs'] is not None
        assert params['radius_range'] is not None
        assert params['weights'] is not None
        assert params['gray_level_range'] is not None


def test_sample_artifact_parameters_reject_unknown_type():
    with pytest.raises(ValueError, match='Unsupported artifact type'):
        _sample_artifact_parameters('unknown_artifact', 64.0)


def test_generate_random_artifact_patch_fades_alpha_near_patch_edges():
    np.random.seed(0)
    torch.manual_seed(0)

    _overlay, alpha = generate_random_artifact_patch(
        3,
        64,
        64,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )

    border = torch.cat(
        [
            alpha[:, :4, :].reshape(-1),
            alpha[:, -4:, :].reshape(-1),
            alpha[:, :, :4].reshape(-1),
            alpha[:, :, -4:].reshape(-1),
        ]
    )
    center = alpha[:, 20:44, 20:44]

    assert float(border.max()) < 0.15
    assert float(border.mean()) < float(center.mean()) * 0.25


def test_generate_random_artifact_patch_keeps_overlay_grayscale_for_rgb():
    overlay, _alpha = generate_random_artifact_patch(
        3,
        48,
        48,
        device=torch.device('cpu'),
        dtype=torch.float32,
        seed=123,
    )

    assert torch.allclose(overlay[0], overlay[1])
    assert torch.allclose(overlay[1], overlay[2])


def test_generate_random_artifact_patch_randomizes_gray_level_and_alpha_scale():
    overlay_a, alpha_a = generate_random_artifact_patch(
        3,
        48,
        48,
        device=torch.device('cpu'),
        dtype=torch.float32,
        seed=100,
    )
    overlay_b, alpha_b = generate_random_artifact_patch(
        3,
        48,
        48,
        device=torch.device('cpu'),
        dtype=torch.float32,
        seed=101,
    )

    assert not torch.allclose(overlay_a, overlay_b)
    assert not torch.allclose(alpha_a, alpha_b)


def test_dust_and_flake_artifacts_use_brighter_gray_levels():
    dust = _sample_artifact_parameters('dust', 64.0)
    flake = _sample_artifact_parameters('flake', 64.0)
    particle_cluster = _sample_artifact_parameters('particle_cluster', 64.0)

    assert dust['gray_level_range'][0] > particle_cluster['gray_level_range'][0]
    assert flake['gray_level_range'][0] > particle_cluster['gray_level_range'][1]
    assert dust['intensity_range_255'][0] > particle_cluster['intensity_range_255'][0]
    assert flake['intensity_range_255'][0] > dust['intensity_range_255'][0]


def test_alpha_from_mask_avoids_runtime_warning_for_tiny_decay_scale(monkeypatch):
    original_sampler = random_artifacts._sample_artifact_parameters

    def _sample_with_tiny_decay(defect_type, size):
        params = dict(original_sampler(defect_type, size))
        params['alpha_out_scale'] = (1e-12, 1e-12)
        return params

    monkeypatch.setattr(random_artifacts, '_sample_artifact_parameters', _sample_with_tiny_decay)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('error', RuntimeWarning)
        alpha = random_artifacts.alpha_from_mask(mask, defect_type='dust', seed=7)

    assert alpha.shape == mask.shape
    assert not caught
