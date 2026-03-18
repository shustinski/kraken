import numpy as np
import pytest
import torch

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
