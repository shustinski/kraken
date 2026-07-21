import numpy as np
import torch

from neuralimage.heads.multi_target import MultiTargetHeadBundle, build_training_output_dict
from neuralimage.losses.composite import compute_auxiliary_head_loss, resolve_auxiliary_head_weights


def test_multi_target_head_bundle_forward():
    heads = MultiTargetHeadBundle(16, supervision_heads=('boundary', 'skeleton', 'sdf'))
    features = torch.randn(2, 16, 32, 32)
    primary, confidence, auxiliary, supervision = heads(features, include_supervision=True)
    assert primary.shape == (2, 1, 32, 32)
    assert confidence.shape == (2, 1, 32, 32)
    assert set(supervision) == {'boundary', 'skeleton', 'sdf'}


def test_build_training_output_dict_includes_supervision():
    primary = torch.randn(1, 1, 8, 8)
    confidence = torch.randn(1, 1, 8, 8)
    supervision = {'boundary': torch.randn(1, 1, 8, 8)}
    payload = build_training_output_dict(primary, confidence=confidence, supervision_outputs=supervision)
    assert 'boundary' in payload


def test_auxiliary_head_loss_combines_enabled_heads():
    outputs = {
        'boundary': torch.randn(2, 1, 8, 8),
        'sdf': torch.randn(2, 1, 8, 8),
    }
    targets = {
        'boundary': torch.rand(2, 1, 8, 8),
        'sdf': torch.rand(2, 1, 8, 8),
    }
    weights = resolve_auxiliary_head_weights(('boundary', 'sdf'))
    loss = compute_auxiliary_head_loss(outputs, targets, head_weights=weights)
    assert loss is not None
    assert loss.shape == (2,)
