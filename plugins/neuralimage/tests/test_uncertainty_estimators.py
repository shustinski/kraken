import numpy as np
import torch
import torch.nn as nn

from neuralimage.uncertainty.estimators import MonteCarloDropoutEstimator, TTAVarianceEstimator


class _DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, inputs):
        return {'mask': self.dropout(self.norm(inputs)), 'confidence': inputs}


def test_mc_dropout_keeps_batch_norm_frozen_and_restores_module_states():
    model = _DropoutModel().train()
    model.norm.eval()
    states = {module: module.training for module in model.modules()}
    inputs = torch.ones((2, 1, 8, 8))
    mean, confidence = MonteCarloDropoutEstimator(samples=4).estimate(
        model, inputs, forward_fn=model
    )
    assert all(module.training == state for module, state in states.items())
    assert mean.shape == confidence.shape == (2, 1, 8, 8)
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))


def test_tta_variance_restores_state_and_is_confident_for_equivariant_model():
    model = nn.Conv2d(1, 1, kernel_size=1).train()
    inputs = torch.rand((1, 1, 9, 11))
    _mean, confidence = TTAVarianceEstimator(use_flips=True).estimate(
        model, inputs, forward_fn=model
    )
    assert model.training
    assert float(confidence.min()) > 0.999
