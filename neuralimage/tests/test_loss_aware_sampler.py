from conftest import safe_import_or_skip

safe_import_or_skip('torch')
safe_import_or_skip('PIL')

import torch

from model.general_neural_handler import LossAwareSampler


def test_loss_aware_sampler_uses_fallback_above_multinomial_limit(monkeypatch):
    sampler = LossAwareSampler(size=4, replacement=True)
    monkeypatch.setattr(LossAwareSampler, 'MULTINOMIAL_MAX_CATEGORIES', 3)

    def _multinomial_should_not_be_called(*_args, **_kwargs):
        raise AssertionError('torch.multinomial must not be called in fallback mode')

    monkeypatch.setattr(torch, 'multinomial', _multinomial_should_not_be_called)

    indices = list(iter(sampler))
    assert len(indices) == 4
    assert all(0 <= idx < 4 for idx in indices)


def test_loss_aware_sampler_uses_multinomial_within_limit(monkeypatch):
    sampler = LossAwareSampler(size=4, replacement=True)
    monkeypatch.setattr(LossAwareSampler, 'MULTINOMIAL_MAX_CATEGORIES', 10)

    called = {'value': False}

    def _multinomial(_weights, num_samples, replacement):
        called['value'] = True
        assert num_samples == 4
        assert replacement is True
        return torch.tensor([3, 2, 1, 0], dtype=torch.long)

    monkeypatch.setattr(torch, 'multinomial', _multinomial)

    indices = list(iter(sampler))
    assert called['value'] is True
    assert indices == [3, 2, 1, 0]
