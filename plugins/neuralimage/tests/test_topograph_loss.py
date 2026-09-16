import torch
from types import SimpleNamespace

from neuralimage.losses.topograph import (
    binary_logits_to_two_channel,
    build_topograph_loss,
    compute_topograph_loss_per_sample,
    extract_critical_region_mask,
)
from neuralimage.losses.topograph_viz import render_critical_regions_overlay
from neuralimage.model.NeuralNetwork.model_train_and_recognition import TrainerProcess


def test_binary_logits_to_two_channel_shapes():
    logits = torch.randn((2, 1, 8, 8))
    labels = torch.randint(0, 2, (2, 1, 8, 8)).float()
    prediction, target = binary_logits_to_two_channel(logits, labels)
    assert prediction.shape == (2, 2, 8, 8)
    assert target.shape == (2, 2, 8, 8)
    assert torch.allclose(target.sum(dim=1), torch.ones((2, 8, 8)))


def test_topograph_loss_backward_flows_to_logits():
    logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
    labels = torch.zeros((1, 1, 16, 16))
    labels[:, :, 4:12, 4:12] = 1.0
    loss_module = build_topograph_loss()
    per_sample = compute_topograph_loss_per_sample(logits, labels, loss_module)
    loss = per_sample.mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_extract_critical_region_mask_detects_bridge_break():
    labels = torch.zeros((1, 1, 16, 16))
    labels[:, :, 7, 4:12] = 1.0
    broken_logits = torch.full((1, 1, 16, 16), 5.0)
    broken_logits[:, :, 7, 7:9] = -5.0
    intact_logits = torch.full((1, 1, 16, 16), 5.0)
    broken_mask = extract_critical_region_mask(broken_logits, labels)
    intact_mask = extract_critical_region_mask(intact_logits, labels)
    assert broken_mask.any()
    assert broken_mask.sum() >= intact_mask.sum()


def test_render_critical_regions_overlay_adds_red_channel():
    pred = torch.full((16, 16), 0.5).numpy()
    mask = torch.zeros((16, 16), dtype=torch.bool)
    mask[4:8, 4:8] = True
    overlay = render_critical_regions_overlay(pred, mask.numpy())
    assert overlay.shape == (16, 16, 3)
    assert overlay[5, 5, 0] > overlay[5, 5, 1]


def test_trainer_skips_topograph_for_multichannel_outputs():
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._topograph_enabled = True
    trainer._topograph_loss_weight = 0.2
    trainer._topograph_binary_skip_logged = False
    trainer._bus = SimpleNamespace(put=lambda *_args, **_kwargs: None)
    outputs = torch.zeros((2, 3, 8, 8))
    labels = torch.zeros((2, 1, 8, 8))
    assert trainer._compute_topograph_component_loss(outputs, labels) is None


def test_disabled_topograph_does_not_change_loss_path_flag():
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._topograph_enabled = False
    trainer._topograph_loss_weight = 0.2
    assert trainer._topograph_is_active() is False
    assert trainer._compute_topograph_component_loss(torch.zeros((1, 1, 8, 8)), torch.zeros((1, 1, 8, 8))) is None
