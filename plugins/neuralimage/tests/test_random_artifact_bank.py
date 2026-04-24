import threading
import time

import torch

from neuralimage.lib.random_artifacts import generate_random_artifact_patch
from neuralimage.model.NeuralNetwork import model_train_and_recognition as target


def test_generate_random_artifact_patch_is_deterministic_for_same_seed():
    overlay_a, alpha_a = generate_random_artifact_patch(
        channels=1,
        height=16,
        width=20,
        device=torch.device('cpu'),
        dtype=torch.float32,
        artifact_types=('dust', 'flake'),
        seed=12345,
    )
    overlay_b, alpha_b = generate_random_artifact_patch(
        channels=1,
        height=16,
        width=20,
        device=torch.device('cpu'),
        dtype=torch.float32,
        artifact_types=('dust', 'flake'),
        seed=12345,
    )

    assert torch.equal(overlay_a, overlay_b)
    assert torch.equal(alpha_a, alpha_b)


def test_random_artifact_bank_prefills_and_refills_in_background(monkeypatch):
    generation_lock = threading.Lock()
    generation_counter = {'count': 0}

    def _fake_generate(
        channels: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        artifact_types: tuple[str, ...] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del artifact_types, seed
        with generation_lock:
            generation_counter['count'] += 1
            current = generation_counter['count']
        time.sleep(0.01)
        overlay = torch.full((channels, height, width), float(current), dtype=dtype, device=device)
        alpha = torch.full((1, height, width), 0.5, dtype=dtype, device=device)
        return overlay, alpha

    monkeypatch.setattr(target, 'generate_random_artifact_patch', _fake_generate)
    bank = target._RandomArtifactBank(
        channels=1,
        artifact_types=('dust',),
        target_per_bucket=2,
        bucket_granularity=4,
        base_seed=7,
    )

    try:
        bank.start(prewarm_buckets=[(5, 5)])
        deadline = time.perf_counter() + 1.0
        while generation_counter['count'] < 2 and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert generation_counter['count'] >= 2

        overlay_a, alpha_a = bank.acquire(height=5, width=5)
        overlay_b, alpha_b = bank.acquire(height=5, width=5)

        assert overlay_a.shape == (1, 5, 5)
        assert alpha_a.shape == (1, 5, 5)
        assert overlay_b.shape == (1, 5, 5)
        assert alpha_b.shape == (1, 5, 5)

        deadline = time.perf_counter() + 1.0
        while generation_counter['count'] < 4 and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert generation_counter['count'] >= 4
    finally:
        bank.stop()
