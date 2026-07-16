from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from neuralimage.model.general_neural_handler import FrameBatchSampler, _validation_grid_step
from neuralimage.model.NeuralNetwork.model_train_and_recognition import (
    VALIDATION_CHECK_INTERVAL_BATCHES,
    TrainerProcess,
)


def test_frame_batch_sampler_never_mixes_frames():
    sampler = FrameBatchSampler((3, 5), batch_size=2)

    assert list(sampler) == [[0, 1], [2], [3, 4], [5, 6], [7]]
    assert len(sampler) == 5


def test_validation_grid_uses_patch_stride_instead_of_dense_training_shift():
    assert _validation_grid_step((256, 256)) == 256
    assert _validation_grid_step((512, 256)) == 256


def test_validation_check_interval_is_fixed_at_ten_thousand_batches():
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._val_dataloader = object()
    trainer._automatic_early_stopping_config = None

    assert VALIDATION_CHECK_INTERVAL_BATCHES == 10_000
    assert trainer._validation_check_interval() == 10_000


def test_validation_runs_every_ten_thousand_batches_or_at_epoch_end_without_duplicate():
    assert not TrainerProcess._periodic_validation_due(9_999, None, 10_000)
    assert TrainerProcess._periodic_validation_due(10_000, None, 10_000)
    assert not TrainerProcess._periodic_validation_due(10_000, 10_000, 10_000)
    assert TrainerProcess._epoch_end_validation_due(15_000, 10_000, 10_000)
    assert not TrainerProcess._epoch_end_validation_due(10_000, 10_000, 10_000)


def test_validation_averages_patch_losses_per_frame_and_reports_progress():
    class _Dataset:
        def __len__(self):
            return 4

        @staticmethod
        def frame_key(index: int) -> str:
            return 'frame_a' if int(index) < 3 else 'frame_b'

    class _Loader:
        dataset = _Dataset()
        batch_size = 3

        def __len__(self):
            return 2

        def __iter__(self):
            yield (
                torch.tensor([1.0, 1.0, 1.0]).reshape(3, 1, 1, 1),
                torch.ones((3, 1, 1, 1)),
                torch.tensor([0, 1, 2]),
            )
            yield (
                torch.tensor([3.0]).reshape(1, 1, 1, 1),
                torch.ones((1, 1, 1, 1)),
                torch.tensor([3]),
            )

    messages: list[list[object]] = []
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._val_dataloader = _Loader()
    trainer._model = torch.nn.Identity()
    trainer._bus = SimpleNamespace(put=messages.append)
    trainer._save_validation_binary_images = False
    trainer._recommended_inference_threshold = 0.5
    trainer._compute_per_sample_loss = lambda outputs, *_args: outputs.flatten()

    result = trainer._run_validation_epoch(
        epoch=0,
        device=torch.device('cpu'),
        bce_criterion=torch.nn.BCEWithLogitsLoss(reduction='none'),
        autocast_ctx=nullcontext,
    )

    # frame_a mean = 1, frame_b mean = 3; frames have equal final weight.
    assert result is not None
    assert result['loss'] == pytest.approx(2.0)
    assert trainer._model.training is True
    progress = [payload for topic, payload in messages if topic == 'metrics' and payload['type'] == 'validation_progress']
    assert progress == [
        {'type': 'validation_progress', 'current': 0, 'total': 2},
        {'type': 'validation_progress', 'current': 1, 'total': 2},
        {'type': 'validation_progress', 'current': 2, 'total': 2},
    ]
