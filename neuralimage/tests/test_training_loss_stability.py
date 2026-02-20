from contextlib import nullcontext

import pytest

torch = pytest.importorskip('torch')

from model.NeuralNetwork.model_train_and_recognition import (
    TrainerProcess,
    _PreparedTrainBatch,
    _RunContext,
    _TrainLoopStrides,
)


class _StubBus:
    def __init__(self):
        self.messages: list[list[object]] = []

    def put(self, item):
        self.messages.append(item)


class _NoOpScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return


class _NaNOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, x):
        shape = (x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.full(shape, float('nan'), device=x.device, dtype=x.dtype) * self.weight.view(1, 1, 1, 1)


def _build_trainer(loss_mode: str) -> TrainerProcess:
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._loss_function = str(loss_mode)
    trainer._dice_loss_weight = 0.5
    trainer._iou_loss_weight = 0.5
    trainer._bus = _StubBus()
    return trainer


@pytest.mark.parametrize('loss_mode', ['bce', 'dice', 'bce_dice', 'iou', 'bce_iou'])
def test_compute_per_sample_loss_sanitizes_non_finite_inputs(loss_mode: str):
    trainer = _build_trainer(loss_mode)
    outputs = torch.tensor(
        [[[[float('nan'), float('inf')], [float('-inf'), 0.0]]]],
        dtype=torch.float32,
    ).repeat(2, 1, 1, 1)
    labels = torch.tensor(
        [[[[0.0, 1.0], [float('nan'), 2.0]]]],
        dtype=torch.float32,
    ).repeat(2, 1, 1, 1)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    per_sample_loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    assert per_sample_loss.shape[0] == 2
    assert bool(torch.isfinite(per_sample_loss).all())


def test_run_train_step_handles_nan_model_outputs():
    trainer = _build_trainer('bce')
    trainer._model = _NaNOutputModel()
    optimizer = torch.optim.SGD(trainer._model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    run_context = _RunContext(
        bce_criterion=criterion,
        optimizer=optimizer,
        scaler=_NoOpScaler(),
        autocast_ctx=lambda: nullcontext(),
        scheduler=None,
        train_size=1,
        train_sampler=None,
        supports_loss_aware_sampling=False,
        strides=_TrainLoopStrides(metric=1, progress=1, log=1, preview=1),
    )

    image = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    label = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    batch = _PreparedTrainBatch(
        data=image,
        target=label,
        sample_indices=None,
        image=image,
        label=label,
        batch_start=0.0,
        data_wait_ms=0.0,
    )

    step_result = trainer._run_train_step(run_context=run_context, batch=batch)

    assert step_result is not None
    assert float(step_result.batch_loss) >= 0.0
    assert float(step_result.batch_loss) < 100.0
