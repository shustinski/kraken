from contextlib import nullcontext

import pytest

torch = pytest.importorskip('torch')

from lib.data_interfaces import OptimizerName, OptimizerParameters, SchedulerName, SchedulerParameters, WarmupParameters
import model.NeuralNetwork.model_train_and_recognition as target
from model.NeuralNetwork.model_train_and_recognition import TrainerProcess, _RunContext, _TrainLoopStrides


class _Bus:
    def __init__(self):
        self.messages: list[list[object]] = []

    def put(self, item):
        self.messages.append(item)


class _FakeScheduler:
    def __init__(self):
        self.calls: list[float | None] = []
        self.last_epoch = -1

    def step(self, value=None):
        self.calls.append(value)
        self.last_epoch += 1


def _build_trainer(
    *,
    model: torch.nn.Module | None = None,
    optimizer_params: OptimizerParameters | None = None,
    scheduler_params: SchedulerParameters | None = None,
    warmup_params: WarmupParameters | None = None,
) -> TrainerProcess:
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._model = model or torch.nn.Linear(4, 2)
    trainer._optimizer_params = optimizer_params or OptimizerParameters()
    trainer._scheduler_params = scheduler_params or SchedulerParameters()
    trainer._warmup_params = warmup_params or WarmupParameters()
    trainer._epochs = 5
    trainer._bus = _Bus()
    return trainer


def test_create_lr_scheduler_returns_requested_scheduler_types():
    parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.001)

    trainer = _build_trainer(
        scheduler_params=SchedulerParameters(name=SchedulerName.reduce_on_plateau),
    )
    scheduler, step_mode = trainer._create_lr_scheduler(optimizer, train_steps_per_epoch=4)
    assert type(scheduler).__name__ == 'ReduceLROnPlateau'
    assert step_mode == 'plateau'

    trainer = _build_trainer(
        scheduler_params=SchedulerParameters(name=SchedulerName.cosine_annealing),
    )
    scheduler, step_mode = trainer._create_lr_scheduler(optimizer, train_steps_per_epoch=4)
    assert type(scheduler).__name__ == 'CosineAnnealingLR'
    assert step_mode == 'epoch'

    trainer = _build_trainer(
        scheduler_params=SchedulerParameters(name=SchedulerName.one_cycle),
    )
    scheduler, step_mode = trainer._create_lr_scheduler(optimizer, train_steps_per_epoch=4)
    assert type(scheduler).__name__ == 'OneCycleLR'
    assert step_mode == 'batch'

    trainer = _build_trainer(
        scheduler_params=SchedulerParameters(name=SchedulerName.step_lr),
    )
    scheduler, step_mode = trainer._create_lr_scheduler(optimizer, train_steps_per_epoch=4)
    assert type(scheduler).__name__ == 'StepLR'
    assert step_mode == 'epoch'


def test_create_warmup_scheduler_is_disabled_for_one_cycle():
    parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.001)
    trainer = _build_trainer(
        scheduler_params=SchedulerParameters(name=SchedulerName.one_cycle),
        warmup_params=WarmupParameters(enabled=True, epochs=2, start_factor=0.1),
    )

    scheduler, total_steps = trainer._create_warmup_scheduler(optimizer, train_steps_per_epoch=4)

    assert scheduler is None
    assert total_steps == 0
    assert any('Warmup disabled for OneCycleLR' in str(message[1]) for message in trainer._bus.messages)


def test_step_batch_schedulers_runs_warmup_before_batch_scheduler():
    trainer = _build_trainer()
    warmup_scheduler = _FakeScheduler()
    batch_scheduler = _FakeScheduler()
    run_context = _RunContext(
        bce_criterion=torch.nn.BCEWithLogitsLoss(reduction='none'),
        optimizer=None,
        scaler=None,
        autocast_ctx=lambda: nullcontext(),
        scheduler=batch_scheduler,
        train_size=1,
        train_sampler=None,
        supports_loss_aware_sampling=False,
        strides=_TrainLoopStrides(metric=1, progress=1, log=1, preview=1),
        scheduler_step_mode='batch',
        warmup_scheduler=warmup_scheduler,
        warmup_total_steps=2,
    )

    trainer._step_batch_schedulers(run_context)
    trainer._step_batch_schedulers(run_context)
    trainer._step_batch_schedulers(run_context)

    assert warmup_scheduler.calls == [None, None]
    assert batch_scheduler.calls == [None]


def test_step_epoch_scheduler_uses_validation_loss_then_train_loss_fallback():
    trainer = _build_trainer()
    plateau_scheduler = _FakeScheduler()
    run_context = _RunContext(
        bce_criterion=torch.nn.BCEWithLogitsLoss(reduction='none'),
        optimizer=None,
        scaler=None,
        autocast_ctx=lambda: nullcontext(),
        scheduler=plateau_scheduler,
        train_size=1,
        train_sampler=None,
        supports_loss_aware_sampling=False,
        strides=_TrainLoopStrides(metric=1, progress=1, log=1, preview=1),
        scheduler_step_mode='plateau',
    )

    trainer._step_epoch_scheduler(
        run_context=run_context,
        validation_result={'loss': 0.4},
        train_loss=0.9,
        distributed=False,
        device=torch.device('cpu'),
        is_main_process=True,
    )
    trainer._step_epoch_scheduler(
        run_context=run_context,
        validation_result=None,
        train_loss=0.9,
        distributed=False,
        device=torch.device('cpu'),
        is_main_process=True,
    )

    assert plateau_scheduler.calls == [0.4, 0.9]


def test_create_adamw_optimizer_returns_torch_adamw():
    trainer = _build_trainer(
        model=torch.nn.Linear(4, 2, bias=False),
        optimizer_params=OptimizerParameters(
            name=OptimizerName.adamw,
            learning_rate=3e-4,
            weight_decay=2e-2,
        ),
    )

    optimizer = trainer._create_optimizer()

    assert isinstance(optimizer, torch.optim.AdamW)
