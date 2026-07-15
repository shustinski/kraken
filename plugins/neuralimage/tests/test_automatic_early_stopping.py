from __future__ import annotations

import math

import pytest

torch = pytest.importorskip('torch')

from neuralimage.model.NeuralNetwork.early_stopping import (
    CheckpointManager,
    EarlyStoppingConfigCalculator,
    EarlyStoppingPolicy,
    MetricEvaluator,
)


@pytest.mark.parametrize(
    ('train_size', 'expected_max_epochs', 'expected_control_size'),
    (
        (100, 200, 32),
        (5_000, 100, 250),
        (20_000, 60, 1_000),
    ),
)
def test_config_calculator_dataset_sizes(train_size, expected_max_epochs, expected_control_size):
    config = EarlyStoppingConfigCalculator.calculate(train_size=train_size, batches_per_epoch=10)
    assert config.max_epochs == expected_max_epochs
    assert config.control_size == expected_control_size


@pytest.mark.parametrize(
    ('train_size', 'expected_max_epochs'),
    ((999, 200), (1_000, 100), (9_999, 100), (10_000, 60)),
)
def test_config_calculator_max_epoch_boundaries(train_size, expected_max_epochs):
    config = EarlyStoppingConfigCalculator.calculate(train_size=train_size, batches_per_epoch=1)
    assert config.max_epochs == expected_max_epochs


@pytest.mark.parametrize(
    ('batches', 'check_every', 'checks'),
    ((20, 20, 1), (21, 11, 2), (200, 100, 2), (201, 51, 4), (30_000, 5_000, 6)),
)
def test_config_calculator_check_intervals(batches, check_every, checks):
    config = EarlyStoppingConfigCalculator.calculate(train_size=20_000, batches_per_epoch=batches)
    assert config.check_every_batches == check_every
    assert config.checks_per_epoch == checks
    assert config.patience_checks == max(3, 2 * checks)
    assert config.warmup_batches == 3 * batches
    assert config.trend_window == 2 * checks


def test_policy_does_not_stop_while_loss_improves_steadily():
    config = EarlyStoppingConfigCalculator.calculate(train_size=100, batches_per_epoch=1)
    policy = EarlyStoppingPolicy(config)
    decisions = [
        policy.update(loss=1.0 * (0.98**index), global_batch=index + 1)
        for index in range(12)
    ]
    assert not any(decision.should_stop for decision in decisions)
    assert policy.best_actual_loss == pytest.approx(1.0 * (0.98**11))


def test_policy_stops_on_noisy_plateau_after_warmup_and_trend_check():
    config = EarlyStoppingConfigCalculator.calculate(train_size=100, batches_per_epoch=1)
    policy = EarlyStoppingPolicy(config)
    decision = None
    for index, loss in enumerate((1.0, 1.001, 0.999, 1.002, 1.0, 1.001), start=1):
        decision = policy.update(loss=loss, global_batch=index)
        if decision.should_stop:
            break
    assert decision is not None and decision.should_stop
    assert decision.trend_improvement is not None
    assert decision.trend_improvement < 0.01


def test_policy_saves_single_actual_outlier_without_resetting_patience():
    config = EarlyStoppingConfigCalculator.calculate(train_size=100, batches_per_epoch=1)
    policy = EarlyStoppingPolicy(config)
    policy.update(loss=1.0, global_batch=1)
    decision = policy.update(loss=0.997, global_batch=2)
    assert decision.actual_best_improved
    assert not decision.significant_improvement
    assert decision.bad_checks == 1
    assert policy.best_actual_loss == pytest.approx(0.997)
    assert policy.patience_reference_loss == pytest.approx(1.0)


def test_max_epochs_is_a_guard_not_a_policy_decision():
    config = EarlyStoppingConfigCalculator.calculate(train_size=999, batches_per_epoch=2)
    assert config.max_epochs == 200
    policy = EarlyStoppingPolicy(config)
    assert not policy.update(loss=1.0, global_batch=1).should_stop


def test_policy_warmup_and_positive_trend_prevent_early_stop():
    config = EarlyStoppingConfigCalculator.calculate(train_size=100, batches_per_epoch=2)
    policy = EarlyStoppingPolicy(config)
    for global_batch, loss in ((1, 1.0), (2, 1.001), (3, 1.0), (4, 1.001), (5, 1.0)):
        assert not policy.update(loss=loss, global_batch=global_batch).should_stop

    improving_policy = EarlyStoppingPolicy(config)
    decision = None
    for global_batch, loss in enumerate((1.0, 0.994, 0.988, 0.982, 0.976, 0.970), start=1):
        decision = improving_policy.update(loss=loss, global_batch=global_batch)
    assert decision is not None
    assert not decision.should_stop


def test_policy_requires_trend_below_one_percent_after_patience():
    config = EarlyStoppingConfigCalculator.calculate(train_size=100, batches_per_epoch=1)
    policy = EarlyStoppingPolicy(config)
    decisions = [
        policy.update(loss=loss, global_batch=step)
        for step, loss in enumerate((0.98, 1.01, 1.01, 1.01, 0.999), start=1)
    ]
    assert decisions[-1].bad_checks >= config.patience_checks
    assert decisions[-1].trend_improvement is not None
    assert decisions[-1].trend_improvement >= 0.01
    assert not decisions[-1].should_stop

    stopped = policy.update(loss=1.0, global_batch=6)
    assert stopped.trend_improvement is not None and stopped.trend_improvement < 0.01
    assert stopped.should_stop


def test_checkpoint_manager_preserves_and_restores_actual_minimum(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    manager = CheckpointManager(tmp_path / 'training.ckpt')
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert manager.save_best(model=model, loss=1.0, global_batch=10)
    with torch.no_grad():
        model.weight.fill_(2.0)
    assert manager.save_best(model=model, loss=0.999, global_batch=20)
    with torch.no_grad():
        model.weight.fill_(3.0)
    assert not manager.save_best(model=model, loss=1.1, global_batch=30)

    payload = manager.restore_best(model)

    assert payload is not None
    assert payload['global_batch'] == 20
    assert model.weight.item() == pytest.approx(2.0)


def test_metric_evaluator_averages_by_object_not_batch():
    model = torch.nn.Identity()
    dataloader = [
        (torch.zeros(3, 1), torch.tensor([1.0, 2.0, 3.0])),
        (torch.zeros(1, 1), torch.tensor([10.0])),
    ]

    value = MetricEvaluator().evaluate(
        model=model,
        dataloader=dataloader,
        device=torch.device('cpu'),
        batch_adapter=lambda batch, _device: batch,
        forward_fn=lambda _model, inputs: inputs,
        per_sample_loss_fn=lambda _outputs, target: target,
    )

    assert value == pytest.approx(4.0)
    assert math.isfinite(value)
