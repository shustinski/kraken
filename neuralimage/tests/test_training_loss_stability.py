from contextlib import nullcontext

import pytest

torch = pytest.importorskip('torch')

import model.NeuralNetwork.model_train_and_recognition as target
from lib.data_interfaces import CutoutParameters, HardMiningParameters, MixupParameters, RandomArtifactsParameters
from model.NeuralNetwork.model_train_and_recognition import (
    TrainerProcess,
    _PreparedTrainBatch,
    _RunContext,
    _TrainLoopStrides,
    _TrainStepResult,
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


class _DatasetWithTimedSetEpoch:
    def __init__(self):
        self.set_epoch_calls = 0

    def set_epoch(self):
        self.set_epoch_calls += 1
        target.time.perf_counter()


class _LoaderWithDataset:
    def __init__(self, dataset, batches):
        self.dataset = dataset
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


def _build_trainer(loss_mode: str) -> TrainerProcess:
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._loss_function = str(loss_mode)
    trainer._dice_loss_weight = 0.5
    trainer._iou_loss_weight = 0.5
    trainer._hard_mining_params = HardMiningParameters()
    trainer._cutout_params = CutoutParameters()
    trainer._random_artifacts_params = RandomArtifactsParameters()
    trainer._mixup_params = MixupParameters()
    trainer._bus = _StubBus()
    trainer._batch_points_by_epoch = {}
    return trainer


@pytest.mark.parametrize(
    'loss_mode',
    [
        'bce',
        'dice',
        'cldice',
        'bce_dice',
        'iou',
        'bce_iou',
        'focal_bce',
        'focal_dice',
        'focal_iou',
        'boundary',
        'focal_tversky',
        'ce',
        'ce_dice',
    ],
)
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


def test_boundary_loss_penalizes_mismatched_full_masks():
    trainer = _build_trainer('boundary')
    outputs = torch.full((1, 1, 8, 8), -20.0, dtype=torch.float32)
    labels = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    per_sample_loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    assert float(per_sample_loss.item()) > 0.5


def test_compute_per_sample_loss_supports_weighted_sum_of_multiple_losses():
    outputs = torch.tensor(
        [[[[0.4, -0.8], [1.5, -2.0]]]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    bce_trainer = _build_trainer('bce')
    dice_trainer = _build_trainer('dice')
    combo_trainer = _build_trainer('bce')
    combo_trainer._loss_term_weights = {'bce': 0.25, 'dice': 0.75}

    bce_loss = bce_trainer._compute_per_sample_loss(outputs, labels, criterion)
    dice_loss = dice_trainer._compute_per_sample_loss(outputs, labels, criterion)
    combo_loss = combo_trainer._compute_per_sample_loss(outputs, labels, criterion)

    expected = (0.25 * bce_loss) + (0.75 * dice_loss)
    assert float(combo_loss.item()) == pytest.approx(float(expected.item()))


def test_cldice_loss_penalizes_missing_centerline_connectivity():
    trainer = _build_trainer('cldice')
    outputs = torch.full((1, 1, 9, 9), -10.0, dtype=torch.float32)
    outputs[:, :, 4, 1:4] = 10.0
    outputs[:, :, 4, 5:8] = 10.0
    labels = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
    labels[:, :, 4, 1:8] = 1.0
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    per_sample_loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    assert float(per_sample_loss.item()) > 0.05


def test_compute_per_sample_loss_supports_cldice_in_weighted_sum():
    outputs = torch.tensor(
        [[[[2.0, -2.0, -2.0], [2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]],
        dtype=torch.float32,
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    bce_trainer = _build_trainer('bce')
    cldice_trainer = _build_trainer('cldice')
    combo_trainer = _build_trainer('bce')
    combo_trainer._loss_term_weights = {'bce': 0.4, 'cldice': 0.6}

    bce_loss = bce_trainer._compute_per_sample_loss(outputs, labels, criterion)
    cldice_loss = cldice_trainer._compute_per_sample_loss(outputs, labels, criterion)
    combo_loss = combo_trainer._compute_per_sample_loss(outputs, labels, criterion)

    expected = (0.4 * bce_loss) + (0.6 * cldice_loss)
    assert float(combo_loss.item()) == pytest.approx(float(expected.item()))


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
        mixup_pair_indices=None,
        mixup_lambdas=None,
        inputs=image,
        image=image,
        context_image=None,
        label=label,
        batch_start=0.0,
        data_wait_ms=0.0,
    )

    step_result = trainer._run_train_step(run_context=run_context, batch=batch)

    assert step_result is not None
    assert float(step_result.batch_loss) >= 0.0
    assert float(step_result.batch_loss) < 100.0


def test_compute_per_sample_loss_hard_pixel_mining_focuses_on_hardest_pixels():
    trainer = _build_trainer('focal_bce')
    outputs = torch.tensor(
        [[[[8.0, -8.0], [-8.0, -4.0]]]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[[[0.0, 0.0], [0.0, 1.0]]]],
        dtype=torch.float32,
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    baseline = trainer._compute_per_sample_loss(outputs, labels, criterion, apply_pixel_mining=False)

    trainer._hard_mining_params = HardMiningParameters(pixel_enabled=True, pixel_keep_ratio=0.25)
    mined = trainer._compute_per_sample_loss(outputs, labels, criterion, apply_pixel_mining=True)

    assert float(mined.item()) > float(baseline.item())


def test_filter_uniform_batch_samples_skips_binary_uniform_labels_scaled_as_255_over_256():
    trainer = _build_trainer('bce')
    trainer._skip_uniform_labels = True

    image = torch.randn((3, 1, 4, 4), dtype=torch.float32)
    label = torch.zeros((3, 1, 4, 4), dtype=torch.float32)
    label[1, :, :, :] = 255.0 / 256.0
    label[2, :, 0, 0] = 255.0 / 256.0
    sample_indices = torch.tensor([10, 11, 12], dtype=torch.long)

    filtered_image, filtered_label, filtered_indices, skipped_here, has_valid = trainer._filter_uniform_batch_samples(
        image,
        label,
        sample_indices,
    )

    assert has_valid is True
    assert skipped_here == 2
    assert filtered_image.shape[0] == 1
    assert filtered_label.shape[0] == 1
    assert bool(torch.equal(filtered_indices, torch.tensor([12], dtype=torch.long)))


def test_publish_train_batch_runtime_preview_uses_filtered_batch_tensors():
    trainer = _build_trainer('bce')
    trainer._epochs = 1
    captured: dict[str, torch.Tensor] = {}

    def _capture_preview(**kwargs):
        captured['data'] = kwargs['data']
        captured['target'] = kwargs['target']

    trainer._publish_batch_preview = _capture_preview  # type: ignore[method-assign]
    run_context = _RunContext(
        bce_criterion=torch.nn.BCEWithLogitsLoss(reduction='none'),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([0.0]))], lr=0.01),
        scaler=_NoOpScaler(),
        autocast_ctx=lambda: nullcontext(),
        scheduler=None,
        train_size=3,
        train_sampler=None,
        supports_loss_aware_sampling=False,
        strides=_TrainLoopStrides(metric=1000, progress=1000, log=1000, preview=1),
    )
    raw_data = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    raw_target = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    filtered_image = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    filtered_label = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    batch = _PreparedTrainBatch(
        data=raw_data,
        target=raw_target,
        sample_indices=None,
        mixup_pair_indices=None,
        mixup_lambdas=None,
        inputs=filtered_image,
        image=filtered_image,
        context_image=None,
        label=filtered_label,
        batch_start=0.0,
        data_wait_ms=0.0,
    )
    step_result = _TrainStepResult(
        outputs=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        per_sample_loss=torch.ones((1,), dtype=torch.float32),
        batch_loss=0.1,
        batch_samples=1,
        forward_ms=0.1,
        backward_ms=0.1,
        optimizer_ms=0.1,
    )

    trainer._publish_train_batch_runtime(
        epoch=0,
        batch_index=0,
        run_context=run_context,
        batch=batch,
        step_result=step_result,
        batch_total_ms=0.5,
    )

    assert captured['data'] is filtered_image
    assert captured['target'] is filtered_label


def test_run_train_epoch_data_wait_starts_after_set_epoch_and_ends_with_batch_prep(monkeypatch):
    trainer = _build_trainer('bce')
    trainer._skip_uniform_labels = False
    trainer._update_loss_aware_sampling = lambda **kwargs: None
    trainer._publish_train_batch_runtime = lambda **kwargs: None
    trainer._step_training_profiler = lambda *_args, **_kwargs: None

    dataset = _DatasetWithTimedSetEpoch()
    batch = (
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
    )
    trainer._train_dataloader = _LoaderWithDataset(dataset, [batch])
    trainer._run_train_step = lambda **kwargs: _TrainStepResult(
        outputs=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        per_sample_loss=torch.ones((1,), dtype=torch.float32),
        batch_loss=0.1,
        batch_samples=1,
        forward_ms=1.0,
        backward_ms=2.0,
        optimizer_ms=3.0,
    )

    timestamps = iter([4.0, 10.0, 15.0, 18.0, 19.0])
    monkeypatch.setattr(target.time, 'perf_counter', lambda: next(timestamps))

    epoch_stats = trainer._run_train_epoch(
        epoch=0,
        device=torch.device('cpu'),
        run_context=None,  # type: ignore[arg-type]
    )

    assert dataset.set_epoch_calls == 1
    assert epoch_stats.data_wait_ms == pytest.approx(5000.0)
    assert epoch_stats.total_ms == pytest.approx(8000.0)


def test_soft_ce_matches_hard_ce_for_binary_targets():
    trainer = _build_trainer('ce')
    outputs = torch.tensor([[[[2.0, -1.0], [0.5, -0.75]]]], dtype=torch.float32)
    labels = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    per_sample_loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    logits_two_class = torch.cat([-outputs, outputs], dim=1)
    expected = torch.nn.functional.cross_entropy(
        logits_two_class,
        labels[:, 0, :, :].long(),
        reduction='none',
    ).view(1, -1).mean(dim=1)
    assert float(per_sample_loss.item()) == pytest.approx(float(expected.item()))


def test_soft_ce_accepts_soft_targets():
    trainer = _build_trainer('ce_dice')
    outputs = torch.tensor([[[[1.5, -0.5], [0.25, -1.25]]]], dtype=torch.float32)
    labels = torch.tensor([[[[0.8, 0.2], [0.65, 0.35]]]], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    per_sample_loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    assert per_sample_loss.shape == (1,)
    assert bool(torch.isfinite(per_sample_loss).all())
    assert float(per_sample_loss.item()) >= 0.0


def test_prepare_train_batch_applies_random_rectangular_cutout_only_to_image(monkeypatch):
    trainer = _build_trainer('bce')
    trainer._skip_uniform_labels = False
    trainer._cutout_params = CutoutParameters(enabled=True, probability=1.0, holes=1, size_ratio=1.0)

    randint_values = iter((2, 3, 1, 0))

    def _fake_randint(_low, _high, _size, device=None):
        return torch.tensor([next(randint_values)], device=device)

    monkeypatch.setattr(target.np.random, 'random', lambda: 0.0)
    monkeypatch.setattr(target.torch, 'randint', _fake_randint)
    monkeypatch.setattr(
        target.torch,
        'rand',
        lambda _size, device=None, dtype=None: torch.tensor([[[0.2]], [[0.4]], [[0.6]]], device=device, dtype=dtype),
    )

    image = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    label = torch.full((1, 1, 4, 4), 0.5, dtype=torch.float32)
    batch, skipped = trainer._prepare_train_batch(
        batch=(image, label),
        device=torch.device('cpu'),
        data_wait_started_at=0.0,
    )

    assert skipped == 0
    assert batch is not None
    assert bool(torch.equal(batch.label, label))
    expected = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    expected[:, 0:1, 1:3, 0:3] = 0.2
    expected[:, 1:2, 1:3, 0:3] = 0.4
    expected[:, 2:3, 1:3, 0:3] = 0.6
    assert bool(torch.equal(batch.image, expected))


def test_prepare_train_batch_applies_random_artifacts_only_to_image(monkeypatch):
    trainer = _build_trainer('bce')
    trainer._skip_uniform_labels = False
    trainer._random_artifacts_params = RandomArtifactsParameters(
        enabled=True,
        probability=1.0,
        count=1,
        size_ratio=1.0,
    )

    randint_values = iter((2, 3, 1, 0))

    def _fake_randint(_low, _high, _size, device=None):
        return torch.tensor([next(randint_values)], device=device)

    def _fake_generate_random_artifact_patch(channels, height, width, *, device, dtype, artifact_types=None):
        assert channels == 3
        assert height == 2
        assert width == 3
        assert artifact_types
        overlay = torch.tensor(
            [
                [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                [[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]],
                [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6]],
            ],
            device=device,
            dtype=dtype,
        )
        alpha = torch.full((1, 2, 3), 0.5, device=device, dtype=dtype)
        return overlay, alpha

    monkeypatch.setattr(target.np.random, 'random', lambda: 0.0)
    monkeypatch.setattr(target.torch, 'randint', _fake_randint)
    monkeypatch.setattr(target, 'generate_random_artifact_patch', _fake_generate_random_artifact_patch)

    image = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    label = torch.full((1, 1, 4, 4), 0.5, dtype=torch.float32)
    batch, skipped = trainer._prepare_train_batch(
        batch=(image, label),
        device=torch.device('cpu'),
        data_wait_started_at=0.0,
    )

    assert skipped == 0
    assert batch is not None
    assert bool(torch.equal(batch.label, label))
    expected = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    expected[:, 0:1, 1:3, 0:3] = 0.1
    expected[:, 1:2, 1:3, 0:3] = 0.2
    expected[:, 2:3, 1:3, 0:3] = 0.3
    assert bool(torch.equal(batch.image, expected))


def test_prepare_train_batch_applies_mixup_to_images_labels_and_indices(monkeypatch):
    trainer = _build_trainer('bce')
    trainer._skip_uniform_labels = False
    trainer._mixup_params = MixupParameters(enabled=True, probability=1.0, alpha=0.2)

    monkeypatch.setattr(target.np.random, 'beta', lambda _a, _b: 0.75)
    monkeypatch.setattr(
        target.torch,
        'randperm',
        lambda n, device=None: torch.arange(n, device=device),
    )

    image = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    label = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    sample_indices = torch.tensor([10, 20], dtype=torch.long)

    batch, skipped = trainer._prepare_train_batch(
        batch=(image, label, sample_indices),
        device=torch.device('cpu'),
        data_wait_started_at=0.0,
    )

    assert skipped == 0
    assert batch is not None
    assert bool(torch.equal(batch.sample_indices, sample_indices))
    assert bool(torch.equal(batch.mixup_pair_indices, torch.tensor([20, 10], dtype=torch.long)))
    assert batch.mixup_lambdas is not None
    assert bool(torch.allclose(batch.mixup_lambdas, torch.tensor([0.75, 0.75], dtype=torch.float32)))
    assert bool(torch.allclose(batch.image[0], torch.full((1, 2, 2), 0.75)))
    assert bool(torch.allclose(batch.image[1], torch.full((1, 2, 2), 0.25)))
    assert bool(torch.allclose(batch.label[0], torch.full((1, 2, 2), 0.75)))
    assert bool(torch.allclose(batch.label[1], torch.full((1, 2, 2), 0.25)))


def test_prepare_train_batch_skips_mixup_for_single_sample():
    trainer = _build_trainer('bce')
    trainer._skip_uniform_labels = False
    trainer._mixup_params = MixupParameters(enabled=True, probability=1.0, alpha=0.2)

    image = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    label = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    sample_indices = torch.tensor([7], dtype=torch.long)

    batch, skipped = trainer._prepare_train_batch(
        batch=(image, label, sample_indices),
        device=torch.device('cpu'),
        data_wait_started_at=0.0,
    )

    assert skipped == 0
    assert batch is not None
    assert batch.mixup_pair_indices is None
    assert batch.mixup_lambdas is None
    assert bool(torch.equal(batch.image, image))
    assert bool(torch.equal(batch.label, label))


def test_update_loss_aware_sampling_distributes_mixup_loss_to_both_sources():
    class _Sampler:
        def __init__(self):
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def update_batch_losses(self, sample_indices: torch.Tensor, sample_losses: torch.Tensor) -> None:
            self.calls.append((sample_indices.clone(), sample_losses.clone()))

    trainer = _build_trainer('bce')
    sampler = _Sampler()
    run_context = _RunContext(
        bce_criterion=torch.nn.BCEWithLogitsLoss(reduction='none'),
        optimizer=None,
        scaler=None,
        autocast_ctx=lambda: nullcontext(),
        scheduler=None,
        train_size=2,
        train_sampler=sampler,
        supports_loss_aware_sampling=True,
        strides=_TrainLoopStrides(metric=1, progress=1, log=1, preview=1),
    )

    trainer._update_loss_aware_sampling(
        run_context=run_context,
        sample_indices=torch.tensor([1, 2], dtype=torch.long),
        per_sample_loss=torch.tensor([2.0, 4.0], dtype=torch.float32),
        mixup_pair_indices=torch.tensor([3, 4], dtype=torch.long),
        mixup_lambdas=torch.tensor([0.75, 0.25], dtype=torch.float32),
    )

    assert len(sampler.calls) == 2
    assert bool(torch.equal(sampler.calls[0][0], torch.tensor([1, 2], dtype=torch.long)))
    assert bool(torch.allclose(sampler.calls[0][1], torch.tensor([1.5, 1.0], dtype=torch.float32)))
    assert bool(torch.equal(sampler.calls[1][0], torch.tensor([3, 4], dtype=torch.long)))
    assert bool(torch.allclose(sampler.calls[1][1], torch.tensor([0.5, 3.0], dtype=torch.float32)))
