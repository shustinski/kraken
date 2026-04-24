import pytest

torch = pytest.importorskip('torch')

from neuralimage.lib.data_interfaces import CutoutParameters, HardMiningParameters, MixupParameters, RandomArtifactsParameters
from neuralimage.model.NeuralNetwork.model_train_and_recognition import TrainerProcess


class _StubBus:
    def put(self, _item):
        return


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


def test_bce_dice_empty_target_low_foreground_probability_stays_low():
    trainer = _build_trainer('bce')
    trainer._loss_term_weights = {'bce': 0.5, 'dice': 0.5}
    outputs = torch.full((1, 1, 64, 64), -5.0, dtype=torch.float32)
    labels = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    assert float(loss.item()) < 0.05


def test_dice_empty_target_uses_average_foreground_probability_penalty():
    trainer = _build_trainer('dice')
    outputs = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    labels = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    assert float(loss.item()) == pytest.approx(0.5, abs=1e-6)


def test_boundary_empty_target_low_foreground_probability_stays_low():
    trainer = _build_trainer('boundary')
    outputs = torch.full((1, 1, 64, 64), -5.0, dtype=torch.float32)
    labels = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    assert float(loss.item()) < 0.01


def test_focal_tversky_empty_target_low_foreground_probability_stays_low():
    trainer = _build_trainer('focal_tversky')
    outputs = torch.full((1, 1, 64, 64), -5.0, dtype=torch.float32)
    labels = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    loss = trainer._compute_per_sample_loss(outputs, labels, criterion)

    assert float(loss.item()) < 0.01
