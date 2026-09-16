import torch
from types import SimpleNamespace

from neuralimage.losses.composite import AuxiliaryHeadLoss, HomoscedasticLossWeighter
from neuralimage.losses.distance_boundary import compute_distance_boundary_loss
from neuralimage.model.NeuralNetwork.model_train_and_recognition import TrainerProcess
from neuralimage.targets.config import build_supervision_targets_parameters


def test_distance_boundary_prefers_correct_side_of_sdf():
    sdf = torch.tensor([[[[-1.0, -0.5, 0.5, 1.0]]]])
    correct = torch.tensor([[[[-8.0, -8.0, 8.0, 8.0]]]])
    inverted = -correct
    assert compute_distance_boundary_loss(correct, sdf).item() < compute_distance_boundary_loss(inverted, sdf).item()


def test_homoscedastic_loss_weighter_is_trainable_and_keeps_state():
    weighter = HomoscedasticLossWeighter(('mask', 'skeleton'))
    value = weighter({'mask': torch.tensor(1.0), 'skeleton': torch.tensor(2.0)})
    value.backward()
    assert all(parameter.grad is not None for parameter in weighter.parameters())
    assert set(weighter.state_dict()) == {'log_variances.mask', 'log_variances.skeleton'}


def test_topology_loss_penalizes_foreground_background_overlap():
    target = torch.zeros((1, 2, 8, 8))
    target[:, 0, 3, 2:6] = 1.0
    target[:, 1, 5, 2:6] = 1.0
    separated = torch.full_like(target, -5.0)
    separated[target > 0] = 5.0
    overlapping = torch.full_like(target, 5.0)
    assert AuxiliaryHeadLoss.compute_single_head(
        separated, target, head_name='topology'
    ).item() < AuxiliaryHeadLoss.compute_single_head(
        overlapping, target, head_name='topology'
    ).item()


def test_trainer_homoscedastic_weighting_receives_auxiliary_gradients():
    supervision = build_supervision_targets_parameters(
        {'basic': {'boundary': True}, 'auxiliary_head_weights': {'boundary': 0.2}}
    )
    dataset = SimpleNamespace(_supervision_targets=supervision, _uncertainty=None)
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._train_dataloader = SimpleNamespace(dataset=dataset)
    trainer._loss_weighting_strategy = 'homoscedastic_uncertainty'
    trainer._mask_loss_weight_floor = 0.4
    trainer._loss_function = 'bce'
    trainer._loss_term_weights = {}
    trainer._dice_loss_weight = 0.5
    trainer._iou_loss_weight = 0.5
    trainer._hard_mining_params = SimpleNamespace(pixel_enabled=False)
    trainer._prepare_loss_weighter(torch.device('cpu'))

    outputs = {
        'mask': torch.zeros((2, 1, 8, 8), requires_grad=True),
        'boundary': torch.zeros((2, 1, 8, 8), requires_grad=True),
    }
    targets = {
        'mask': torch.zeros((2, 1, 8, 8)),
        'boundary': torch.zeros((2, 1, 8, 8)),
        'boundary__valid': torch.ones((2, 1, 8, 8)),
    }
    loss = trainer._compute_per_sample_loss(
        outputs,
        targets,
        torch.nn.BCEWithLogitsLoss(reduction='none'),
    ).mean()
    loss.backward()

    assert trainer._loss_weighter is not None
    assert set(trainer._loss_weighter.log_variances) == {'mask', 'boundary'}
    assert all(parameter.grad is not None for parameter in trainer._loss_weighter.parameters())
