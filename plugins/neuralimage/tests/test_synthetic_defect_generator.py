import pytest
import torch
import numpy as np

pytest.importorskip('PyQt6')

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services.workflow_mapper import build_workflow_parameters
from neuralimage.lib.data_interfaces import SampleCutMode, SampleGenerationSettings, SamplePrepareSettings, TrainingParameters
from neuralimage.model.NeuralNetwork.dataset import SyntheticDefectDataset
from tests.helpers import make_test_dir


def test_build_workflow_parameters_maps_synthetic_defect_generator():
    root = make_test_dir('workflow_synthetic_defect_generator')
    main = MainWindowState(
        work_mode='train_only',
        sample_folder=str(root / 'images'),
        label_folder=str(root / 'labels'),
        result_folder=str(root / 'result'),
        epochs=1,
    )
    settings = SettingsState(
        synthetic_defect_generator={
            'enabled': True,
            'epoch_size_factor': 1.5,
            'trace_count_range': [5, 7],
            'segment_count_range': [3, 5],
            'trace_half_width_range': [2, 3],
            'background_noise_sigma_range': [0.01, 0.03],
            'trace_noise_sigma_range': [0.02, 0.04],
            'defects': {
                'enabled': True,
                'defect_probability': 0.9,
                'min_defects': 2,
                'max_defects': 4,
                'defect_probabilities': {
                    'break': 1.0,
                    'short': 0.0,
                },
            },
        }
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.synthetic_defect_generator.enabled is True
    assert training.synthetic_defect_generator.epoch_size_factor == pytest.approx(1.5)
    assert training.synthetic_defect_generator.trace_count_range == (5, 7)
    assert training.synthetic_defect_generator.segment_count_range == (3, 5)
    assert training.synthetic_defect_generator.trace_half_width_range == (2, 3)
    assert training.synthetic_defect_generator.background_noise_sigma_range == pytest.approx((0.01, 0.03))
    assert training.synthetic_defect_generator.trace_noise_sigma_range == pytest.approx((0.02, 0.04))
    assert training.synthetic_defect_generator.defects.defect_probability == pytest.approx(0.9)
    assert training.synthetic_defect_generator.defects.min_defects == 2
    assert training.synthetic_defect_generator.defects.max_defects == 4
    assert training.synthetic_defect_generator.defects.defect_probabilities['break'] == pytest.approx(1.0)
    assert training.synthetic_defect_generator.defects.defect_probabilities['short'] == pytest.approx(0.0)
    assert training.pcb_defects.defect_probability == pytest.approx(0.9)


def test_synthetic_defect_dataset_returns_deterministic_pair_for_same_epoch():
    root = make_test_dir('synthetic_defect_dataset')
    settings = TrainingParameters(
        image_path=root / 'images',
        label_path=root / 'labels',
        shuffle=True,
        validation=False,
        validation_percent=0,
        batch_size=2,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=SampleGenerationSettings(
            step=64,
            segment_size=(96, 80),
            vertical_rotation=False,
            horizontal_rotation=False,
            channels=1,
        ),
        prepare=SamplePrepareSettings(),
        synthetic_defect_generator={
            'enabled': True,
            'image_size_xy': [96, 80],
            'trace_count': 6,
            'segment_count': 4,
            'trace_half_width': 2,
            'defects': {
                'enabled': True,
                'defect_probability': 1.0,
                'min_defects': 1,
                'max_defects': 1,
            },
        },
    )
    dataset = SyntheticDefectDataset(4, settings, apply_train_only_transforms=True)

    image_a, label_a = dataset[0]
    image_b, label_b = dataset[0]

    assert torch.equal(image_a, image_b)
    assert torch.equal(label_a, label_b)
    assert tuple(image_a.shape) == (1, 80, 96)
    assert tuple(label_a.shape) == (1, 80, 96)
    assert float(label_a.sum().item()) > 0.0


def test_synthetic_defect_dataset_generates_full_frames_then_cuts_into_patches():
    root = make_test_dir('synthetic_defect_dataset_frame_cutting')
    settings = TrainingParameters(
        image_path=root / 'images',
        label_path=root / 'labels',
        shuffle=True,
        validation=False,
        validation_percent=0,
        batch_size=2,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=SampleGenerationSettings(
            step=64,
            segment_size=(256, 256),
            vertical_rotation=False,
            horizontal_rotation=False,
            channels=1,
        ),
        prepare=SamplePrepareSettings(),
        synthetic_defect_generator={
            'enabled': True,
            'epoch_size_factor': 1.0,
            'image_size_xy': [1024, 1024],
            'trace_count_range': [6, 6],
            'segment_count_range': [4, 4],
            'trace_half_width_range': [2, 2],
        },
    )
    dataset = SyntheticDefectDataset(3, settings, apply_train_only_transforms=False)

    assert len(dataset) == 3 * 169
    assert dataset.describe_sample(0).startswith('synthetic_frame_000000__part_')
    assert dataset.describe_sample(168).startswith('synthetic_frame_000000__part_')
    assert dataset.describe_sample(169).startswith('synthetic_frame_000001__part_')

    image_patch, label_patch = dataset[0]

    assert tuple(image_patch.shape) == (1, 256, 256)
    assert tuple(label_patch.shape) == (1, 256, 256)
    assert float(label_patch.sum().item()) > 0.0


def test_synthetic_defect_dataset_skip_uniform_labels_filters_patches_before_batching(monkeypatch):
    root = make_test_dir('synthetic_defect_dataset_skip_uniform_labels')
    settings = TrainingParameters(
        image_path=root / 'images',
        label_path=root / 'labels',
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=2,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=SampleGenerationSettings(
            step=2,
            segment_size=(2, 2),
            vertical_rotation=False,
            horizontal_rotation=False,
            channels=1,
            shuffle_patches_in_frame=False,
        ),
        prepare=SamplePrepareSettings(),
        skip_uniform_labels=True,
        synthetic_defect_generator={
            'enabled': True,
            'image_size_xy': [4, 4],
            'trace_count_range': [1, 1],
            'segment_count_range': [1, 1],
            'trace_half_width_range': [1, 1],
            'defects': {'enabled': False},
        },
    )

    def _fake_generate(self, *, size_hw, channels, seed):
        image = np.arange(16, dtype=np.float32).reshape(1, 4, 4) / 15.0
        label = np.zeros((1, 4, 4), dtype=np.float32)
        label[:, 2:, 2:] = 1.0
        label[:, 0:2, 2:] = 1.0
        label[:, 2, 1] = 1.0
        return image, label

    monkeypatch.setattr(
        'model.NeuralNetwork.dataset.SyntheticTopologyGenerator.generate',
        _fake_generate,
    )

    dataset = SyntheticDefectDataset(1, settings, apply_train_only_transforms=False)

    assert len(dataset) == 1
    image_patch, label_patch = dataset[0]
    assert tuple(image_patch.shape) == (1, 2, 2)
    assert tuple(label_patch.shape) == (1, 2, 2)
    assert float(label_patch.sum().item()) > 0.0
    assert float((label_patch <= 0.5).sum().item()) > 0.0


def test_synthetic_defect_dataset_keeps_label_unchanged_when_defects_are_enabled():
    root = make_test_dir('synthetic_defect_dataset_image_only_defects')
    settings = TrainingParameters(
        image_path=root / 'images',
        label_path=root / 'labels',
        shuffle=True,
        validation=False,
        validation_percent=0,
        batch_size=2,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=SampleGenerationSettings(
            step=64,
            segment_size=(96, 80),
            vertical_rotation=False,
            horizontal_rotation=False,
            channels=1,
        ),
        prepare=SamplePrepareSettings(),
        synthetic_defect_generator={
            'enabled': True,
            'image_size_xy': [96, 80],
            'trace_count_range': [6, 6],
            'segment_count_range': [4, 4],
            'trace_half_width_range': [2, 2],
            'defects': {
                'enabled': True,
                'defect_probability': 1.0,
                'min_defects': 1,
                'max_defects': 1,
            },
        },
    )
    clean_dataset = SyntheticDefectDataset(2, settings, apply_train_only_transforms=False)
    defect_dataset = SyntheticDefectDataset(2, settings, apply_train_only_transforms=True)

    clean_image, clean_label = clean_dataset[0]
    defect_image, defect_label = defect_dataset[0]

    assert torch.equal(clean_label, defect_label)
    assert not torch.equal(clean_image, defect_image)


def test_synthetic_defect_dataset_supports_context_branch():
    root = make_test_dir('synthetic_defect_dataset_context')
    settings = TrainingParameters(
        image_path=root / 'images',
        label_path=root / 'labels',
        shuffle=True,
        validation=False,
        validation_percent=0,
        batch_size=2,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=SampleGenerationSettings(
            step=64,
            segment_size=(96, 96),
            vertical_rotation=False,
            horizontal_rotation=False,
            channels=1,
        ),
        prepare=SamplePrepareSettings(),
        use_context_branch=True,
        local_crop_size=(96, 96),
        context_crop_size=(160, 160),
        context_input_size=(96, 96),
        synthetic_defect_generator={'enabled': True},
    )
    dataset = SyntheticDefectDataset(2, settings, apply_train_only_transforms=True)

    inputs, label = dataset[0]

    assert isinstance(inputs, dict)
    assert tuple(inputs['local_image'].shape) == (1, 96, 96)
    assert tuple(inputs['context_image'].shape) == (1, 96, 96)
    assert tuple(label.shape) == (1, 96, 96)


def test_synthetic_defect_dataset_supports_ic_domain():
    root = make_test_dir('synthetic_defect_dataset_ic_domain')
    settings = TrainingParameters(
        image_path=root / 'images',
        label_path=root / 'labels',
        shuffle=True,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=SampleGenerationSettings(
            step=64,
            segment_size=(96, 96),
            vertical_rotation=False,
            horizontal_rotation=False,
            channels=1,
        ),
        prepare=SamplePrepareSettings(),
        synthetic_defect_generator={
            'enabled': True,
            'topology_domain': 'ic',
            'topology_family': 'ic_cell_array',
            'image_size_xy': [96, 96],
            'trace_count_range': [8, 8],
            'segment_count_range': [2, 3],
            'trace_half_width_range': [1, 2],
            'ic_defects': {
                'enabled': True,
                'defect_probability': 1.0,
                'min_defects': 1,
                'max_defects': 1,
                'defect_probabilities': {
                    'line_break': 1.0,
                    'bridge': 0.0,
                    'necking': 0.0,
                    'missing_metal': 0.0,
                    'spur': 0.0,
                    'pinhole': 0.0,
                    'via_open': 0.0,
                    'line_shift': 0.0,
                },
            },
        },
    )
    clean_dataset = SyntheticDefectDataset(2, settings, apply_train_only_transforms=False)
    defect_dataset = SyntheticDefectDataset(2, settings, apply_train_only_transforms=True)

    clean_image, clean_label = clean_dataset[0]
    defect_image, defect_label = defect_dataset[0]

    assert torch.equal(clean_label, defect_label)
    assert not torch.equal(clean_image, defect_image)
