from conftest import safe_import_or_skip

safe_import_or_skip('torch')
safe_import_or_skip('torchvision')
safe_import_or_skip('PIL')
safe_import_or_skip('cv2')

import random

import cv2
import numpy as np
import pytest
from PIL import Image

from augmentations import TechVariationAugmentor
from application.dto import MainWindowState, SettingsState
from application.services.workflow_mapper import build_workflow_parameters
from lib.data_interfaces import (
    SampleCutMode,
    SampleGenerationSettings,
    SamplePrepareSettings,
    TrainingParameters,
    build_tech_augmentation_config,
)
from model.NeuralNetwork.dataset import NoCutDataset
from tests.helpers import make_test_dir


def _tech_aug_config(**overrides):
    config = {
        'enabled': True,
        'min_operations': 1,
        'max_operations': 1,
        'global_width': {
            'probability': 1.0,
            'kernel_size_range': [1, 1],
            'erosion_probability': 0.0,
        },
        'scale_rethreshold': {'probability': 0.0},
        'blur_threshold': {'probability': 0.0},
        'boundary_aware': {'probability': 0.0},
        'local_morphology': {'probability': 0.0},
        'gap_variation': {'probability': 0.0},
    }
    config.update(overrides)
    return config


def test_tech_variation_augmentor_debug_mode_returns_original_and_augmented_pair():
    random.seed(7)
    np.random.seed(7)

    mask = np.zeros((1, 16, 16), dtype=np.float32)
    mask[:, 4:12, 7] = 1.0
    augmentor = TechVariationAugmentor(
        _tech_aug_config(debug_return_pair=True)
    )

    original, augmented = augmentor(mask)

    assert original.shape == mask.shape
    assert augmented.shape == mask.shape
    assert np.array_equal(original, mask)
    assert not np.array_equal(augmented, mask)


def test_build_workflow_parameters_maps_tech_aug_config():
    source = make_test_dir("workflow_source_tech_aug")
    result = make_test_dir("workflow_result_tech_aug")
    sample = make_test_dir("workflow_sample_tech_aug")
    label = make_test_dir("workflow_label_tech_aug")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        tech_aug={
            'enabled': True,
            'min_operations': 1,
            'max_operations': 2,
            'boundary_aware': {
                'probability': 0.9,
                'band_width_range': [1, 2],
            },
            'global_width': {
                'probability': 0.8,
                'kernel_size_range': [1, 3],
            },
        }
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.generation.tech_aug.enabled is True
    assert training.generation.tech_aug.min_operations == 1
    assert training.generation.tech_aug.max_operations == 2
    assert training.generation.tech_aug.boundary_aware.probability == pytest.approx(0.9)
    assert training.generation.tech_aug.boundary_aware.band_width_range == (1, 2)
    assert training.generation.tech_aug.global_width.kernel_size_range == (1, 3)


def test_no_cut_dataset_applies_tech_aug_only_for_train_path():
    root = make_test_dir('no_cut_dataset_tech_aug')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / 'sample.png'
    label_path = label_dir / 'sample.png'
    payload = np.zeros((16, 16), dtype=np.uint8)
    payload[4:12, 7] = 255
    Image.fromarray(payload, mode='L').save(image_path)
    Image.fromarray(payload, mode='L').save(label_path)

    generation = SampleGenerationSettings(
        step=16,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        shuffle_patches_in_frame=False,
        tech_aug=build_tech_augmentation_config(_tech_aug_config()),
    )
    settings = TrainingParameters(
        image_path=image_dir,
        label_path=label_dir,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=SamplePrepareSettings(enable_crop=False, enable_resize=False),
    )

    train_dataset = NoCutDataset(
        [(image_path, label_path)],
        settings,
        apply_train_only_transforms=True,
    )
    val_dataset = NoCutDataset(
        [(image_path, label_path)],
        settings,
        apply_train_only_transforms=False,
    )

    train_image, train_label = train_dataset[0]
    val_image, val_label = val_dataset[0]

    assert train_dataset._tech_augmentor is not None
    assert val_dataset._tech_augmentor is None
    assert not np.array_equal(train_label, val_label)
    assert not np.array_equal(train_image, val_image)
    assert np.array_equal(train_image, train_label)


def test_tech_variation_global_width_adds_connected_branch():
    random.seed(11)
    np.random.seed(11)

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[14:18, 6:22] = 1.0
    augmentor = TechVariationAugmentor(
        _tech_aug_config(
            global_width={
                'probability': 1.0,
                'kernel_size_range': [2, 2],
            },
        )
    )

    augmented = augmentor(mask)
    augmented_u8 = (np.asarray(augmented) > 0).astype(np.uint8)
    component_count, _labels = cv2.connectedComponents(augmented_u8, connectivity=8)

    assert np.count_nonzero(augmented) > np.count_nonzero(mask)
    assert component_count == 2


def test_tech_variation_gap_variation_closes_small_gap():
    random.seed(5)
    np.random.seed(5)

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[13:18, 5:12] = 1.0
    mask[13:18, 16:23] = 1.0
    augmentor = TechVariationAugmentor(
        _tech_aug_config(
            global_width={'probability': 0.0},
            gap_variation={
                'probability': 1.0,
                'kernel_size_range': [2, 2],
            },
        )
    )

    before_components, _labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    augmented = augmentor(mask)
    after_components, _labels = cv2.connectedComponents((np.asarray(augmented) > 0).astype(np.uint8), connectivity=8)

    assert before_components == 3
    assert after_components == 2


def test_tech_variation_boundary_aware_stays_trace_like_without_blob_polygons():
    random.seed(13)
    np.random.seed(13)

    mask = np.zeros((48, 48), dtype=np.float32)
    mask[22:26, 8:36] = 1.0
    augmentor = TechVariationAugmentor(
        _tech_aug_config(
            global_width={'probability': 0.0},
            boundary_aware={
                'probability': 1.0,
                'band_width_range': [2, 2],
                'smoothing_kernel_size': 2,
            },
        )
    )

    augmented = np.asarray(augmentor(mask)) > 0
    added_region = np.logical_and(augmented, mask <= 0)
    distance = cv2.distanceTransform(added_region.astype(np.uint8), cv2.DIST_L2, 3)

    assert np.count_nonzero(added_region) > 0
    assert float(distance.max()) <= 2.5
