from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip('cv2')

from augmentations import PCBDefectAugmentor
from lib.data_interfaces import SampleCutMode, SampleGenerationSettings, SamplePrepareSettings, TrainingParameters
from model.NeuralNetwork.dataset import NoCutDataset


def _save_binary_image(path: Path, matrix: np.ndarray) -> None:
    Image.fromarray((matrix * 255).astype(np.uint8), mode='L').save(path)


def _make_trace_patch(size: int = 64) -> np.ndarray:
    patch = np.zeros((size, size), dtype=np.float32)
    patch[size // 2 - 2:size // 2 + 2, 8:size - 8] = 1.0
    return patch


def test_pcb_defect_augmentor_returns_passthrough_when_disabled():
    patch = _make_trace_patch()
    augmentor = PCBDefectAugmentor({'enabled': False})

    augmented, defect_mask = augmentor(patch, patch)

    assert np.array_equal(augmented, patch)
    assert np.count_nonzero(defect_mask) == 0


def test_pcb_defect_augmentor_generates_debug_output_and_non_empty_mask():
    patch = _make_trace_patch()
    augmentor = PCBDefectAugmentor(
        {
            'enabled': True,
            'defect_probability': 1.0,
            'min_defects': 1,
            'max_defects': 1,
            'defect_probabilities': {
                'break': 1.0,
                'short': 0.0,
                'missing_copper': 0.0,
                'excess_copper': 0.0,
                'pinhole': 0.0,
                'spurious_copper': 0.0,
                'via': 0.0,
                'misalignment': 0.0,
            },
        }
    )

    original, augmented, defect_mask = augmentor(patch, patch, seed=7, return_debug=True)

    assert np.array_equal(original, patch)
    assert augmented.shape == patch.shape
    assert defect_mask.shape == patch.shape
    assert set(np.unique(defect_mask)).issubset({0.0, 1.0})
    assert np.count_nonzero(defect_mask) > 0
    assert not np.array_equal(augmented, patch)


def test_no_cut_dataset_uses_defect_mask_only_for_train(tmp_path: Path):
    image_patch = _make_trace_patch()
    label_patch = image_patch.copy()
    image_path = tmp_path / 'sample.png'
    label_path = tmp_path / 'label.png'
    _save_binary_image(image_path, image_patch)
    _save_binary_image(label_path, label_patch)

    generation = SampleGenerationSettings(
        step=64,
        segment_size=(64, 64),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
    )
    settings = TrainingParameters(
        image_path=tmp_path,
        label_path=tmp_path,
        shuffle=False,
        validation=False,
        validation_percent=20,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=SamplePrepareSettings(),
        pcb_defects=PCBDefectAugmentor(
            {
                'enabled': True,
                'defect_probability': 1.0,
                'min_defects': 1,
                'max_defects': 1,
                'defect_probabilities': {
                    'break': 1.0,
                    'short': 0.0,
                    'missing_copper': 0.0,
                    'excess_copper': 0.0,
                    'pinhole': 0.0,
                    'spurious_copper': 0.0,
                    'via': 0.0,
                    'misalignment': 0.0,
                },
            }
        ).config,
    )

    train_dataset = NoCutDataset([(image_path, label_path)], settings, apply_train_only_transforms=True)
    val_dataset = NoCutDataset([(image_path, label_path)], settings, apply_train_only_transforms=False)

    train_image, train_label = train_dataset[0]
    val_image, val_label = val_dataset[0]

    assert np.count_nonzero(train_label) > 0
    assert np.count_nonzero(val_label) == np.count_nonzero(label_patch)
    assert not np.array_equal(train_image, val_image)
    assert np.array_equal(val_label[0], label_patch)
