"""Image workflow regressions using generated inputs and the current contracts."""

import numpy as np
import pytest
from PIL import Image

from neuralimage.lib.data_interfaces import (
    CutSettings,
    SampleCutMode,
    SampleGenerationSettings,
    SamplePrepareSettings,
    TrainingParameters,
)
from neuralimage.lib.images import ImagePreparator, SampleCalculator, SampleFastCutter, SampleWorker
from neuralimage.model.NeuralNetwork.dataset import NoCutDataset


@pytest.fixture
def sample_pair(tmp_path):
    image_root, label_root = tmp_path / "images", tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    image, label = image_root / "sample.png", label_root / "sample.png"
    pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    Image.fromarray(pixels).save(image)
    Image.fromarray((pixels > 128).astype(np.uint8) * 255).save(label)
    return image, label


@pytest.fixture
def generation():
    return SampleGenerationSettings(
        step=16,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
    )


def test_image_parts_calculation(sample_pair, generation):
    calculator = SampleCalculator.from_path(sample_pair[0], generation)
    assert len(calculator) == 4
    assert calculator.size == (2, 2)


def test_path_sample_calculation(sample_pair):
    settings = CutSettings(
        step=16,
        x_size=16,
        y_size=16,
        vertical_rotation=False,
        horizontal_rotation=False,
        color_mode="L",
        model="S 660k",
    )
    assert len(SampleWorker(sample_pair[0].parent, settings)) == 4


def test_image_cut_on_fly(sample_pair, generation):
    pair = tuple(ImagePreparator(path, SamplePrepareSettings()).image for path in sample_pair)
    cutter = SampleFastCutter.from_image(pair, generation, shuffle=False)
    assert len(cutter) == 4
    for image, label in cutter:
        assert image.shape == label.shape == (1, 16, 16)


def test_image_preparation(sample_pair):
    parameters = SamplePrepareSettings(enable_crop=True, edge_cut=(2, 2), enable_resize=True, target_size=(16, 16))
    prepared = ImagePreparator(sample_pair[0], parameters)
    assert prepared.image.size == prepared.size == (16, 16)


def test_image_preparation_and_cut(sample_pair, generation):
    parameters = SamplePrepareSettings(enable_resize=True, target_size=(16, 16))
    pair = tuple(ImagePreparator(path, parameters).image for path in sample_pair)
    cutter = SampleFastCutter.from_image(pair, generation, shuffle=False)
    assert len(cutter) == 1
    assert cutter[0][0].shape == cutter[0][1].shape == (1, 16, 16)


def test_total_amount_of_samples(sample_pair, generation):
    image = sample_pair[0]
    image.with_name("second.png").write_bytes(image.read_bytes())
    total = sum(len(SampleCalculator.from_path(path, generation)) for path in image.parent.glob("*.png"))
    assert total == 8


def test_no_cut_dataset(sample_pair, generation):
    image, label = sample_pair
    settings = TrainingParameters(
        image_path=image.parent,
        label_path=label.parent,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=SamplePrepareSettings(),
    )
    dataset = NoCutDataset([sample_pair], settings)
    assert len(dataset) == 4
    for index in range(len(dataset)):
        image_tensor, label_tensor = dataset[index]
        assert tuple(image_tensor.shape) == tuple(label_tensor.shape) == (1, 16, 16)
