import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace

from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.preprocessing.pipeline import SemPreprocessingPipeline, image_to_channel_first_float01
from neuralimage.preprocessing.statistics import compute_dataset_statistics
from neuralimage.model.NeuralNetwork.recognition_pipeline import (
    _apply_preprocessing_to_channel_first,
    cut_image_prepare,
)
from neuralimage.model.general_neural_handler import GeneralNeuralHandler
from neuralimage.targets.dataset_hooks import apply_dataset_preprocessing
from neuralimage.lib.data_interfaces import SamplePrepareSettings


def test_preprocessing_pipeline_identity_when_disabled():
    image = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
    result = SemPreprocessingPipeline(PreprocessingConfig()).apply(image)
    assert result.shape == image.shape
    assert result.max() <= 1.0
    assert np.isclose(result[-1, -1], 1.0)


def test_percentile_normalization_stretches_contrast():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 200
    config = PreprocessingConfig(mode='per_image_percentile', percentile_low=1.0, percentile_high=99.0)
    result = SemPreprocessingPipeline(config).apply(image)
    assert result.max() > result.min()


def test_clahe_runs_on_uint8_input():
    rng = np.random.default_rng(0)
    image = rng.integers(20, 220, size=(32, 32), dtype=np.uint8)
    config = PreprocessingConfig(clahe=True)
    result = SemPreprocessingPipeline(config).apply(image)
    assert result.shape == image.shape


def test_uint16_input_preserves_sub_uint8_intensity_differences():
    image = np.array([[1000, 1001], [1002, 1003]], dtype=np.uint16)
    result = SemPreprocessingPipeline(PreprocessingConfig()).apply(image)
    assert result.dtype == np.float32
    assert len(np.unique(result)) == 4
    assert np.isclose(result[0, 0], 1000.0 / 65535.0)


def test_float_input_in_unit_range_is_not_rescaled():
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    result = SemPreprocessingPipeline(PreprocessingConfig()).apply(image)
    assert np.array_equal(result, image)


def test_channel_conversion_preserves_uint16_grayscale():
    image = np.array([[0, 1], [65534, 65535]], dtype=np.uint16)
    result = image_to_channel_first_float01(image, channels=1)
    assert result.shape == (1, 2, 2)
    assert len(np.unique(result)) == 4
    assert result[0, 1, 1] == 1.0


def test_percentile_normalization_handles_nearly_constant_image():
    image = np.full((16, 16), 0.5, dtype=np.float32)
    image[0, 0] += 1e-8
    config = PreprocessingConfig(mode='per_image_percentile')
    result = SemPreprocessingPipeline(config).apply(image)
    assert np.count_nonzero(result) == 0
    assert np.isfinite(result).all()


def test_dataset_zscore_uses_fixed_statistics():
    image = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    config = PreprocessingConfig(mode='dataset_zscore', dataset_mean=0.5, dataset_std=0.25)
    result = SemPreprocessingPipeline(config).apply(image)
    expected = (image.astype(np.float32) / 255.0 - 0.5) / 0.25
    assert np.allclose(result, expected)


def test_dataset_zscore_requires_training_statistics():
    with pytest.raises(ValueError, match='training dataset'):
        SemPreprocessingPipeline(PreprocessingConfig(mode='dataset_zscore')).apply(
            np.zeros((2, 2), dtype=np.uint8)
        )


def test_dataset_statistics_are_stable_and_constant_data_is_safe():
    stats = compute_dataset_statistics(
        (
            np.array([[0, 255]], dtype=np.uint8),
            np.array([[16384, 32768]], dtype=np.uint16),
        )
    )
    expected = np.array([0.0, 1.0, 16384.0 / 65535.0, 32768.0 / 65535.0])
    assert stats.pixel_count == 4
    assert np.isclose(stats.mean, expected.mean())
    assert np.isclose(stats.std, expected.std())

    constant = compute_dataset_statistics([np.full((8, 8), 7, dtype=np.uint8)])
    assert constant.std == 1.0


def test_training_and_inference_apply_identical_normalization():
    image = np.array([[[0.0, 0.25], [0.75, 1.0]]], dtype=np.float32)
    config = PreprocessingConfig(mode='dataset_zscore', dataset_mean=0.4, dataset_std=0.2)
    training = apply_dataset_preprocessing(image, config)
    inference = _apply_preprocessing_to_channel_first(image, config)
    assert np.array_equal(training, inference)


def test_inference_patch_preparation_preserves_uint16_and_zscore_domain(tmp_path):
    image_path = tmp_path / 'uint16.png'
    image = np.array([[0, 16384], [32768, 65535]], dtype=np.uint16)
    Image.fromarray(image).save(image_path)
    config = PreprocessingConfig(mode='dataset_zscore', dataset_mean=0.5, dataset_std=0.25)

    payload = cut_image_prepare(
        image_path,
        segment_size=(1, 2, 2),
        overlap=0,
        preprocessing=config,
    )

    patch = payload['cutted_image'][0, 0]
    expected = (image.astype(np.float32) / 65535.0 - 0.5) / 0.25
    assert np.allclose(patch, expected)
    assert patch.min() < 0.0


def test_handler_calculates_dataset_zscore_from_passed_train_samples_only(tmp_path):
    train_path = tmp_path / 'train.png'
    validation_path = tmp_path / 'validation.png'
    label_path = tmp_path / 'label.png'
    Image.fromarray(np.array([[0, 64], [128, 255]], dtype=np.uint8)).save(train_path)
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(validation_path)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(label_path)

    messages = []
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.tranining_parameters = SimpleNamespace(
        preprocessing=PreprocessingConfig(mode='dataset_zscore'),
        generation=SimpleNamespace(channels=1),
        prepare=SamplePrepareSettings(),
    )
    handler.message_bus = SimpleNamespace(publish=lambda *args: messages.append(args))

    handler._resolve_training_preprocessing([(train_path, label_path)])

    resolved = handler.tranining_parameters.preprocessing
    expected = np.array([0, 64, 128, 255], dtype=np.float64) / 255.0
    assert np.isclose(resolved.dataset_mean, expected.mean())
    assert np.isclose(resolved.dataset_std, expected.std())
    assert not np.isclose(resolved.dataset_mean, 1.0)
    assert messages


def test_further_training_inherits_preprocessing_from_model_artifact():
    artifact_config = PreprocessingConfig(
        mode='dataset_zscore',
        dataset_mean=0.35,
        dataset_std=0.12,
    )
    model = SimpleNamespace(
        _neuralimage_artifact_metadata={
            'preprocessing': {
                'config': artifact_config.__dict__,
                'hash': artifact_config.stable_hash(),
            }
        }
    )
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.tranining_parameters = SimpleNamespace(preprocessing=PreprocessingConfig())

    handler._apply_artifact_preprocessing_for_training(model)

    assert handler.tranining_parameters.preprocessing.stable_hash() == artifact_config.stable_hash()


def test_further_training_rejects_incompatible_preprocessing_override():
    artifact_config = PreprocessingConfig(mode='per_image_percentile')
    model = SimpleNamespace(
        _neuralimage_artifact_metadata={
            'preprocessing': {
                'config': artifact_config.__dict__,
                'hash': artifact_config.stable_hash(),
            }
        }
    )
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.tranining_parameters = SimpleNamespace(
        preprocessing=PreprocessingConfig(
            mode='dataset_zscore',
            dataset_mean=0.5,
            dataset_std=0.2,
        )
    )

    with pytest.raises(ValueError, match='incompatible'):
        handler._apply_artifact_preprocessing_for_training(model)


def test_invalid_preprocessing_configuration_fails_early():
    with pytest.raises(ValueError, match='normalization mode'):
        PreprocessingConfig(mode='unknown')
    with pytest.raises(ValueError, match='Percentile range'):
        PreprocessingConfig(percentile_low=99.0, percentile_high=1.0)
    with pytest.raises(ValueError, match='odd integer'):
        PreprocessingConfig(illumination_kernel_size=50)


def test_scan_line_suppression_removes_row_spike_without_flattening_conductor():
    image = np.full((64, 64), 0.2, dtype=np.float32)
    image[:, 28:36] = 0.8
    image[20] += 0.1
    config = PreprocessingConfig(
        scan_line_suppression=True,
        scan_line_strength=1.0,
        scan_profile_kernel=15,
    )
    result = SemPreprocessingPipeline(config).apply(image)
    assert abs(float(result[20, :20].mean() - result[19, :20].mean())) < 0.02
    assert float(result[:, 28:36].mean() - result[:, :20].mean()) > 0.5
