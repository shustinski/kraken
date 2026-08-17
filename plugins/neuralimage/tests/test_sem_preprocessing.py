import numpy as np

from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.preprocessing.pipeline import SemPreprocessingPipeline


def test_preprocessing_pipeline_identity_when_disabled():
    image = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
    result = SemPreprocessingPipeline(PreprocessingConfig()).apply(image)
    assert result.shape == image.shape
    assert result.max() <= 1.0


def test_percentile_normalization_stretches_contrast():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 200
    config = PreprocessingConfig(percentile_normalization=True, percentile_low=1.0, percentile_high=99.0)
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


def test_invalid_preprocessing_configuration_fails_early():
    import pytest

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
