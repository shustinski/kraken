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
