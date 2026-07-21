"""Uncertainty estimation for segmentation models."""

from neuralimage.uncertainty.config import UncertaintyConfig, build_uncertainty_config
from neuralimage.uncertainty.estimators import (
    ConfidenceHeadEstimator,
    MonteCarloDropoutEstimator,
    TTAVarianceEstimator,
    estimate_uncertainty,
)

__all__ = [
    'UncertaintyConfig',
    'build_uncertainty_config',
    'ConfidenceHeadEstimator',
    'MonteCarloDropoutEstimator',
    'TTAVarianceEstimator',
    'estimate_uncertainty',
]
