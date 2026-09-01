"""Uncertainty estimation for segmentation models."""

from neuralimage.uncertainty.config import (
    ConfidenceTrainingConfig,
    InferenceUncertaintyConfig,
    UncertaintyConfig,
    build_confidence_training_config,
    build_inference_uncertainty_config,
    build_uncertainty_config,
    combine_uncertainty_config,
    migrate_legacy_uncertainty_config,
)
from neuralimage.uncertainty.estimators import (
    ConfidenceHeadEstimator,
    MonteCarloDropoutEstimator,
    TTAVarianceEstimator,
    estimate_uncertainty,
)

__all__ = [
    'ConfidenceTrainingConfig',
    'InferenceUncertaintyConfig',
    'UncertaintyConfig',
    'build_confidence_training_config',
    'build_inference_uncertainty_config',
    'build_uncertainty_config',
    'combine_uncertainty_config',
    'migrate_legacy_uncertainty_config',
    'ConfidenceHeadEstimator',
    'MonteCarloDropoutEstimator',
    'TTAVarianceEstimator',
    'estimate_uncertainty',
]
