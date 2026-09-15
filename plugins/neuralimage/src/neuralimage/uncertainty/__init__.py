"""Public exports loaded on demand to keep configuration imports lightweight."""
import importlib

_EXPORTS = {'ConfidenceTrainingConfig': ('neuralimage.uncertainty.config', 'ConfidenceTrainingConfig'), 'InferenceUncertaintyConfig': ('neuralimage.uncertainty.config', 'InferenceUncertaintyConfig'), 'UncertaintyConfig': ('neuralimage.uncertainty.config', 'UncertaintyConfig'), 'build_confidence_training_config': ('neuralimage.uncertainty.config', 'build_confidence_training_config'), 'build_inference_uncertainty_config': ('neuralimage.uncertainty.config', 'build_inference_uncertainty_config'), 'build_uncertainty_config': ('neuralimage.uncertainty.config', 'build_uncertainty_config'), 'combine_uncertainty_config': ('neuralimage.uncertainty.config', 'combine_uncertainty_config'), 'migrate_legacy_uncertainty_config': ('neuralimage.uncertainty.config', 'migrate_legacy_uncertainty_config'), 'ConfidenceHeadEstimator': ('neuralimage.uncertainty.estimators', 'ConfidenceHeadEstimator'), 'MonteCarloDropoutEstimator': ('neuralimage.uncertainty.estimators', 'MonteCarloDropoutEstimator'), 'TTAVarianceEstimator': ('neuralimage.uncertainty.estimators', 'TTAVarianceEstimator'), 'estimate_uncertainty': ('neuralimage.uncertainty.estimators', 'estimate_uncertainty')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(importlib.import_module(module), symbol)
    globals()[name] = value
    return value
