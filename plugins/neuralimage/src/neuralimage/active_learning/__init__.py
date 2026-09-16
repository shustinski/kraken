"""Public exports loaded on demand to keep configuration imports lightweight."""
import importlib

_EXPORTS = {'ActiveLearningConfig': ('neuralimage.active_learning.config', 'ActiveLearningConfig'), 'build_active_learning_config': ('neuralimage.active_learning.config', 'build_active_learning_config'), 'ActiveLearningExporter': ('neuralimage.active_learning.export', 'ActiveLearningExporter'), 'UncertainSampleRecord': ('neuralimage.active_learning.export', 'UncertainSampleRecord'), 'score_prediction_uncertainty': ('neuralimage.active_learning.scoring', 'score_prediction_uncertainty')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(importlib.import_module(module), symbol)
    globals()[name] = value
    return value
