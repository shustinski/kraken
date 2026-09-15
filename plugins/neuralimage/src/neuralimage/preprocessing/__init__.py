"""Public exports loaded on demand to keep configuration imports lightweight."""
import importlib

_EXPORTS = {'NORMALIZATION_MODES': ('neuralimage.preprocessing.config', 'NORMALIZATION_MODES'), 'PreprocessingConfig': ('neuralimage.preprocessing.config', 'PreprocessingConfig'), 'build_preprocessing_config': ('neuralimage.preprocessing.config', 'build_preprocessing_config'), 'SemPreprocessingPipeline': ('neuralimage.preprocessing.pipeline', 'SemPreprocessingPipeline'), 'apply_preprocessing': ('neuralimage.preprocessing.pipeline', 'apply_preprocessing'), 'image_to_channel_first_float01': ('neuralimage.preprocessing.pipeline', 'image_to_channel_first_float01'), 'to_float01': ('neuralimage.preprocessing.pipeline', 'to_float01'), 'DatasetStatistics': ('neuralimage.preprocessing.statistics', 'DatasetStatistics'), 'compute_dataset_statistics': ('neuralimage.preprocessing.statistics', 'compute_dataset_statistics')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(importlib.import_module(module), symbol)
    globals()[name] = value
    return value
