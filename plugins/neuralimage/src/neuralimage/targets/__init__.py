"""Public exports loaded on demand to keep configuration imports lightweight."""
import importlib

_EXPORTS = {'SupervisionTargetConfig': ('neuralimage.targets.config', 'SupervisionTargetConfig'), 'GeometrySupervisionConfig': ('neuralimage.targets.config', 'GeometrySupervisionConfig'), 'TargetGeneratorRegistry': ('neuralimage.targets.registry', 'TargetGeneratorRegistry'), 'generate_supervision_targets': ('neuralimage.targets.registry', 'generate_supervision_targets'), 'collate_supervision_targets': ('neuralimage.targets.batch', 'collate_supervision_targets'), 'extract_mask_from_target': ('neuralimage.targets.batch', 'extract_mask_from_target')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(importlib.import_module(module), symbol)
    globals()[name] = value
    return value
