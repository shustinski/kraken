from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    'ICDefectAugmentor',
    'PCBDefectAugmentor',
    'SyntheticTopologyGenerator',
    'SyntheticTopologyParameters',
    'TechVariationAugmentor',
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    'ICDefectAugmentor': ('neuralimage.augmentations.ic_defects', 'ICDefectAugmentor'),
    'PCBDefectAugmentor': ('neuralimage.augmentations.pcb_defects', 'PCBDefectAugmentor'),
    'SyntheticTopologyGenerator': (
        'neuralimage.augmentations.synthetic_topology',
        'SyntheticTopologyGenerator',
    ),
    'SyntheticTopologyParameters': (
        'neuralimage.augmentations.synthetic_topology',
        'SyntheticTopologyParameters',
    ),
    'TechVariationAugmentor': (
        'neuralimage.augmentations.tech_variations',
        'TechVariationAugmentor',
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
