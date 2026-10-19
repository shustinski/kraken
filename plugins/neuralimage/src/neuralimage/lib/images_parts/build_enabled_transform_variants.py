"""images: build enabled transform variants."""
from __future__ import annotations
import importlib
import numpy as np
from neuralimage.lib.data_interfaces import SampleGenerationSettings

def resolved_augmentation_variant_count(parameters: SampleGenerationSettings | object) -> int:
    multiplier = max(0.0, min(20.0, float(getattr(parameters, 'augmentation_multiplier', 0.0) or 0.0)))
    if multiplier <= 0.0 and bool(getattr(parameters, 'additional_augmentation', False)):
        return 1
    return int(round(multiplier))

def _augmentation_slot_count(parameters: SampleGenerationSettings | object) -> int:
    return resolved_augmentation_variant_count(parameters) + 1

def _build_enabled_transform_variants(
    *,
    enable_rotate_180: bool,
    enable_rotate_90: bool,
    enable_flip_x: bool,
    enable_flip_y: bool,
    square_patch: bool,
) -> tuple[str, ...]:
    variants: list[str] = ['identity']
    if bool(enable_rotate_180):
        variants.append('rotate_180')
    if bool(enable_rotate_90) and bool(square_patch):
        variants.extend(('rotate_90', 'rotate_270'))
    if bool(enable_flip_x):
        variants.append('flip_x')
    if bool(enable_flip_y):
        variants.append('flip_y')
    return tuple(variants)

def _apply_transform_variant(patch: np.ndarray, variant: str) -> np.ndarray:
    if variant == 'rotate_180':
        return patch[:, ::-1, ::-1].copy()
    if variant == 'rotate_90':
        return np.rot90(patch, k=1, axes=(1, 2)).copy()
    if variant == 'rotate_270':
        return np.rot90(patch, k=-1, axes=(1, 2)).copy()
    if variant == 'flip_x':
        return patch[:, ::-1, :].copy()
    if variant == 'flip_y':
        return patch[:, :, ::-1].copy()
    return patch

def _load_sample_fast_cutter_getitem():
    try:
        module = importlib.import_module('lib.sample_fast_cutter_pyx')
        getitem_fn = getattr(module, 'sample_fast_cutter_getitem', None)
        module_file = str(getattr(module, '__file__', '')).lower()
        is_accelerated = module_file.endswith('.pyd') or module_file.endswith('.so')
        return getitem_fn, is_accelerated
    except Exception:
        return None, False

def is_sample_fast_cutter_accelerated() -> bool:
    from neuralimage.lib.images import _sample_fast_cutter_accelerated, _sample_fast_cutter_getitem

    return bool(_sample_fast_cutter_accelerated and _sample_fast_cutter_getitem is not None)

def _resolve_crops_per_image(params: object) -> int:
    return max(1, int(getattr(params, 'crops_per_image', 64)))
