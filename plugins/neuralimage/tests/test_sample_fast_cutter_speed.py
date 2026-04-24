from __future__ import annotations
import importlib
import sys
from statistics import median
from time import perf_counter

import numpy as np
import pytest

pytest.importorskip('PIL')

from neuralimage.lib.data_interfaces import SampleGenerationSettings
from neuralimage.lib.images import SampleFastCutter


def _load_compiled_cython_class():
    module = importlib.import_module('lib.sample_fast_cutter_pyx')
    cython_class = getattr(module, 'SampleFastCutterCython', None)
    if cython_class is None:
        pytest.skip('SampleFastCutterCython class is unavailable in lib.sample_fast_cutter_pyx')

    module_file = str(getattr(module, '__file__', '')).lower()
    if not (module_file.endswith('.pyd') or module_file.endswith('.so')):
        pytest.skip('SampleFastCutterCython is not compiled (loaded python shim)')
    return cython_class



def _build_cutter(*, use_accelerator: bool) -> SampleFastCutter:
    rng = np.random.default_rng(12345)
    image_matrix = rng.random((1, 2000, 2000)).astype(np.float32)
    label_matrix = rng.random((1, 2000, 2000)).astype(np.float32)

    params = SampleGenerationSettings(
        step=16,
        segment_size=(64, 64),
        vertical_rotation=True,
        horizontal_rotation=True,
        channels=1,
    )
    cutter = SampleFastCutter((image_matrix, label_matrix), params, shuffle=False)
    cutter._use_accelerator = bool(use_accelerator)
    return cutter


def _measure_getitem_seconds(cutter: SampleFastCutter, indexes: list[int], rounds: int = 7) -> float:
    for _ in range(2):
        for index in indexes:
            cutter[index]

    samples: list[float] = []
    for _ in range(rounds):
        started = perf_counter()
        for index in indexes:
            cutter[index]
        samples.append(perf_counter() - started)
    return float(median(samples))


def test_sample_fast_cutter_cython_matches_python_results():
    cython_class = _load_compiled_cython_class()
    rng = np.random.default_rng(123456)
    image_matrix = rng.random((1, 128, 128)).astype(np.float32)
    label_matrix = rng.random((1, 128, 128)).astype(np.float32)

    for vertical_rotation, horizontal_rotation in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        params = SampleGenerationSettings(
            step=16,
            segment_size=(32, 32),
            vertical_rotation=vertical_rotation,
            horizontal_rotation=horizontal_rotation,
            channels=1,
        )
        pure_python = SampleFastCutter((image_matrix, label_matrix), params, shuffle=False)
        pure_python._use_accelerator = False
        accelerated = cython_class(
            pure_python._parts_list,
            pure_python._vertical_rotation,
            pure_python._horizontal_rotation,
            pure_python._width_steps,
            pure_python._step,
            pure_python._sample_x,
            pure_python._sample_y,
            pure_python._base_w,
            pure_python._base_h,
            pure_python.image_matrix,
            pure_python.label_matrix,
        )

        assert len(accelerated) == len(pure_python)
        for index in range(len(accelerated)):
            image_fast, label_fast = accelerated[index]
            image_py, label_py = pure_python[index]
            assert np.allclose(image_fast, image_py), f'image mismatch at index={index}'
            assert np.allclose(label_fast, label_py), f'label mismatch at index={index}'


def test_sample_fast_cutter_getitem_compiled_vs_python_speed():
    if sys.gettrace() is not None:
        pytest.skip('Skipping speed benchmark under debugger (debugpy can crash with native extension)')

    cython_class = _load_compiled_cython_class()

    pure_python = _build_cutter(use_accelerator=False)
    accelerated = cython_class(
        pure_python._parts_list,
        pure_python._vertical_rotation,
        pure_python._horizontal_rotation,
        pure_python._width_steps,
        pure_python._step,
        pure_python._sample_x,
        pure_python._sample_y,
        pure_python._base_w,
        pure_python._base_h,
        pure_python.image_matrix,
        pure_python.label_matrix,
    )

    probe_indexes = [0, len(accelerated) // 3, len(accelerated) // 2, len(accelerated) - 1]
    for index in probe_indexes:
        image_fast, label_fast = accelerated[index]
        image_py, label_py = pure_python[index]
        assert np.allclose(image_fast, image_py)
        assert np.allclose(label_fast, label_py)

    indexes = list(range(len(accelerated)))
    accelerated_seconds = _measure_getitem_seconds(accelerated, indexes)
    pure_python_seconds = _measure_getitem_seconds(pure_python, indexes)

    speedup = pure_python_seconds / accelerated_seconds if accelerated_seconds > 0 else float('inf')
    slowdown = accelerated_seconds / pure_python_seconds if pure_python_seconds > 0 else float('inf')
    # breakpoint()
    print(
        'SampleFastCutter.__getitem__ benchmark: '
        f'accelerated={accelerated_seconds:.6f}s, '
        f'python={pure_python_seconds:.6f}s, '
        f'speedup={speedup:.3f}x, '
        f'slowdown={slowdown:.3f}x'
    )
    assert speedup >= 1.0, (
        'Accelerated __getitem__ is significantly slower than Python fallback: '
        f'accelerated={accelerated_seconds:.6f}s, '
        f'python={pure_python_seconds:.6f}s, '
        f'speedup={speedup:.3f}x, '
        f'slowdown={slowdown:.3f}x'
    )
