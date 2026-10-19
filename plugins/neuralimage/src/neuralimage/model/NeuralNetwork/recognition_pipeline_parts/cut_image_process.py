"""recognition_pipeline: cut image process."""
from __future__ import annotations
from typing import TYPE_CHECKING
import multiprocessing as mp
from pathlib import Path
from typing import cast
import numpy as np
from multiprocessing.synchronize import Event as MpEvent
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.preprocessing.pipeline import SemPreprocessingPipeline

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.module_state import _QUEUE_EMPTY

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.cut_image_prepare import cut_image_prepare

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.load_preview_array import _store_payload_array_for_multiprocessing

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import _try_get_queue_item

def _publish_memory_metrics(
    *,
    publish: Publisher,
    collect_memory_metrics: MemoryMetricsCollector,
) -> None:
    payload = collect_memory_metrics()
    if payload is not None:
        publish('metrics', {'type': 'system_memory', **payload})

def cut_image_process(
    cut_queue: mp.Queue,
    cutted_queue: mp.Queue,
    size: tuple[int, int, int],
    overlap: int,
    stop_event: MpEvent,
    use_context_branch: bool = False,
    use_cross_attention: bool = True,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
    compression_factor: int = 1,
    source_root: Path | None = None,
    preprocessing: PreprocessingConfig | None = None,
    stop_token: str = '__STOP__',
) -> None:
    from neuralimage.model.NeuralNetwork.recognition_pipeline import _try_get_queue_item

    from neuralimage.model.NeuralNetwork.recognition_pipeline import cut_image_prepare

    while not stop_event.is_set():
        image_path = _try_get_queue_item(cut_queue, timeout=0.2)
        if image_path is _QUEUE_EMPTY:
            continue
        if image_path == stop_token:
            break
        image_payload = cut_image_prepare(
            cast(Path, image_path),
            size,
            overlap,
            use_context_branch=use_context_branch,
            use_cross_attention=use_cross_attention,
            context_crop_size=context_crop_size,
            context_input_size=context_input_size,
            compression_factor=compression_factor,
            source_root=source_root,
            preprocessing=preprocessing,
        )
        _store_payload_array_for_multiprocessing(image_payload, 'cutted_image')
        _store_payload_array_for_multiprocessing(image_payload, 'context_image')
        _store_payload_array_for_multiprocessing(image_payload, 'global_image')
        _store_payload_array_for_multiprocessing(image_payload, 'patch_coords_norm')
        _store_payload_array_for_multiprocessing(image_payload, 'patch_coords_px')
        _store_payload_array_for_multiprocessing(image_payload, 'source_size_hw')
        cutted_queue.put(image_payload)

def _apply_preprocessing_to_channel_first(
    work_image: np.ndarray,
    preprocessing: PreprocessingConfig | None,
) -> np.ndarray:
    pipeline = SemPreprocessingPipeline(preprocessing)
    if work_image.shape[0] == 1:
        processed = pipeline.apply(work_image[0])
        return np.expand_dims(processed, axis=0).astype(np.float32, copy=False)
    processed_planes = [pipeline.apply(plane) for plane in work_image]
    return np.stack(processed_planes, axis=0).astype(np.float32, copy=False)

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.module_state import MemoryMetricsCollector, Publisher
