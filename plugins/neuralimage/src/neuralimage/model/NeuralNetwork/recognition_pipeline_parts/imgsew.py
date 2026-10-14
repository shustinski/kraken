"""recognition_pipeline: imgsew."""
from __future__ import annotations
import os
import multiprocessing as mp
from pathlib import Path
import torch
from multiprocessing.synchronize import Event as MpEvent
from PIL import Image
from neuralimage.active_learning.config import ActiveLearningConfig

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.module_state import _QUEUE_EMPTY

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.load_preview_array import _restore_payload_array_from_multiprocessing

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import _try_get_queue_item

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.sew import sew

def create_batches(tensor_data: torch.Tensor, batch_size: int):
    total = len(tensor_data)
    for index in range(0, total, batch_size):
        yield tensor_data[index : index + batch_size]

def imgsew(
    output_dir: Path | str,
    sew_queue: mp.Queue,
    sewed_queue: mp.Queue,
    jpeg_quality: int,
    stop_event: MpEvent,
    threshold: float | None = None,
    postprocess_kernel_size: int = 0,
    confidence_save_mode: str = 'off',
    active_learning: ActiveLearningConfig | None = None,
    active_learning_metadata: dict[str, object] | None = None,
    stop_token: str = '__STOP__',
) -> None:
    while not stop_event.is_set():
        item = _try_get_queue_item(sew_queue, timeout=0.2)
        if item is _QUEUE_EMPTY:
            continue
        if item == stop_token:
            break
        restored_item = _restore_payload_array_from_multiprocessing(item, 'predicted_image')
        restored_item = _restore_payload_array_from_multiprocessing(restored_item, 'confidence_image')
        restored_item = _restore_payload_array_from_multiprocessing(restored_item, 'variance_image')
        restored_item = _restore_payload_array_from_multiprocessing(restored_item, 'disagreement_image')
        output_path = sew(
            output_dir,
            restored_item,
            jpeg_quality=jpeg_quality,
            threshold=threshold,
            postprocess_kernel_size=postprocess_kernel_size,
            confidence_save_mode=confidence_save_mode,
            active_learning=active_learning,
            active_learning_metadata=active_learning_metadata,
        )
        sewed_queue.put(
            {
                'name': restored_item['name'],
                'source_path': restored_item.get('source_path'),
                'output_path': output_path,
            }
        )

def sew_from_queue(
    output_dir: Path | str,
    sew_queue: mp.Queue,
    sewed_queue: mp.Queue,
    jpeg_quality: int = 95,
    threshold: float | None = None,
    postprocess_kernel_size: int = 0,
    confidence_save_mode: str = 'off',
    active_learning: ActiveLearningConfig | None = None,
    active_learning_metadata: dict[str, object] | None = None,
) -> None:
    item = _try_get_queue_item(sew_queue, timeout=0.2)
    if item is _QUEUE_EMPTY:
        return
    output_path = sew(
        output_dir,
        item,
        jpeg_quality=jpeg_quality,
        threshold=threshold,
        postprocess_kernel_size=postprocess_kernel_size,
        confidence_save_mode=confidence_save_mode,
        active_learning=active_learning,
        active_learning_metadata=active_learning_metadata,
    )
    sewed_queue.put(
        {
            'name': item['name'],
            'source_path': item.get('source_path'),
            'output_path': output_path,
        }
    )

def _confidence_output_root(save_dir: Path | str) -> Path:
    result_dir = Path(save_dir)
    result_name = result_dir.name or 'result'
    return result_dir.parent / f'confidence_{result_name}'

def _save_jpeg_atomic(image: Image.Image, path: Path, *, quality: int) -> None:
    temp_path = path.with_name(f'.{path.name}.tmp')
    image.save(temp_path, format='JPEG', quality=quality)
    os.replace(temp_path, path)

def _save_png_atomic(image: Image.Image, path: Path) -> None:
    temp_path = path.with_name(f'.{path.name}.tmp')
    image.save(temp_path, format='PNG', optimize=False, compress_level=6)
    os.replace(temp_path, path)
