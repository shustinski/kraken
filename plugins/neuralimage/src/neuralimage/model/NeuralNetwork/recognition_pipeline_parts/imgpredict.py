"""recognition_pipeline: imgpredict."""
from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from multiprocessing.synchronize import Event as MpEvent
from PIL import Image
from neuralimage.lib.file_retry import retry_file_read
from neuralimage.uncertainty.config import UncertaintyConfig
from neuralimage.model.NeuralNetwork.model_io import load_model_artifact

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.module_state import _QUEUE_EMPTY

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.gpu_predict import gpu_predict

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.load_preview_array import (
    _restore_payload_array_from_multiprocessing,
    _store_payload_array_for_multiprocessing,
)

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import _try_get_queue_item

def get_array_from_image(path: Path | str, channels: int) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    with retry_file_read(lambda: Image.open(path), path=path) as source_image:
        work_image = np.array(source_image).astype('float32')
    return _to_channel_first(work_image, channels=channels)

def _to_channel_first(work_image: np.ndarray, *, channels: int) -> np.ndarray:
    if work_image.ndim == 2:
        work_image = np.expand_dims(work_image, axis=2)
    if work_image.ndim != 3:
        raise ValueError(f'Unsupported image shape: {work_image.shape}')

    current_channels = int(work_image.shape[2])
    if channels <= 0:
        raise ValueError(f'Invalid channels count: {channels}')

    if current_channels == channels:
        return work_image.transpose(2, 0, 1)
    if channels == 1 and current_channels > 1:
        grayscale = work_image.mean(axis=2, keepdims=True)
        return grayscale.transpose(2, 0, 1)
    if current_channels == 1 and channels > 1:
        expanded = np.repeat(work_image, channels, axis=2)
        return expanded.transpose(2, 0, 1)
    if current_channels > channels:
        trimmed = work_image[:, :, :channels]
        return trimmed.transpose(2, 0, 1)

    # Pad missing channels by repeating the last available one.
    missing_channels = channels - current_channels
    padding = np.repeat(work_image[:, :, -1:], missing_channels, axis=2)
    padded = np.concatenate([work_image, padding], axis=2)
    return padded.transpose(2, 0, 1)

def imgpredict(
    prediction_queue: mp.Queue,
    predicted_queue: mp.Queue,
    model_path: str | Path,
    gpu: torch.device,
    batch_size: int,
    stop_event: MpEvent,
    recognition_tta_enabled: bool = False,
    confidence_tta_enabled: bool = False,
    uncertainty: UncertaintyConfig | None = None,
    stop_token: str = '__STOP__',
) -> None:
    from neuralimage.model.NeuralNetwork.recognition_pipeline import _try_get_queue_item, gpu_predict

    model = load_model_artifact(model_path, map_location='cpu')
    model.eval()
    model.to(gpu)
    if gpu.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    while not stop_event.is_set():
        item = _try_get_queue_item(prediction_queue, timeout=0.2)
        if item is _QUEUE_EMPTY:
            continue
        if item == stop_token:
            break
        restored = _restore_payload_array_from_multiprocessing(item, 'cutted_image')
        restored = _restore_payload_array_from_multiprocessing(restored, 'context_image')
        restored = _restore_payload_array_from_multiprocessing(restored, 'global_image')
        restored = _restore_payload_array_from_multiprocessing(restored, 'patch_coords_norm')
        restored = _restore_payload_array_from_multiprocessing(restored, 'patch_coords_px')
        restored = _restore_payload_array_from_multiprocessing(restored, 'source_size_hw')
        predicted = gpu_predict(
            restored,
            model,
            gpu,
            batch_size,
            recognition_tta_enabled=recognition_tta_enabled,
            confidence_tta_enabled=confidence_tta_enabled,
            uncertainty=uncertainty,
        )
        predicted.pop('cutted_image', None)
        predicted.pop('context_image', None)
        predicted.pop('global_image', None)
        predicted.pop('patch_coords_norm', None)
        predicted.pop('patch_coords_px', None)
        predicted.pop('source_size_hw', None)
        _store_payload_array_for_multiprocessing(predicted, 'predicted_image')
        _store_payload_array_for_multiprocessing(predicted, 'confidence_image')
        _store_payload_array_for_multiprocessing(predicted, 'variance_image')
        _store_payload_array_for_multiprocessing(predicted, 'disagreement_image')
        predicted_queue.put(predicted)
