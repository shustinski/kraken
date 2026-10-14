"""recognition_pipeline: run cut worker."""
from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
import torch
from multiprocessing.synchronize import Event as MpEvent
from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.uncertainty.config import UncertaintyConfig

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.module_state import _QUEUE_EMPTY

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.cut_image_process import cut_image_process

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.imgpredict import imgpredict

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.imgsew import imgsew

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import (
    RuntimeWorkerConfig,
    _report_worker_exception,
    _try_get_queue_item,
)

def _run_cut_worker(
    error_queue: mp.Queue,
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
    try:
        cut_image_process(
            cut_queue,
            cutted_queue,
            size,
            overlap,
            stop_event,
            use_context_branch,
            use_cross_attention,
            context_crop_size,
            context_input_size,
            compression_factor,
            source_root,
            preprocessing,
            stop_token,
        )
    except Exception:
        _report_worker_exception(error_queue, 'cut')
        raise

def _run_predict_worker(
    error_queue: mp.Queue,
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
    try:
        imgpredict(
            prediction_queue,
            predicted_queue,
            model_path,
            gpu,
            batch_size,
            stop_event,
            recognition_tta_enabled,
            confidence_tta_enabled,
            uncertainty,
            stop_token,
        )
    except Exception:
        _report_worker_exception(error_queue, 'predict')
        raise

def _run_sew_worker(
    error_queue: mp.Queue,
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
    try:
        imgsew(
            output_dir,
            sew_queue,
            sewed_queue,
            jpeg_quality,
            stop_event,
            threshold,
            postprocess_kernel_size,
            confidence_save_mode,
            active_learning,
            active_learning_metadata,
            stop_token,
        )
    except Exception:
        _report_worker_exception(error_queue, 'sew')
        raise

def _try_get_worker_error_message(error_queue: mp.Queue | None) -> str | None:
    payload = _try_get_queue_item(error_queue, timeout=0.0)
    if payload is _QUEUE_EMPTY or not isinstance(payload, dict):
        return None
    worker_name = str(payload.get('worker', 'worker'))
    pid = payload.get('pid')
    traceback_text = str(payload.get('traceback', '')).strip()
    base_message = f'Child process failed. worker={worker_name}, pid={pid}.'
    if not traceback_text:
        return base_message
    return f'{base_message}\n{traceback_text}'

def run_multiprocessing_recognition(
    *,
    workload: RecognitionWorkload,
    worker_counts: WorkerCounts,
    stop_event: MpEvent,
    publish: Publisher,
    stop_token: str,
) -> None:
    runner = MultiprocessingRecognitionRunner(
        config=RuntimeWorkerConfig(
            workload=workload,
            worker_counts=worker_counts,
            stop_token=stop_token,
        ),
        stop_event=stop_event,
        publish=publish,
    )
    runner.run()
