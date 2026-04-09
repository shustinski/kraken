from __future__ import annotations

import os
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterable, cast
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing.synchronize import Event as MpEvent
from PIL import Image

from lib.file_retry import retry_file_read
from lib.image_processing import cut_image, sew_image
from model.NeuralNetwork.model_io import load_model_artifact
from model.NeuralNetwork.context_utils import build_context_batch
from model.NeuralNetwork.blocks import extract_confidence_output, extract_mask_outputs


Publisher = Callable[[str, Any], None]
MemoryMetricsCollector = Callable[[], dict[str, float] | None]
_QUEUE_EMPTY = object()


def _extract_confidence_calibration(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    metadata = getattr(model, '_neuralimage_artifact_metadata', None)
    if not isinstance(metadata, dict):
        return None
    confidence = metadata.get('confidence')
    if not isinstance(confidence, dict):
        return None
    raw_edges = confidence.get('calibration_bin_edges')
    raw_values = confidence.get('calibration_bin_values')
    if not isinstance(raw_edges, (list, tuple)) or not isinstance(raw_values, (list, tuple)):
        return None
    if len(raw_edges) != len(raw_values) + 1 or len(raw_values) <= 0:
        return None
    try:
        edges = torch.tensor([float(value) for value in raw_edges[1:-1]], dtype=torch.float32)
        values = torch.tensor([float(value) for value in raw_values], dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if values.numel() <= 0:
        return None
    return edges, values


def _apply_confidence_calibration(
    confidence_outputs: torch.Tensor,
    calibration: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if calibration is None:
        return confidence_outputs
    edges, values = calibration
    bucket_edges = edges.to(device=confidence_outputs.device, dtype=confidence_outputs.dtype)
    calibrated_values = values.to(device=confidence_outputs.device, dtype=confidence_outputs.dtype)
    bucket_indices = torch.bucketize(confidence_outputs, bucket_edges)
    return calibrated_values[bucket_indices]


@dataclass(frozen=True)
class RecognitionWorkload:
    source_files: list[Path]
    result_folder: Path
    part_size: tuple[int, int]
    overlap: int
    batch_size: int
    colors: int
    jpeg_quality: int
    binarize_output: bool
    threshold: float | None
    postprocess_enabled: bool
    postprocess_kernel_size: int
    recognition_tta_enabled: bool
    confidence_tta_enabled: bool
    confidence_save_mode: str
    devices: list[torch.device]
    model_source: str | Path
    use_context_branch: bool = False
    context_crop_size: tuple[int, int] | None = None
    context_input_size: tuple[int, int] | None = None

    @property
    def frame_count(self) -> int:
        return int(len(self.source_files))

    @property
    def segment_shape(self) -> tuple[int, int, int]:
        return (self.colors, self.part_size[0], self.part_size[1])


@dataclass(frozen=True)
class WorkerCounts:
    cut: int
    predict: int
    sew: int


@dataclass(frozen=True)
class RuntimeWorkerConfig:
    workload: RecognitionWorkload
    worker_counts: WorkerCounts
    stop_token: str


@dataclass
class PipelineQueues:
    cut: mp.Queue
    predict: mp.Queue
    sew: mp.Queue
    sewed: mp.Queue
    errors: mp.Queue

    def close(self) -> None:
        for queue in (self.cut, self.predict, self.sew, self.sewed, self.errors):
            try:
                queue.close()
            except Exception:
                pass


@dataclass
class WorkerGroups:
    cut: list[mp.Process]
    predict: list[mp.Process]
    sew: list[mp.Process]

    def all(self) -> list[mp.Process]:
        return [*self.cut, *self.predict, *self.sew]


class ProgressReporter:
    def __init__(self, publish: Publisher, total_frames: int):
        self._publish = publish
        self._total_frames = int(total_frames)

    def publish_started(self) -> None:
        self._publish(
            'metrics',
            {'type': 'recognition_progress', 'current': 0, 'total': int(self._total_frames)},
        )

    def publish_worker_plan(self, worker_counts: WorkerCounts) -> None:
        self._publish(
            'logging',
            (
                'Recognition worker plan: '
                f'cut={worker_counts.cut}, '
                f'predict={worker_counts.predict}, '
                f'sew={worker_counts.sew}'
            ),
        )

    def publish_frame(self, current_frame: int) -> None:
        self._publish(
            'metrics',
            {
                'type': 'recognition_progress',
                'current': int(current_frame),
                'total': int(self._total_frames),
            },
        )


def _try_get_queue_item(queue: Any, *, timeout: float) -> Any:
    """Poll a multiprocessing queue without surfacing Empty in debuggers."""
    if queue is None:
        return _QUEUE_EMPTY
    reader = getattr(queue, '_reader', None)
    if reader is not None:
        try:
            if not bool(reader.poll(timeout)):
                return _QUEUE_EMPTY
            # After poll() reports readiness, a blocking get() avoids get_nowait()
            # spuriously raising Empty in debugger-attached multiprocessing runs.
            return queue.get()
        except (Empty, EOFError, OSError, ValueError):
            return _QUEUE_EMPTY

    try:
        return queue.get(timeout=timeout)
    except Empty:
        return _QUEUE_EMPTY


def _report_worker_exception(error_queue: mp.Queue | None, worker_name: str) -> None:
    if error_queue is None:
        return
    try:
        error_queue.put(
            {
                'worker': str(worker_name),
                'pid': int(os.getpid()),
                'traceback': traceback.format_exc(),
            }
        )
    except Exception:
        pass


def _run_cut_worker(
    error_queue: mp.Queue,
    cut_queue: mp.Queue,
    cutted_queue: mp.Queue,
    size: tuple[int, int, int],
    overlap: int,
    stop_event: MpEvent,
    use_context_branch: bool = False,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
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
            context_crop_size,
            context_input_size,
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


class MultiprocessingRecognitionRunner:
    def __init__(
        self,
        *,
        config: RuntimeWorkerConfig,
        stop_event: MpEvent,
        publish: Publisher,
    ) -> None:
        self._config = config
        self._stop_event = stop_event
        self._publish = publish
        self._reporter = ProgressReporter(publish=publish, total_frames=config.workload.frame_count)

    def run(self) -> None:
        queues = self._create_queues()
        self._errors_queue_ref = getattr(queues, 'errors', None)
        groups = WorkerGroups(cut=[], predict=[], sew=[])
        predict_stopped = False
        sew_stopped = False
        completed = 0
        failure_message: str | None = None
        try:
            self._prime_cut_queue(queues)
            self._publish_runtime_plan()
            groups = self._start_workers(queues)
            predict_stopped, sew_stopped, completed, failure_message = self._monitor_processing(
                groups=groups,
                queues=queues,
            )
        finally:
            self._send_missing_stop_tokens(
                queues=queues,
                predict_stopped=predict_stopped,
                sew_stopped=sew_stopped,
            )
            self._shutdown_workers(groups)
            queues.close()
        if failure_message is not None:
            raise RuntimeError(failure_message)
        if (not self._stop_event.is_set()) and completed < self._config.workload.frame_count:
            raise RuntimeError(
                'Recognition pipeline stopped before all frames were processed '
                f'({completed}/{self._config.workload.frame_count}).'
            )

    def _create_queues(self) -> PipelineQueues:
        worker_total = max(
            4,
            int(self._config.worker_counts.cut + self._config.worker_counts.predict + self._config.worker_counts.sew),
        )
        predict_queue_size = max(8, worker_total * 2, len(self._config.workload.devices) * 6)
        return PipelineQueues(
            cut=mp.Queue(),
            predict=mp.Queue(maxsize=predict_queue_size),
            sew=mp.Queue(maxsize=predict_queue_size),
            sewed=mp.Queue(),
            errors=mp.Queue(),
        )

    def _prime_cut_queue(self, queues: PipelineQueues) -> None:
        for image in self._config.workload.source_files:
            queues.cut.put(image)
        for _ in range(self._config.worker_counts.cut):
            queues.cut.put(self._config.stop_token)

    def _publish_runtime_plan(self) -> None:
        self._reporter.publish_started()
        self._reporter.publish_worker_plan(self._config.worker_counts)

    def _start_workers(self, queues: PipelineQueues) -> WorkerGroups:
        cut_workers = self._start_cut_workers(queues)
        predict_workers = self._start_predict_workers(queues)
        sew_workers = self._start_sew_workers(queues)
        return WorkerGroups(cut=cut_workers, predict=predict_workers, sew=sew_workers)

    def _start_cut_workers(self, queues: PipelineQueues) -> list[mp.Process]:
        self._publish('logging', 'Starting cut workers.')
        processes: list[mp.Process] = []
        for _ in range(self._config.worker_counts.cut):
            process = mp.Process(
                target=_run_cut_worker,
                args=(
                    queues.errors,
                    queues.cut,
                    queues.predict,
                    self._config.workload.segment_shape,
                    self._config.workload.overlap,
                    self._stop_event,
                    self._config.workload.use_context_branch,
                    self._config.workload.context_crop_size,
                    self._config.workload.context_input_size,
                    self._config.stop_token,
                ),
            )
            process.start()
            processes.append(process)
        return processes

    def _start_predict_workers(self, queues: PipelineQueues) -> list[mp.Process]:
        self._publish('logging', 'Starting predict workers.')
        processes: list[mp.Process] = []
        for worker_index in range(self._config.worker_counts.predict):
            device = self._resolve_worker_device(worker_index)
            process = mp.Process(
                target=_run_predict_worker,
                args=(
                    queues.errors,
                    queues.predict,
                    queues.sew,
                    self._config.workload.model_source,
                    device,
                    self._config.workload.batch_size,
                    self._stop_event,
                    self._config.workload.recognition_tta_enabled,
                    self._config.workload.confidence_tta_enabled,
                    self._config.stop_token,
                ),
            )
            process.start()
            processes.append(process)
        return processes

    def _start_sew_workers(self, queues: PipelineQueues) -> list[mp.Process]:
        self._publish('logging', 'Starting sew workers.')
        processes: list[mp.Process] = []
        for _ in range(self._config.worker_counts.sew):
            process = mp.Process(
                target=_run_sew_worker,
                args=(
                    queues.errors,
                    self._config.workload.result_folder,
                    queues.sew,
                    queues.sewed,
                    self._config.workload.jpeg_quality,
                    self._stop_event,
                    self._config.workload.threshold,
                    (
                        self._config.workload.postprocess_kernel_size
                        if self._config.workload.threshold is not None and self._config.workload.postprocess_enabled
                        else 0
                    ),
                    self._config.workload.confidence_save_mode,
                    self._config.stop_token,
                ),
            )
            process.start()
            processes.append(process)
        return processes

    def _resolve_worker_device(self, worker_index: int) -> torch.device:
        devices = self._config.workload.devices
        if not devices:
            return torch.device('cpu')
        device_index = min(worker_index, len(devices) - 1)
        return devices[device_index]

    def _monitor_processing(
        self,
        *,
        groups: WorkerGroups,
        queues: PipelineQueues,
    ) -> tuple[bool, bool, int, str | None]:
        completed = 0
        predict_stopped = False
        sew_stopped = False
        failure_message: str | None = None
        while completed < self._config.workload.frame_count:
            if self._stop_event.is_set():
                break

            if not predict_stopped and self._all_stopped(groups.cut):
                self._signal_predict_stop(queues.predict)
                predict_stopped = True

            if predict_stopped and (not sew_stopped) and self._all_stopped(groups.predict):
                self._signal_sew_stop(queues.sew)
                sew_stopped = True

            completed = self._consume_completed_frame(queues=queues, completed=completed)
            failed_process_message = self._get_failed_process_message(groups)
            if failed_process_message is not None:
                failure_message = failed_process_message
                self._stop_event.set()
                break

        return predict_stopped, sew_stopped, completed, failure_message

    @staticmethod
    def _all_stopped(processes: Iterable[mp.Process]) -> bool:
        return all(not process.is_alive() for process in processes)

    def _signal_predict_stop(self, predict_queue: mp.Queue) -> None:
        for _ in range(self._config.worker_counts.predict):
            predict_queue.put(self._config.stop_token)

    def _signal_sew_stop(self, sew_queue: mp.Queue) -> None:
        for _ in range(self._config.worker_counts.sew):
            sew_queue.put(self._config.stop_token)

    def _consume_completed_frame(self, *, queues: PipelineQueues, completed: int) -> int:
        completed_item = _try_get_queue_item(queues.sewed, timeout=0.2)
        if completed_item is _QUEUE_EMPTY:
            return completed
        if isinstance(completed_item, dict):
            _publish_recognition_preview(
                publish=self._publish,
                source_path=completed_item.get('source_path'),
                output_path=completed_item.get('output_path'),
                frame_name=str(completed_item.get('name', '')).strip(),
            )
        updated = completed + 1
        self._reporter.publish_frame(updated)
        return updated

    def _get_failed_process_message(self, groups: WorkerGroups) -> str | None:
        detailed_error = _try_get_worker_error_message(getattr(self, '_errors_queue_ref', None))
        if detailed_error is not None:
            self._publish('logging', detailed_error)
            return detailed_error
        for process in groups.all():
            if process.exitcode in (None, 0):
                continue
            message = f'Child process failed. pid={process.pid}, code={process.exitcode}.'
            self._publish('logging', message)
            return message
        return None

    def _send_missing_stop_tokens(
        self,
        *,
        queues: PipelineQueues,
        predict_stopped: bool,
        sew_stopped: bool,
    ) -> None:
        if not predict_stopped:
            self._signal_predict_stop(queues.predict)
        if not sew_stopped:
            self._signal_sew_stop(queues.sew)

    @staticmethod
    def _shutdown_workers(groups: WorkerGroups) -> None:
        for process in groups.all():
            process.join(timeout=5)
        for process in groups.all():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


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


def run_single_thread_recognition(
    *,
    source_files: list[Path],
    result_folder: Path,
    part_size: tuple[int, int],
    overlap: int,
    batch_size: int,
    colors: int,
    model: nn.Module,
    device: torch.device,
    stop_event: Any,
    publish: Publisher,
    collect_memory_metrics: MemoryMetricsCollector,
    jpeg_quality: int = 95,
    binarize_output: bool = True,
    threshold: float | None = None,
    postprocess_enabled: bool = False,
    postprocess_kernel_size: int = 3,
    recognition_tta_enabled: bool = False,
    confidence_tta_enabled: bool = False,
    confidence_save_mode: str = 'off',
    use_context_branch: bool = False,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
) -> None:
    model.eval()
    model.to(device)
    publish('logging', 'Recognition started in single-thread mode.')

    reporter = ProgressReporter(publish=publish, total_frames=len(source_files))
    reporter.publish_started()
    _publish_memory_metrics(publish=publish, collect_memory_metrics=collect_memory_metrics)
    compile_fallback_checked = False

    shape = (colors, part_size[0], part_size[1])
    for index, image_path in enumerate(source_files, start=1):
        if stop_event.is_set():
            break
        prepared = cut_image_prepare(
            image_path,
            shape,
            overlap,
            use_context_branch=use_context_branch,
            context_crop_size=context_crop_size,
            context_input_size=context_input_size,
        )
        predicted = gpu_predict(
            prepared,
            model,
            device,
            batch_size,
            recognition_tta_enabled=recognition_tta_enabled,
            confidence_tta_enabled=confidence_tta_enabled,
        )
        # Some torch.compile configurations can produce degenerate outputs on inference.
        if (not compile_fallback_checked) and hasattr(model, '_orig_mod'):
            compile_fallback_checked = True
            compiled_stats = predicted.get('_prediction_stats')
            compiled_max = (
                float(compiled_stats.get('max', 0.0))
                if isinstance(compiled_stats, dict)
                else 0.0
            )
            original_model = getattr(model, '_orig_mod', None)
            if isinstance(original_model, nn.Module) and compiled_max <= 0.05:
                publish(
                    'logging',
                    (
                        'Recognition: low-probability output detected on torch.compile model; '
                        'retrying with uncompiled model.'
                    ),
                )
                original_model.eval()
                original_model.to(device)
                fallback_predicted = gpu_predict(
                    prepared,
                    original_model,
                    device,
                    batch_size,
                    recognition_tta_enabled=recognition_tta_enabled,
                    confidence_tta_enabled=confidence_tta_enabled,
                )
                fallback_stats = fallback_predicted.get('_prediction_stats')
                fallback_max = (
                    float(fallback_stats.get('max', 0.0))
                    if isinstance(fallback_stats, dict)
                    else 0.0
                )
                if fallback_max > (compiled_max + 0.01):
                    model = original_model
                    predicted = fallback_predicted
                    publish(
                        'logging',
                        (
                            'Recognition: switched to uncompiled model '
                            f'(max prob {compiled_max:.6f} -> {fallback_max:.6f}).'
                        ),
                    )
        stats = predicted.get('_prediction_stats')
        if isinstance(stats, dict):
            prob_min = float(stats.get('min', 0.0))
            prob_max = float(stats.get('max', 0.0))
            prob_mean = float(stats.get('mean', 0.0))
            non_finite = int(stats.get('non_finite', 0))
            fp32_retries = int(stats.get('fp32_retries', 0))
            should_log_stats = index == 1 or non_finite > 0 or prob_max <= 0.05
            if should_log_stats:
                publish(
                    'logging',
                    (
                        f'Recognition output stats [{index}/{len(source_files)}]: '
                        f'min={prob_min:.6f}, max={prob_max:.6f}, mean={prob_mean:.6f}, '
                        f'non_finite={non_finite}, fp32_retries={fp32_retries}'
                    ),
                )
            if prob_max <= 0.05:
                publish(
                    'logging',
                    (
                        'Recognition warning: output probabilities are very low '
                        '(max <= 0.05), resulting masks can look black.'
                    ),
                )
        output_path = sew(
            result_folder,
            predicted,
            jpeg_quality=jpeg_quality,
            threshold=(float(threshold) if binarize_output and threshold is not None else None),
            postprocess_kernel_size=(int(postprocess_kernel_size) if postprocess_enabled else 0),
            confidence_save_mode=confidence_save_mode,
        )
        _publish_recognition_preview(
            publish=publish,
            source_path=image_path,
            output_path=output_path,
            frame_name=image_path.name,
        )
        reporter.publish_frame(index)
        _publish_memory_metrics(publish=publish, collect_memory_metrics=collect_memory_metrics)


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
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
    stop_token: str = '__STOP__',
) -> None:
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
            context_crop_size=context_crop_size,
            context_input_size=context_input_size,
        )
        _store_payload_array_for_multiprocessing(image_payload, 'cutted_image')
        _store_payload_array_for_multiprocessing(image_payload, 'context_image')
        cutted_queue.put(image_payload)


def cut_image_prepare(
    img_path: Path,
    segment_size: tuple[int, int, int],
    overlap: int,
    *,
    use_context_branch: bool = False,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'baseim_size': None,
        'segment_size': segment_size,
        'overlap': overlap,
        'cutted_image': None,
        'context_image': None,
        'name': img_path.name,
        'source_path': img_path,
    }

    Image.MAX_IMAGE_PIXELS = None
    channels = segment_size[0]
    with retry_file_read(lambda: Image.open(img_path), path=img_path) as source_image:
        if source_image.mode != 'L' and channels == 1:
            source_image = source_image.convert('L')
        payload['baseim_size'] = source_image.size
        work_image = np.array(source_image).astype('float32')

    work_image = _to_channel_first(work_image, channels=channels)
    payload['cutted_image'] = cut_image(work_image, segment_size, overlap)
    if use_context_branch and context_crop_size is not None and context_input_size is not None:
        payload['context_image'] = (
            build_context_batch(
                work_image,
                local_patch_size_xy=(int(segment_size[1]), int(segment_size[2])),
                overlap=overlap,
                context_crop_size_xy=tuple(context_crop_size),
                context_input_size_xy=tuple(context_input_size),
            )
            / 255.0
        ).astype(np.float32, copy=False)
    return payload


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
    stop_token: str = '__STOP__',
) -> None:
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
        predicted = gpu_predict(
            restored,
            model,
            gpu,
            batch_size,
            recognition_tta_enabled=recognition_tta_enabled,
            confidence_tta_enabled=confidence_tta_enabled,
        )
        predicted.pop('cutted_image', None)
        predicted.pop('context_image', None)
        _store_payload_array_for_multiprocessing(predicted, 'predicted_image')
        _store_payload_array_for_multiprocessing(predicted, 'confidence_image')
        predicted_queue.put(predicted)


def gpu_predict(
    img: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    *,
    recognition_tta_enabled: bool = False,
    confidence_tta_enabled: bool = False,
) -> dict[str, Any]:
    predicted_image = np.empty_like(img['cutted_image'])
    confidence_image = np.empty_like(img['cutted_image'])
    parts_in_image = len(img['cutted_image'])

    source_batches = img['cutted_image']
    if torch.is_tensor(source_batches):
        tensor_data = source_batches.float()
    else:
        tensor_data = torch.from_numpy(source_batches).float()
    context_batches = img.get('context_image')
    context_tensor_data: torch.Tensor | None = None
    if context_batches is not None:
        if torch.is_tensor(context_batches):
            context_tensor_data = context_batches.float()
        else:
            context_tensor_data = torch.from_numpy(context_batches).float()
    use_amp = device.type == 'cuda'
    if use_amp and hasattr(tensor_data, 'pin_memory'):
        with suppress(Exception):
            tensor_data = tensor_data.pin_memory()
    if use_amp and context_tensor_data is not None and hasattr(context_tensor_data, 'pin_memory'):
        with suppress(Exception):
            context_tensor_data = context_tensor_data.pin_memory()
    min_prob = float('inf')
    max_prob = float('-inf')
    mean_acc = 0.0
    processed_batches = 0
    non_finite_values = 0
    fp32_retries = 0
    tta_flip = bool(recognition_tta_enabled or confidence_tta_enabled)
    multiscale_values = str(os.getenv('NEURALIMAGE_MS_SCALES', '1.0')).strip()
    ms_scales: list[float] = []
    for token in multiscale_values.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            scale = float(token)
        except ValueError:
            continue
        if scale <= 0.0:
            continue
        ms_scales.append(scale)
    if not ms_scales:
        ms_scales = [1.0]
    confidence_calibration = _extract_confidence_calibration(model)

    def _extract_prediction_tensors(outputs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        mask_outputs = extract_mask_outputs(outputs)
        if torch.is_tensor(mask_outputs):
            primary = mask_outputs
        elif isinstance(mask_outputs, (list, tuple)) and mask_outputs and torch.is_tensor(mask_outputs[0]):
            primary = cast(torch.Tensor, mask_outputs[0])
        else:
            raise TypeError('Recognition model must return a tensor or a non-empty sequence of tensors.')
        confidence = extract_confidence_output(outputs)
        return primary, confidence

    def _predict_once(
        batch_tensor: torch.Tensor,
        context_batch_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if context_batch_tensor is None:
                return _extract_prediction_tensors(model(batch_tensor))
            return _extract_prediction_tensors(model({'local_image': batch_tensor, 'context_image': context_batch_tensor}))

    def _predict_with_multi_scale_and_tta(
        batch_tensor: torch.Tensor,
        context_batch_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_h, base_w = int(batch_tensor.shape[-2]), int(batch_tensor.shape[-1])
        mask_acc = None
        confidence_acc: torch.Tensor | None = None
        confidence_from_mask_acc: torch.Tensor | None = None
        mask_weight = 0.0
        confidence_weight = 0.0
        for scale in ms_scales:
            if abs(scale - 1.0) < 1e-8:
                scaled = batch_tensor
                scaled_context = context_batch_tensor
            else:
                scaled_h = max(8, int(round(base_h * scale)))
                scaled_w = max(8, int(round(base_w * scale)))
                scaled = F.interpolate(
                    batch_tensor,
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False,
                )
                scaled_context = None
                if context_batch_tensor is not None:
                    context_h, context_w = int(context_batch_tensor.shape[-2]), int(context_batch_tensor.shape[-1])
                    scaled_context_h = max(8, int(round(context_h * scale)))
                    scaled_context_w = max(8, int(round(context_w * scale)))
                    scaled_context = F.interpolate(
                        context_batch_tensor,
                        size=(scaled_context_h, scaled_context_w),
                        mode='bilinear',
                        align_corners=False,
                    )
            logits, confidence_logits = _predict_once(scaled, scaled_context)
            if logits.shape[-2:] != (base_h, base_w):
                logits = F.interpolate(logits, size=(base_h, base_w), mode='bilinear', align_corners=False)
            if mask_acc is None:
                mask_acc = torch.zeros_like(logits)
            mask_acc += logits
            mask_weight += 1.0
            if confidence_logits is not None and confidence_logits.shape[-2:] != (base_h, base_w):
                confidence_logits = F.interpolate(
                    confidence_logits,
                    size=(base_h, base_w),
                    mode='bilinear',
                    align_corners=False,
                )
            if confidence_logits is not None:
                if confidence_acc is None:
                    confidence_acc = torch.zeros_like(logits)
                confidence_acc += confidence_logits
            else:
                if confidence_from_mask_acc is None:
                    confidence_from_mask_acc = torch.zeros_like(logits)
                confidence_from_mask_acc += logits
            confidence_weight += 1.0
            if tta_flip:
                flipped_context = None
                if scaled_context is not None:
                    flipped_context = torch.flip(scaled_context, dims=[-1])
                logits_h, confidence_h = _predict_once(torch.flip(scaled, dims=[-1]), flipped_context)
                logits_h = torch.flip(logits_h, dims=[-1])
                if logits_h.shape[-2:] != (base_h, base_w):
                    logits_h = F.interpolate(logits_h, size=(base_h, base_w), mode='bilinear', align_corners=False)
                if bool(recognition_tta_enabled):
                    mask_acc += logits_h
                    mask_weight += 1.0
                if confidence_h is not None:
                    confidence_h = torch.flip(confidence_h, dims=[-1])
                    if confidence_h.shape[-2:] != (base_h, base_w):
                        confidence_h = F.interpolate(
                            confidence_h,
                            size=(base_h, base_w),
                            mode='bilinear',
                            align_corners=False,
                        )
                if bool(confidence_tta_enabled):
                    if confidence_h is not None:
                        if confidence_acc is None:
                            confidence_acc = torch.zeros_like(logits_h)
                        confidence_acc += confidence_h
                    else:
                        if confidence_from_mask_acc is None:
                            confidence_from_mask_acc = torch.zeros_like(logits_h)
                        confidence_from_mask_acc += logits_h
                    confidence_weight += 1.0
        if mask_acc is None:
            raise RuntimeError('Recognition TTA accumulator is empty.')
        averaged_mask = mask_acc / max(mask_weight, 1.0)
        if confidence_acc is not None:
            averaged_confidence = confidence_acc / max(confidence_weight, 1.0)
        elif confidence_from_mask_acc is not None:
            averaged_confidence = confidence_from_mask_acc / max(confidence_weight, 1.0)
        else:
            averaged_confidence = None
        return averaged_mask, averaged_confidence

    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, parts_in_image, batch_size)):
            end = min(start + batch_size, parts_in_image)
            batch = tensor_data[start:end].to(device, non_blocking=use_amp)
            context_batch = None
            if context_tensor_data is not None:
                context_batch = context_tensor_data[start:end].to(device, non_blocking=use_amp)
            outputs, confidence_outputs = _predict_with_multi_scale_and_tta(batch, context_batch)

            if not bool(torch.isfinite(outputs).all()):
                fp32_retries += 1
                if context_batch is None:
                    outputs, confidence_outputs = _extract_prediction_tensors(model(batch.float()))
                else:
                    outputs, confidence_outputs = _extract_prediction_tensors(
                        model(
                            {
                                'local_image': batch.float(),
                                'context_image': context_batch.float(),
                            }
                        )
                    )

            finite_mask = torch.isfinite(outputs)
            if not bool(finite_mask.all()):
                non_finite_values += int((~finite_mask).sum().item())
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)

            outputs = torch.sigmoid(outputs)
            if not bool(torch.isfinite(outputs).all()):
                finite_after_sigmoid = torch.isfinite(outputs)
                non_finite_values += int((~finite_after_sigmoid).sum().item())
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
            if confidence_outputs is None:
                confidence_outputs = 1.0 - (torch.abs(outputs - 0.5) * 2.0)
            else:
                confidence_outputs = torch.sigmoid(confidence_outputs)
            if not bool(torch.isfinite(confidence_outputs).all()):
                confidence_outputs = torch.nan_to_num(confidence_outputs, nan=0.0, posinf=1.0, neginf=0.0)
            confidence_outputs = torch.clamp(confidence_outputs, min=0.0, max=1.0)
            confidence_outputs = torch.clamp(
                _apply_confidence_calibration(confidence_outputs, confidence_calibration),
                min=0.0,
                max=1.0,
            )

            predictions = outputs.detach().cpu().numpy()
            confidence_predictions = confidence_outputs.detach().cpu().numpy()
            batch_min = float(np.min(predictions))
            batch_max = float(np.max(predictions))
            batch_mean = float(np.mean(predictions))
            min_prob = min(min_prob, batch_min)
            max_prob = max(max_prob, batch_max)
            mean_acc += batch_mean
            processed_batches += 1
            predicted_image[start:end] = predictions[: end - start]
            confidence_image[start:end] = confidence_predictions[: end - start]

    img['predicted_image'] = predicted_image
    img['confidence_image'] = confidence_image
    if processed_batches > 0:
        img['_prediction_stats'] = {
            'min': float(min_prob),
            'max': float(max_prob),
            'mean': float(mean_acc / processed_batches),
            'non_finite': int(non_finite_values),
            'fp32_retries': int(fp32_retries),
        }
    else:
        img['_prediction_stats'] = {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'non_finite': int(non_finite_values),
            'fp32_retries': int(fp32_retries),
        }
    return img


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
        output_path = sew(
            output_dir,
            restored_item,
            jpeg_quality=jpeg_quality,
            threshold=threshold,
            postprocess_kernel_size=postprocess_kernel_size,
            confidence_save_mode=confidence_save_mode,
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
    )
    sewed_queue.put(
        {
            'name': item['name'],
            'source_path': item.get('source_path'),
            'output_path': output_path,
        }
    )


def sew(
    save_dir: Path | str,
    item: dict[str, Any],
    jpeg_quality: int = 95,
    *,
    threshold: float | None = None,
    postprocess_kernel_size: int = 0,
    confidence_save_mode: str = 'off',
) -> Path:
    output_name = '.'.join(item['name'].split('.')[:-1]) + '.jpg'
    output_path = os.path.join(str(save_dir), output_name)
    image = cast(
        Image.Image,
        sew_image(
            base_image=item['baseim_size'],
            predictions=item['predicted_image'],
            overlap=item['overlap'],
            threshold=threshold,
            postprocess_kernel_size=postprocess_kernel_size,
        ),
    )
    quality = max(1, min(100, int(jpeg_quality)))
    image.save(output_path, format='JPEG', quality=quality)
    if str(confidence_save_mode).strip().lower() != 'separate_grayscale':
        return Path(output_path)
    confidence_predictions = item.get('confidence_image')
    if confidence_predictions is None:
        return Path(output_path)
    confidence_name = '.'.join(item['name'].split('.')[:-1]) + '_confidence.jpg'
    confidence_path = os.path.join(str(save_dir), confidence_name)
    confidence_image = cast(
        Image.Image,
        sew_image(
            base_image=item['baseim_size'],
            predictions=confidence_predictions,
            overlap=item['overlap'],
            threshold=None,
            postprocess_kernel_size=0,
        ),
    )
    confidence_image.save(confidence_path, format='JPEG', quality=quality)
    return Path(output_path)


def _load_preview_array(path: Path | str | None) -> np.ndarray | None:
    if path is None:
        return None
    try:
        with retry_file_read(lambda: Image.open(path), path=path) as source_image:
            if source_image.mode == 'L':
                preview_image = source_image.copy()
            elif source_image.mode == 'RGB':
                preview_image = source_image.copy()
            else:
                preview_image = source_image.convert('RGB')
    except Exception:
        return None

    preview_array = np.asarray(preview_image)
    if preview_array.ndim == 2:
        return np.ascontiguousarray(preview_array.astype(np.uint8, copy=False))
    if preview_array.ndim == 3 and preview_array.shape[2] >= 3:
        if preview_array.shape[2] != 3:
            preview_array = preview_array[:, :, :3]
        return np.ascontiguousarray(preview_array.astype(np.uint8, copy=False))
    return None


def _publish_recognition_preview(
    *,
    publish: Publisher,
    source_path: Path | str | None,
    output_path: Path | str | None,
    frame_name: str,
) -> None:
    original_image = _load_preview_array(source_path)
    output_image = _load_preview_array(output_path)
    if original_image is None and output_image is None:
        return
    publish(
        'metrics',
        {
            'type': 'recognition_preview',
            'sample_name': str(frame_name).strip(),
            'image': original_image,
            'outputs': output_image,
        },
    )


def _store_payload_array_for_multiprocessing(payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, np.ndarray):
        return
    contiguous = np.ascontiguousarray(value)
    if _should_use_torch_tensor_transport():
        payload[key] = torch.from_numpy(contiguous)
        return
    payload[key] = contiguous


def _restore_payload_array_from_multiprocessing(payload: dict[str, Any], key: str) -> dict[str, Any]:
    restored_payload = dict(payload)
    value = restored_payload.get(key)
    if torch.is_tensor(value):
        restored_payload[key] = value.numpy()
    return restored_payload


def _should_use_torch_tensor_transport() -> bool:
    return os.name != 'nt'
