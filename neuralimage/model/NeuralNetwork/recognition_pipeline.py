from __future__ import annotations

import datetime
import os
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterable, cast

import numpy as np
import torch
import torch.nn as nn
from multiprocessing.synchronize import Event as MpEvent
from PIL import Image

from lib.image_processing import cut_image, sew_image
from model.NeuralNetwork.model_io import load_model_artifact


Publisher = Callable[[str, Any], None]
MemoryMetricsCollector = Callable[[], dict[str, float] | None]


@dataclass(frozen=True)
class RecognitionWorkload:
    source_files: list[Path]
    result_folder: Path
    part_size: tuple[int, int]
    overlap: int
    batch_size: int
    colors: int
    devices: list[torch.device]
    model_source: str | Path

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

    def close(self) -> None:
        for queue in (self.cut, self.predict, self.sew, self.sewed):
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
        self._started_at = datetime.datetime.now()
        self._last_frame_at = time.perf_counter()

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
        frame_elapsed = round(time.perf_counter() - self._last_frame_at, 3)
        elapsed = datetime.datetime.now() - self._started_at
        elapsed = elapsed - datetime.timedelta(microseconds=elapsed.microseconds)
        self._publish(
            'logging',
            (
                f'Frame: {int(current_frame)}/{int(self._total_frames)}. '
                f'Per-frame time: {frame_elapsed} sec. Elapsed: {elapsed}'
            ),
        )
        self._publish(
            'metrics',
            {
                'type': 'recognition_progress',
                'current': int(current_frame),
                'total': int(self._total_frames),
            },
        )
        self._last_frame_at = time.perf_counter()


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
        groups = WorkerGroups(cut=[], predict=[], sew=[])
        predict_stopped = False
        sew_stopped = False
        try:
            self._prime_cut_queue(queues)
            self._publish_runtime_plan()
            groups = self._start_workers(queues)
            predict_stopped, sew_stopped = self._monitor_processing(groups=groups, queues=queues)
        finally:
            self._send_missing_stop_tokens(
                queues=queues,
                predict_stopped=predict_stopped,
                sew_stopped=sew_stopped,
            )
            self._shutdown_workers(groups)
            queues.close()

    def _create_queues(self) -> PipelineQueues:
        predict_queue_size = max(4, len(self._config.workload.devices) * 4)
        return PipelineQueues(
            cut=mp.Queue(),
            predict=mp.Queue(maxsize=predict_queue_size),
            sew=mp.Queue(maxsize=predict_queue_size),
            sewed=mp.Queue(),
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
                target=cut_image_process,
                args=(
                    queues.cut,
                    queues.predict,
                    self._config.workload.segment_shape,
                    self._config.workload.overlap,
                    self._stop_event,
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
                target=imgpredict,
                args=(
                    queues.predict,
                    queues.sew,
                    self._config.workload.model_source,
                    device,
                    self._config.workload.batch_size,
                    self._stop_event,
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
                target=imgsew,
                args=(
                    self._config.workload.result_folder,
                    queues.sew,
                    queues.sewed,
                    self._stop_event,
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
    ) -> tuple[bool, bool]:
        completed = 0
        predict_stopped = False
        sew_stopped = False
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
            if self._has_failed_process(groups):
                self._stop_event.set()
                break

        return predict_stopped, sew_stopped

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
        try:
            queues.sewed.get(timeout=0.2)
        except Empty:
            return completed
        updated = completed + 1
        self._reporter.publish_frame(updated)
        return updated

    def _has_failed_process(self, groups: WorkerGroups) -> bool:
        for process in groups.all():
            if process.exitcode in (None, 0):
                continue
            self._publish(
                'logging',
                f'Child process failed. pid={process.pid}, code={process.exitcode}.',
            )
            return True
        return False

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
        prepared = cut_image_prepare(image_path, shape, overlap)
        predicted = gpu_predict(prepared, model, device, batch_size)
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
                fallback_predicted = gpu_predict(prepared, original_model, device, batch_size)
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
        sew(result_folder, predicted)
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
    stop_token: str = '__STOP__',
) -> None:
    while not stop_event.is_set():
        try:
            image_path = cut_queue.get(timeout=0.2)
        except Empty:
            continue
        if image_path == stop_token:
            break
        image_payload = cut_image_prepare(cast(Path, image_path), size, overlap)
        cutted_queue.put(image_payload)


def cut_image_prepare(img_path: Path, segment_size: tuple[int, int, int], overlap: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'baseim_size': None,
        'segment_size': segment_size,
        'overlap': overlap,
        'cutted_image': None,
        'name': img_path.name,
    }

    Image.MAX_IMAGE_PIXELS = None
    channels = segment_size[0]
    with Image.open(img_path) as source_image:
        if source_image.mode != 'L' and channels == 1:
            source_image = source_image.convert('L')
        payload['baseim_size'] = source_image.size
        work_image = np.array(source_image).astype('float32')

    work_image = _to_channel_first(work_image, channels=channels)
    payload['cutted_image'] = cut_image(work_image, segment_size, overlap)
    return payload


def get_array_from_image(path: Path | str, channels: int) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as source_image:
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
    stop_token: str = '__STOP__',
) -> None:
    model = load_model_artifact(model_path, map_location='cpu')
    model.eval()
    model.to(gpu)
    if gpu.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    while not stop_event.is_set():
        try:
            item = prediction_queue.get(timeout=0.2)
        except Empty:
            continue
        if item == stop_token:
            break
        predicted = gpu_predict(item, model, gpu, batch_size)
        predicted.pop('cutted_image', None)
        predicted_queue.put(predicted)


def gpu_predict(img: dict[str, Any], model: nn.Module, device: torch.device, batch_size: int) -> dict[str, Any]:
    predicted_image = np.empty_like(img['cutted_image'])
    parts_in_image = len(img['cutted_image'])

    tensor_data = torch.from_numpy(img['cutted_image']).float()
    use_amp = device.type == 'cuda'
    min_prob = float('inf')
    max_prob = float('-inf')
    mean_acc = 0.0
    processed_batches = 0
    non_finite_values = 0
    fp32_retries = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(create_batches(tensor_data, batch_size)):
            batch = batch.to(device, non_blocking=use_amp)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(batch)

            if not bool(torch.isfinite(outputs).all()):
                fp32_retries += 1
                outputs = model(batch.float())

            finite_mask = torch.isfinite(outputs)
            if not bool(finite_mask.all()):
                non_finite_values += int((~finite_mask).sum().item())
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)

            outputs = torch.sigmoid(outputs)
            if not bool(torch.isfinite(outputs).all()):
                finite_after_sigmoid = torch.isfinite(outputs)
                non_finite_values += int((~finite_after_sigmoid).sum().item())
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)

            predictions = outputs.detach().cpu().numpy()
            batch_min = float(np.min(predictions))
            batch_max = float(np.max(predictions))
            batch_mean = float(np.mean(predictions))
            min_prob = min(min_prob, batch_min)
            max_prob = max(max_prob, batch_max)
            mean_acc += batch_mean
            processed_batches += 1
            start = batch_size * batch_index
            end = min(start + batch_size, parts_in_image)
            predicted_image[start:end] = predictions[: end - start]

    img['predicted_image'] = predicted_image
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
    stop_event: MpEvent,
    stop_token: str = '__STOP__',
) -> None:
    while not stop_event.is_set():
        try:
            item = sew_queue.get(timeout=0.2)
        except Empty:
            continue
        if item == stop_token:
            break
        sew(output_dir, item)
        sewed_queue.put(item['name'])


def sew_from_queue(output_dir: Path | str, sew_queue: mp.Queue, sewed_queue: mp.Queue) -> None:
    item = sew_queue.get()
    sew(output_dir, item)
    sewed_queue.put(item['name'])


def sew(save_dir: Path | str, item: dict[str, Any]) -> None:
    output_name = '.'.join(item['name'].split('.')[:-1]) + '.jpg'
    output_path = os.path.join(str(save_dir), output_name)
    image = cast(
        Image.Image,
        sew_image(
            base_image=item['baseim_size'],
            predictions=item['predicted_image'],
            overlap=item['overlap'],
        ),
    )
    image.save(output_path)
