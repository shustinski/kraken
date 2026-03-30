import importlib
import os
import datetime
import time
import socket
import sys
import importlib.util
import math
import random
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

from collections import deque
from collections.abc import Callable, Mapping, Sized
from contextlib import nullcontext
from typing import Any, ContextManager, Protocol, cast

import multiprocessing as mp
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, ImageDraw, ImageFont

from lib import System
from model.NeuralNetwork.dataset import CustomDataset, NoCutDataset, index_in_list
from lib.data_interfaces import (
    CutoutParameters,
    RecognitionParameters,
    OptimizerParameters,
    OptimizerName,
    MixupParameters,
    MixedPrecisionMode,
    EarlyStoppingParameters,
    RandomArtifactsParameters,
    SchedulerName,
    SchedulerParameters,
    WarmupParameters,
    HardMiningParameters,
    normalize_multi_gpu_mode,
)
from lib.file_func import filter_images
from lib.func import get_input_channels
from lib.images import ImagePreparator, SampleCalculator
from lib.loss_config import format_loss_formula, resolve_loss_term_weights, sanitize_loss_term_weights
from lib.message_bus import AbstractMessageBus
from lib.random_artifacts import generate_random_artifact_patch
from model.NeuralNetwork.context_utils import normalize_size_pair
from model.NeuralNetwork.model_io import load_model_artifact, save_model_artifact
from model.NeuralNetwork.recognition_pipeline import (
    RecognitionWorkload,
    WorkerCounts,
    cut_image_prepare as _cut_image_prepare,
    create_batches as _create_batches,
    cut_image_process as _cut_image_process,
    get_array_from_image as _get_array_from_image,
    gpu_predict as _gpu_predict,
    imgpredict as _imgpredict,
    imgsew as _imgsew,
    run_multiprocessing_recognition,
    run_single_thread_recognition,
    sew as _sew,
    sew_from_queue as _sew_from_queue,
)

CHECKPOINT_SUFFIX = '.ckpt'
STOP_TOKEN = '__STOP__'
# Global profiler switch (set in code, not via env var).
TRAINING_PROFILER_ENABLED = False
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
FOCAL_TVERSKY_ALPHA = 0.3
FOCAL_TVERSKY_BETA = 0.7
BOUNDARY_LOSS_KERNEL_SIZE = 3
CLDICE_SKELETON_ITERATIONS = 16
VALIDATION_THRESHOLD_CANDIDATES: tuple[float, ...] = tuple(value / 100.0 for value in range(10, 95, 5))
RECOGNITION_AUX_WORKERS_PER_GPU = 4
VALIDATION_EXPORT_CACHE_MAX_BYTES = 512 * 1024 * 1024
_ARTIFACT_NAME_SANITIZE_PATTERN = re.compile(r'[^A-Za-z0-9._-]+')
_RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET = 4
_RANDOM_ARTIFACT_BANK_BUCKET_GRANULARITY = 8
_RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC = 0.25


class _NoOpQueue:
    def put(self, item: Any) -> None:
        return


class _RecordingQueue:
    def __init__(self, queue: Any, log_lines: list[str]) -> None:
        self._queue = queue
        self._log_lines = log_lines

    def put(self, item: Any) -> None:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == 'logging':
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                for line in str(item[1]).splitlines() or ['']:
                    self._log_lines.append(f'[{timestamp}] {line}')
        except Exception:
            pass
        self._queue.put(item)


def _drain_process_queue(
    queue: Any,
    publish_message: Callable[[Any], None],
) -> None:
    if queue is None:
        return
    reader = getattr(queue, '_reader', None)
    poll_fn = getattr(reader, 'poll', None)
    get_fn = getattr(queue, 'get', None)
    if callable(poll_fn) and callable(get_fn):
        while True:
            try:
                if not bool(poll_fn(0)):
                    break
                queued_message = get_fn()
            except (Empty, EOFError, OSError, ValueError):
                break
            publish_message(queued_message)
        return
    get_nowait = getattr(queue, 'get_nowait', None)
    if callable(get_nowait):
        while True:
            try:
                queued_message = get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            publish_message(queued_message)
        return
    empty_fn = getattr(queue, 'empty', None)
    if not callable(empty_fn) or not callable(get_fn):
        return
    while True:
        try:
            if bool(empty_fn()):
                break
            queued_message = get_fn()
        except Empty:
            break
        except (EOFError, OSError, ValueError):
            break
        publish_message(queued_message)


def _join_process_with_escalation(process: Any, *, join_timeout: float = 5.0) -> None:
    if process is None:
        return
    try:
        process.join(timeout=join_timeout)
    except Exception:
        return
    if not process.is_alive():
        return
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.join(timeout=join_timeout)
    except Exception:
        return
    if not process.is_alive():
        return
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.join(timeout=join_timeout)
    except Exception:
        pass


def _ddp_worker_entry(rank: int, trainer: 'TrainerProcess', world_size: int, master_port: int) -> None:
    trainer._run_ddp_worker(rank=rank, world_size=world_size, master_port=master_port)


class _SupportsSetEpoch(Protocol):
    def set_epoch(self) -> None:
        ...


class _SupportsLossAwareSampling(Protocol):
    strength: float
    ema_alpha: float

    def update_batch_losses(self, sample_indices: torch.Tensor, sample_losses: torch.Tensor) -> None:
        ...

    def resize(self, size: int, *, reset: bool = False) -> None:
        ...


@dataclass(frozen=True)
class _TrainLoopStrides:
    metric: int
    progress: int
    log: int
    preview: int


@dataclass
class _EpochStats:
    train_loss_sum: torch.Tensor | None = None
    train_samples_count: int = 0
    skipped_uniform_count: int = 0
    skipped_non_finite_count: int = 0
    data_wait_ms: float = 0.0
    augmentation_ms: float = 0.0
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    total_ms: float = 0.0

    def add_batch(
        self,
        *,
        batch_samples: int,
        batch_loss: torch.Tensor | float,
        data_wait_ms: float,
        augmentation_ms: float,
        forward_ms: float,
        backward_ms: float,
        optimizer_ms: float,
        total_ms: float,
    ) -> None:
        if torch.is_tensor(batch_loss):
            detached_loss = batch_loss.detach().to(dtype=torch.float32)
        else:
            detached_loss = torch.tensor(float(batch_loss), dtype=torch.float32)
        scaled_loss = detached_loss * float(batch_samples)
        self.train_loss_sum = scaled_loss if self.train_loss_sum is None else (self.train_loss_sum + scaled_loss)
        self.train_samples_count += int(batch_samples)
        self.data_wait_ms += data_wait_ms
        self.augmentation_ms += augmentation_ms
        self.forward_ms += forward_ms
        self.backward_ms += backward_ms
        self.optimizer_ms += optimizer_ms
        self.total_ms += total_ms


@dataclass
class _EarlyStoppingState:
    best_loss: float | None = None
    bad_epochs: int = 0
    best_epoch: int = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_threshold: float = 0.5


@dataclass(frozen=True)
class _EarlyStoppingConfig:
    enabled: bool
    patience: int
    min_delta: float


@dataclass(frozen=True)
class _RunContext:
    bce_criterion: nn.Module
    optimizer: Any
    scaler: Any
    autocast_ctx: Callable[[], ContextManager[Any]]
    scheduler: Any
    train_size: int
    train_sampler: Any
    supports_loss_aware_sampling: bool
    strides: _TrainLoopStrides
    scheduler_step_mode: str = 'none'
    warmup_scheduler: Any = None
    warmup_total_steps: int = 0


@dataclass(frozen=True)
class _TrainingProfilerConfig:
    enabled: bool
    max_batches: int
    record_shapes: bool
    profile_memory: bool
    with_stack: bool
    row_limit: int
    output_dir_name: str


@dataclass
class _ActiveTrainProfiler:
    profiler: Any
    steps_left: int
    trace_path: Path
    summary_sort_key: str
    row_limit: int
    is_closed: bool = False


@dataclass
class _PreparedTrainBatch:
    data: Any
    target: Any
    sample_indices: Any
    mixup_pair_indices: Any
    mixup_lambdas: torch.Tensor | None
    inputs: Any
    image: torch.Tensor
    context_image: torch.Tensor | None
    label: torch.Tensor
    batch_start: float
    data_wait_ms: float
    augmentation_ms: float = 0.0


@dataclass
class _TrainStepResult:
    outputs: torch.Tensor
    per_sample_loss: torch.Tensor
    metric_loss: torch.Tensor | float | None = None
    batch_loss: torch.Tensor | float | None = None
    batch_samples: int = 0
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.metric_loss is None and self.batch_loss is None:
            raise ValueError('Train step result requires metric_loss or batch_loss.')
        if self.metric_loss is None:
            self.metric_loss = self.batch_loss
        if self.batch_loss is None:
            self.batch_loss = self.metric_loss


@dataclass(frozen=True)
class _TrainingRuntimeState:
    is_main_process: bool
    run_context: _RunContext
    start_epoch: int
    early_stopping_state: _EarlyStoppingState
    early_stopping_config: _EarlyStoppingConfig
    active_profiler: _ActiveTrainProfiler | None


class _RandomArtifactBank:
    def __init__(
        self,
        *,
        channels: int,
        artifact_types: tuple[str, ...],
        target_per_bucket: int = _RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET,
        bucket_granularity: int = _RANDOM_ARTIFACT_BANK_BUCKET_GRANULARITY,
        base_seed: int | None = None,
    ) -> None:
        self._channels = max(1, int(channels))
        self._artifact_types = tuple(artifact_types)
        self._target_per_bucket = max(1, int(target_per_bucket))
        self._bucket_granularity = max(1, int(bucket_granularity))
        self._base_seed = int(base_seed if base_seed is not None else (torch.initial_seed() & 0x7FFFFFFF))
        self._seed_counter = 0
        self._cache: dict[tuple[int, int], deque[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._pending_requests: dict[tuple[int, int], int] = {}
        self._request_queue: deque[tuple[int, int]] = deque()
        self._condition = threading.Condition()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def channels(self) -> int:
        return int(self._channels)

    @property
    def artifact_types(self) -> tuple[str, ...]:
        return tuple(self._artifact_types)

    def matches(self, *, channels: int, artifact_types: tuple[str, ...]) -> bool:
        return int(channels) == int(self._channels) and tuple(artifact_types) == tuple(self._artifact_types)

    def start(self, prewarm_buckets: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ()) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._worker_loop,
                name='random-artifact-bank',
                daemon=True,
            )
            self._thread.start()
            for bucket in prewarm_buckets:
                self._ensure_bucket_target_locked(self._normalize_bucket(*bucket))
            self._condition.notify_all()

    def stop(self, *, join_timeout: float = 1.0) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(join_timeout)))
        with self._condition:
            self._thread = None
            self._cache.clear()
            self._pending_requests.clear()
            self._request_queue.clear()

    def wait_until_ready(self, timeout: float) -> bool:
        return bool(self._ready_event.wait(timeout=max(0.0, float(timeout))))

    def acquire(self, *, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        requested_height = max(1, int(height))
        requested_width = max(1, int(width))
        requested_bucket = self._normalize_bucket(requested_height, requested_width)
        entry, source_bucket = self._try_take_entry(requested_bucket)
        if entry is None:
            self.wait_until_ready(timeout=_RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC)
            entry, source_bucket = self._try_take_entry(requested_bucket)
        if entry is None:
            source_bucket = requested_bucket
            entry = self._generate_bucket_entry(requested_bucket)
        overlay, alpha = entry
        with self._condition:
            self._ensure_bucket_target_locked(requested_bucket)
            if source_bucket is not None:
                self._ensure_bucket_target_locked(source_bucket)
            self._condition.notify_all()
        if overlay.shape[-2:] != (requested_height, requested_width):
            overlay = F.interpolate(
                overlay.unsqueeze(0),
                size=(requested_height, requested_width),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            alpha = F.interpolate(
                alpha.unsqueeze(0),
                size=(requested_height, requested_width),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            alpha = torch.clamp(alpha, min=0.0, max=1.0)
        return overlay, alpha

    def _next_seed(self) -> int:
        with self._condition:
            seed = (self._base_seed + self._seed_counter) & 0x7FFFFFFF
            self._seed_counter += 1
        return int(seed)

    def _normalize_bucket(self, height: int, width: int) -> tuple[int, int]:
        granularity = int(self._bucket_granularity)
        bucket_height = max(granularity, int(math.ceil(max(1, int(height)) / granularity) * granularity))
        bucket_width = max(granularity, int(math.ceil(max(1, int(width)) / granularity) * granularity))
        return int(bucket_height), int(bucket_width)

    def _ensure_bucket_target_locked(self, bucket: tuple[int, int]) -> None:
        cache = self._cache.setdefault(bucket, deque())
        pending = int(self._pending_requests.get(bucket, 0))
        required = max(0, int(self._target_per_bucket) - (len(cache) + pending))
        for _ in range(required):
            self._request_queue.append(bucket)
            pending += 1
        self._pending_requests[bucket] = pending

    def _try_take_entry(
        self,
        requested_bucket: tuple[int, int],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, tuple[int, int] | None]:
        with self._condition:
            self._ensure_bucket_target_locked(requested_bucket)
            exact_cache = self._cache.get(requested_bucket)
            if exact_cache:
                entry = exact_cache.popleft()
                return entry, requested_bucket
            nearest_bucket: tuple[int, int] | None = None
            nearest_distance: int | None = None
            for bucket, cache in self._cache.items():
                if not cache:
                    continue
                distance = abs(int(bucket[0]) - int(requested_bucket[0])) + abs(int(bucket[1]) - int(requested_bucket[1]))
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_bucket = bucket
            if nearest_bucket is None:
                return None, None
            entry = self._cache[nearest_bucket].popleft()
            return entry, nearest_bucket

    def _generate_bucket_entry(self, bucket: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        return generate_random_artifact_patch(
            channels=int(self._channels),
            height=int(bucket[0]),
            width=int(bucket[1]),
            device=torch.device('cpu'),
            dtype=torch.float32,
            artifact_types=self._artifact_types,
            seed=self._next_seed(),
        )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while not self._stop_event.is_set() and not self._request_queue:
                    self._condition.wait(timeout=0.1)
                if self._stop_event.is_set():
                    return
                bucket = self._request_queue.popleft()
                self._pending_requests[bucket] = max(0, int(self._pending_requests.get(bucket, 0)) - 1)
            entry = self._generate_bucket_entry(bucket)
            with self._condition:
                if self._stop_event.is_set():
                    return
                self._cache.setdefault(bucket, deque()).append(entry)
                self._ready_event.set()
                self._condition.notify_all()


@dataclass
class _ValidationNoCutFrameExportCache:
    frame_index: int
    image_path: Path
    baseim_size: tuple[int, int]
    overlap: int
    parts_count: int
    part_lookup: list[int] | None
    patches: dict[int, np.ndarray]


@dataclass
class _ValidationExportCache:
    mode: str
    dataset: Any
    sample_predictions: dict[int, np.ndarray] | None = None
    frame_predictions: dict[int, _ValidationNoCutFrameExportCache] | None = None


def _release_torch_memory() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        # Best-effort release; ignore cleanup failures.
        pass


def _collect_memory_metrics() -> dict[str, float] | None:
    ram_mb: float | None = None
    try:
        import psutil  # type: ignore

        ram_mb = float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        ram_mb = None

    vram_alloc_mb: float | None = None
    vram_reserved_mb: float | None = None
    if torch.cuda.is_available():
        try:
            vram_alloc_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            vram_reserved_mb = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
        except Exception:
            vram_alloc_mb = None
            vram_reserved_mb = None

    if ram_mb is None and vram_alloc_mb is None and vram_reserved_mb is None:
        return None

    payload: dict[str, float] = {}
    if ram_mb is not None:
        payload['ram_mb'] = ram_mb
    if vram_alloc_mb is not None:
        payload['vram_allocated_mb'] = vram_alloc_mb
    if vram_reserved_mb is not None:
        payload['vram_reserved_mb'] = vram_reserved_mb
    return payload


def _is_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _get_torch_compile_unavailable_reason(target_device_type: str) -> str | None:
    if target_device_type != 'cuda':
        return None
    if not _is_module_available('triton'):
        return 'Triton is not installed for CUDA backend.'
    try:
        from torch.utils import _triton as torch_triton
    except Exception:
        torch_triton = None

    if torch_triton is not None:
        has_triton_package = getattr(torch_triton, 'has_triton_package', None)
        if callable(has_triton_package):
            try:
                if not bool(has_triton_package()):
                    return 'Triton package is incompatible with this PyTorch build.'
                return None
            except Exception as error:
                return f'Triton compatibility probe failed: {error}'

    try:
        from triton.compiler.compiler import triton_key
    except Exception as error:
        return f'Triton package is incompatible with this PyTorch build: {error}'
    if triton_key is None:
        return 'Triton package is incompatible with this PyTorch build.'
    return None


def _is_debugger_attached() -> bool:
    gettrace = getattr(sys, 'gettrace', None)
    if not callable(gettrace):
        return False
    try:
        return bool(gettrace())
    except Exception:
        return False


_TORCH_COMPILE_MODES = {'default', 'reduce-overhead', 'max-autotune'}
_MAX_AUTOTUNE_MIN_SMS = 68


def _resolve_torch_compile_mode(
    target_device_type: str,
    device: torch.device | None = None,
) -> tuple[str, str]:
    raw_mode = str(os.getenv('NEURALIMAGE_TORCH_COMPILE_MODE', '')).strip().lower()
    if raw_mode in _TORCH_COMPILE_MODES:
        return raw_mode, 'env'

    if target_device_type != 'cuda' or not torch.cuda.is_available():
        return 'default', 'device'

    try:
        if device is not None and device.type == 'cuda' and device.index is not None:
            props = torch.cuda.get_device_properties(device.index)
        else:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm_count = int(getattr(props, 'multi_processor_count', 0))
    except Exception:
        return 'default', 'fallback'

    if sm_count < _MAX_AUTOTUNE_MIN_SMS:
        return 'reduce-overhead', f'sm={sm_count}'
    return 'max-autotune', f'sm={sm_count}'


class ModelTrainer(threading.Thread):
    def __init__(self,train_dataloader:DataLoader, val_dataloader:DataLoader | None,
                 model:nn.Module, save_path:Path,epochs:int, message_bus:AbstractMessageBus,
                 callback:Callable[..., None]|None = None,
                 optimizer_params: OptimizerParameters | None = None,
                 mixed_precision: MixedPrecisionMode = MixedPrecisionMode.bf16,
                 loss_function: str = 'bce',
                 loss_term_weights: dict[str, float] | None = None,
                 dice_loss_weight: float = 0.5,
                 iou_loss_weight: float = 0.5,
                 hard_mining_params: HardMiningParameters | None = None,
                 cutout_params: CutoutParameters | None = None,
                 random_artifacts_params: RandomArtifactsParameters | None = None,
                 mixup_params: MixupParameters | None = None,
                 early_stopping_params: EarlyStoppingParameters | None = None,
                 warmup_params: WarmupParameters | None = None,
                 scheduler_params: SchedulerParameters | None = None,
                 skip_uniform_labels: bool = False,
                 resume_from_checkpoint: bool = False,
                 use_multi_gpu: bool = True,
                 multi_gpu_mode: str | None = None,
                 show_batch_preview: bool = True,
                 log_update_frequency: int = 0,
                 save_validation_binary_images: bool = False):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._model = model
        self._save_path = save_path
        self._epochs = epochs
        self._bus = message_bus
        self.callback = callback
        self._optimizer_params = optimizer_params or OptimizerParameters()
        self._mixed_precision = mixed_precision
        self._loss_function = str(loss_function or 'bce').strip().lower()
        self._loss_term_weights = sanitize_loss_term_weights(loss_term_weights)
        self._dice_loss_weight = float(dice_loss_weight)
        self._iou_loss_weight = float(iou_loss_weight)
        self._hard_mining_params = hard_mining_params or HardMiningParameters()
        self._cutout_params = cutout_params or CutoutParameters()
        self._random_artifacts_params = random_artifacts_params or RandomArtifactsParameters()
        self._mixup_params = mixup_params or MixupParameters()
        self._early_stopping_params = early_stopping_params or EarlyStoppingParameters()
        self._warmup_params = warmup_params or WarmupParameters()
        self._scheduler_params = scheduler_params or SchedulerParameters()
        self._skip_uniform_labels = bool(skip_uniform_labels)
        self._resume_from_checkpoint = resume_from_checkpoint
        self._multi_gpu_mode = normalize_multi_gpu_mode(
            multi_gpu_mode,
            use_multi_gpu_fallback=bool(use_multi_gpu),
        )
        self._use_multi_gpu = self._multi_gpu_mode != 'off'
        self._show_batch_preview = show_batch_preview
        self._log_update_frequency = max(0, int(log_update_frequency))
        self._save_validation_binary_images = bool(save_validation_binary_images)
        self._stop_event = threading.Event()
        self.message_queue = mp.Queue()
        self.succeeded = False
        self.error_message: str | None = None
        self._recommended_inference_threshold = 0.5


    def _validate_training_inputs(self) -> bool:
        if self._train_dataloader is not None and self._model is not None:
            return True
        self.error_message = 'Ошибка: отсутствуют необходимые данные для обучения модели.'
        self._bus.publish('error', self.error_message)
        return False

    def _create_training_process(self) -> 'TrainerProcess':
        train_dataloader = cast(DataLoader, self._train_dataloader)
        val_dataloader = cast(DataLoader | None, self._val_dataloader)
        model = cast(nn.Module, self._model)
        return TrainerProcess(
            train_dataloader,
            val_dataloader,
            model,
            self._save_path,
            self._epochs,
            self.message_queue,
            optimizer_params=self._optimizer_params,
            mixed_precision=self._mixed_precision,
            loss_function=self._loss_function,
            loss_term_weights=self._loss_term_weights,
            dice_loss_weight=self._dice_loss_weight,
            iou_loss_weight=self._iou_loss_weight,
            hard_mining_params=self._hard_mining_params,
            cutout_params=self._cutout_params,
            random_artifacts_params=self._random_artifacts_params,
            mixup_params=self._mixup_params,
            early_stopping_params=self._early_stopping_params,
            warmup_params=self._warmup_params,
            scheduler_params=self._scheduler_params,
            skip_uniform_labels=self._skip_uniform_labels,
            resume_from_checkpoint=self._resume_from_checkpoint,
            use_multi_gpu=self._use_multi_gpu,
            multi_gpu_mode=self._multi_gpu_mode,
            show_batch_preview=self._show_batch_preview,
            log_update_frequency=self._log_update_frequency,
            save_validation_binary_images=self._save_validation_binary_images,
        )

    @staticmethod
    def _elapsed_suffix(since: float) -> str:
        elapsed_seconds = int(max(0.0, time.perf_counter() - since))
        elapsed_hours, remainder = divmod(elapsed_seconds, 3600)
        elapsed_minutes, elapsed_secs = divmod(remainder, 60)
        return f' Прошло: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}'

    def _publish_training_message(
        self,
        message: Any,
        *,
        append_elapsed_suffix: bool,
        started_at: float,
    ) -> None:
        topic, payload = message[0], message[1]
        if append_elapsed_suffix and isinstance(payload, str) and topic != 'error':
            payload = payload + self._elapsed_suffix(started_at)
        if topic == 'error' and isinstance(payload, str):
            self.error_message = payload
        self._bus.publish(topic, payload)

    def _drain_training_queue(self, *, append_elapsed_suffix: bool, started_at: float) -> None:
        def _publish(queued_message: Any) -> None:
            self._publish_training_message(
                queued_message,
                append_elapsed_suffix=append_elapsed_suffix,
                started_at=started_at,
            )
        _drain_process_queue(self.message_queue, _publish)

    def _finalize_training_result(self, training_process: Any) -> bool:
        if self._stop_event.is_set():
            self.succeeded = False
            return False
        exit_code = training_process.exitcode if training_process is not None else 1
        if exit_code not in (0, None):
            if self.error_message is None:
                self.error_message = f'Training error: process exited with code {exit_code}.'
                self._bus.publish('error', self.error_message)
            self.succeeded = False
            return False
        self.succeeded = True
        if self.callback is not None:
            self.callback()
        print('Model saved successfully!')
        return True

    def run(self):
        training_process: Any = None
        try:
            if not self._validate_training_inputs():
                return

            if _is_debugger_attached():
                self._bus.publish(
                    'logging',
                    'Debugger detected: training multiprocessing disabled for this run.',
                )
                training_process = self._create_training_process()
                started_at = time.perf_counter()
                try:
                    training_process.run()
                except Exception as error:
                    self.error_message = f'Training error: {error}'
                    self._bus.publish('error', self.error_message)
                    self.succeeded = False
                    return
                self._drain_training_queue(append_elapsed_suffix=False, started_at=started_at)
                self.succeeded = True
                if self.callback is not None:
                    self.callback()
                return

            training_process = self._create_training_process()
            training_process.start()

            started_at = time.perf_counter()
            while training_process.is_alive():
                if self._stop_event.is_set():
                    _join_process_with_escalation(training_process)
                    break
                self._drain_training_queue(
                    append_elapsed_suffix=True,
                    started_at=started_at,
                )
                time.sleep(1)

            _join_process_with_escalation(training_process)
            self._drain_training_queue(append_elapsed_suffix=False, started_at=started_at)
            self._finalize_training_result(training_process)
        finally:
            if training_process is not None and training_process.is_alive():
                _join_process_with_escalation(training_process)
            try:
                self.message_queue.close()
            except Exception:
                pass
            # Drop heavyweight references from parent thread after process completion.
            self._model = None
            self._train_dataloader = None
            self._val_dataloader = None
            _release_torch_memory()

    def stop(self):
        self._stop_event.set()

class TrainerProcess(mp.Process):
    def __init__(self,train_dataloader:DataLoader, val_dataloader:DataLoader | None,
                 model:nn.Module, save_path:Path, epochs:int, message_bus:mp.Queue,
                 callback:Callable[...,None]|None = None,
                 optimizer_params: OptimizerParameters | None = None,
                 mixed_precision: MixedPrecisionMode = MixedPrecisionMode.bf16,
                 loss_function: str = 'bce',
                 loss_term_weights: dict[str, float] | None = None,
                 dice_loss_weight: float = 0.5,
                 iou_loss_weight: float = 0.5,
                 hard_mining_params: HardMiningParameters | None = None,
                 cutout_params: CutoutParameters | None = None,
                 random_artifacts_params: RandomArtifactsParameters | None = None,
                 mixup_params: MixupParameters | None = None,
                 early_stopping_params: EarlyStoppingParameters | None = None,
                 warmup_params: WarmupParameters | None = None,
                 scheduler_params: SchedulerParameters | None = None,
                 skip_uniform_labels: bool = False,
                 resume_from_checkpoint: bool = False,
                 use_multi_gpu: bool = True,
                 multi_gpu_mode: str | None = None,
                 show_batch_preview: bool = True,
                 log_update_frequency: int = 0,
                 save_validation_binary_images: bool = False):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._model = model
        self._save_path = save_path
        self._epochs = epochs
        self._bus = message_bus
        self.callback = callback
        self._optimizer_params = optimizer_params or OptimizerParameters()
        self._mixed_precision = mixed_precision
        self._loss_function = str(loss_function or 'bce').strip().lower()
        self._loss_term_weights = sanitize_loss_term_weights(loss_term_weights)
        self._dice_loss_weight = float(dice_loss_weight)
        self._iou_loss_weight = float(iou_loss_weight)
        self._hard_mining_params = hard_mining_params or HardMiningParameters()
        self._cutout_params = cutout_params or CutoutParameters()
        self._random_artifacts_params = random_artifacts_params or RandomArtifactsParameters()
        self._mixup_params = mixup_params or MixupParameters()
        self._early_stopping_params = early_stopping_params or EarlyStoppingParameters()
        self._warmup_params = warmup_params or WarmupParameters()
        self._scheduler_params = scheduler_params or SchedulerParameters()
        self._skip_uniform_labels = bool(skip_uniform_labels)
        self._resume_from_checkpoint = resume_from_checkpoint
        self._multi_gpu_mode = normalize_multi_gpu_mode(
            multi_gpu_mode,
            use_multi_gpu_fallback=bool(use_multi_gpu),
        )
        self._use_multi_gpu = self._multi_gpu_mode != 'off'
        self._show_batch_preview = show_batch_preview
        self._log_update_frequency = max(0, int(log_update_frequency))
        self._save_validation_binary_images = bool(save_validation_binary_images)
        self._training_profiler_config = self._resolve_training_profiler_config()
        self._uncompiled_model: nn.Module | None = None
        self._torch_compile_active = False
        self._recommended_inference_threshold = 0.5
        self._train_epoch_history: list[tuple[float, float]] = []
        self._val_epoch_history: list[tuple[float, float]] = []
        self._val_iou_history: list[tuple[float, float]] = []
        self._val_dice_history: list[tuple[float, float]] = []
        self._batch_points_by_epoch: dict[int, list[tuple[float, float]]] = {}
        self._training_log_lines: list[str] = []
        self._bus = _RecordingQueue(message_bus, self._training_log_lines)
        self._random_artifact_bank: _RandomArtifactBank | None = None

    @property
    def _base_model(self) -> nn.Module:
        """Return the original model instance even if wrapped by DDP/DataParallel."""
        if isinstance(self._model, (DDP, nn.DataParallel)):
            return self._model.module
        return self._model

    def _resolve_model_artifact_metadata(self) -> tuple[str, int, dict[str, Any]]:
        base_model = self._base_model
        model_name = str(getattr(base_model, '_neuralimage_model_name', base_model.__class__.__name__))
        input_channels = getattr(base_model, '_neuralimage_input_channels', None)
        if input_channels is None:
            input_channels = get_input_channels(base_model)
        model_kwargs = getattr(base_model, '_neuralimage_model_kwargs', {})
        if not isinstance(model_kwargs, dict):
            model_kwargs = {}
        return model_name, int(input_channels), dict(model_kwargs)

    def _resolve_artifact_runtime_metadata(self) -> dict[str, Any]:
        threshold = float(min(max(getattr(self, '_recommended_inference_threshold', 0.5), 0.0), 1.0))
        return {
            'inference': {
                'recommended_threshold': threshold,
            },
        }

    def _save_model_artifact(self) -> None:
        model_name, input_channels, model_kwargs = self._resolve_model_artifact_metadata()
        save_model_artifact(
            self._base_model,
            self._save_path,
            model_name=model_name,
            input_channels=input_channels,
            model_kwargs=model_kwargs,
            metadata=self._resolve_artifact_runtime_metadata(),
        )

    @staticmethod
    def _sanitize_artifact_name(value: str, *, fallback: str = 'sample') -> str:
        cleaned = _ARTIFACT_NAME_SANITIZE_PATTERN.sub('_', str(value or '').strip()).strip('._-')
        return cleaned or fallback

    @staticmethod
    def _render_chart_image(
        *,
        save_path: Path,
        title: str,
        x_label: str,
        y_label: str,
        series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]],
        y_formatter: Callable[[float], str] | None = None,
    ) -> None:
        non_empty_series = [(label, points, color) for label, points, color in series if points]
        if not non_empty_series:
            return

        width, height = 1280, 720
        margin_left, margin_right = 100, 40
        margin_top, margin_bottom = 70, 90
        plot_left = margin_left
        plot_top = margin_top
        plot_right = width - margin_right
        plot_bottom = height - margin_bottom
        plot_width = max(1, plot_right - plot_left)
        plot_height = max(1, plot_bottom - plot_top)
        background_color = (255, 255, 255)
        axis_color = (40, 40, 40)
        grid_color = (222, 228, 236)
        text_color = (25, 25, 25)

        all_points = [point for _label, points, _color in non_empty_series for point in points]
        x_values = [float(point[0]) for point in all_points]
        y_values = [float(point[1]) for point in all_points]
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        if math.isclose(x_min, x_max):
            x_min -= 1.0
            x_max += 1.0
        if math.isclose(y_min, y_max):
            delta = 1.0 if math.isclose(y_min, 0.0) else abs(y_min) * 0.1
            y_min -= delta
            y_max += delta
        if y_min > 0.0:
            y_min = 0.0

        x_span = max(1e-9, x_max - x_min)
        y_span = max(1e-9, y_max - y_min)
        format_y = y_formatter or (lambda value: f'{float(value):.4f}')

        image = Image.new('RGB', (width, height), color=background_color)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        draw.text((plot_left, 24), str(title), fill=text_color, font=font)

        tick_count = 5
        for tick_index in range(tick_count + 1):
            x = plot_left + (plot_width * tick_index / tick_count)
            y = plot_top + (plot_height * tick_index / tick_count)
            draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
            draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)

            x_value = x_min + (x_span * tick_index / tick_count)
            y_value = y_max - (y_span * tick_index / tick_count)
            x_text = f'{x_value:.0f}' if abs(x_value - round(x_value)) < 1e-6 else f'{x_value:.2f}'
            y_text = format_y(y_value)
            draw.text((x - 12, plot_bottom + 12), x_text, fill=text_color, font=font)
            draw.text((12, y - 6), y_text, fill=text_color, font=font)

        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
        draw.text((plot_right - 50, plot_bottom + 40), str(x_label), fill=text_color, font=font)
        draw.text((16, plot_top - 24), str(y_label), fill=text_color, font=font)

        for series_index, (label, points, color) in enumerate(non_empty_series):
            scaled_points: list[tuple[float, float]] = []
            for x_value, y_value in points:
                normalized_x = (float(x_value) - x_min) / x_span
                normalized_y = (float(y_value) - y_min) / y_span
                scaled_x = plot_left + (normalized_x * plot_width)
                scaled_y = plot_bottom - (normalized_y * plot_height)
                scaled_points.append((scaled_x, scaled_y))
            if len(scaled_points) == 1:
                x_coord, y_coord = scaled_points[0]
                draw.ellipse((x_coord - 3, y_coord - 3, x_coord + 3, y_coord + 3), fill=color)
            else:
                draw.line(scaled_points, fill=color, width=3)
                for x_coord, y_coord in scaled_points:
                    draw.ellipse((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2), fill=color)

            legend_x = plot_right - 220
            legend_y = 24 + (series_index * 22)
            draw.line((legend_x, legend_y + 7, legend_x + 24, legend_y + 7), fill=color, width=3)
            draw.text((legend_x + 32, legend_y), str(label), fill=text_color, font=font)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)

    @staticmethod
    def _save_chart_csv(
        *,
        save_path: Path,
        x_label: str,
        series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]],
    ) -> None:
        non_empty_series = [(label, points) for label, points, _color in series if points]
        if not non_empty_series:
            return

        x_values = sorted({float(x_value) for _label, points in non_empty_series for x_value, _y_value in points})
        series_lookup: dict[str, dict[float, float]] = {
            str(label): {float(x_value): float(y_value) for x_value, y_value in points}
            for label, points in non_empty_series
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open('w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            labels = [str(label) for label, _points in non_empty_series]
            writer.writerow([str(x_label), *labels])
            for x_value in x_values:
                x_cell = int(x_value) if abs(x_value - round(x_value)) < 1e-9 else x_value
                row: list[Any] = [x_cell]
                for label in labels:
                    row.append(series_lookup[label].get(float(x_value), ''))
                writer.writerow(row)

    def _save_metric_charts(self) -> None:
        artifact_dir = self._save_path.parent
        if self._train_epoch_history or self._val_epoch_history:
            loss_chart_series = [
                ('Train Loss', list(self._train_epoch_history), (80, 186, 255)),
                ('Val Loss', list(self._val_epoch_history), (255, 170, 92)),
            ]
            self._render_chart_image(
                save_path=artifact_dir / 'training_metrics_loss_by_epoch.png',
                title='Loss vs Epoch',
                x_label='Epoch',
                y_label='Loss',
                series=loss_chart_series,
            )
            self._save_chart_csv(
                save_path=artifact_dir / 'training_metrics_loss_by_epoch.csv',
                x_label='Epoch',
                series=loss_chart_series,
            )
        if self._val_iou_history or self._val_dice_history:
            validation_quality_series = [
                ('IoU', list(self._val_iou_history), (142, 210, 110)),
                ('Dice', list(self._val_dice_history), (255, 111, 145)),
            ]
            self._render_chart_image(
                save_path=artifact_dir / 'training_metrics_validation_quality.png',
                title='Validation IoU / Dice vs Epoch',
                x_label='Epoch',
                y_label='Score',
                series=validation_quality_series,
                y_formatter=lambda value: f'{float(value) * 100.0:.1f}%',
            )
            self._save_chart_csv(
                save_path=artifact_dir / 'training_metrics_validation_quality.csv',
                x_label='Epoch',
                series=validation_quality_series,
            )
        if self._batch_points_by_epoch:
            last_epoch = max(self._batch_points_by_epoch)
            batch_chart_series = [
                (
                    'Train Loss',
                    list(self._batch_points_by_epoch.get(last_epoch, [])),
                    (137, 228, 125),
                ),
            ]
            self._render_chart_image(
                save_path=artifact_dir / f'training_metrics_train_loss_epoch_{int(last_epoch):04d}.png',
                title=f'Train Loss vs Batch (Epoch {int(last_epoch)})',
                x_label='Batch',
                y_label='Loss',
                series=batch_chart_series,
            )
            self._save_chart_csv(
                save_path=artifact_dir / f'training_metrics_train_loss_epoch_{int(last_epoch):04d}.csv',
                x_label='Batch',
                series=batch_chart_series,
            )

    @staticmethod
    def _format_elapsed_duration(seconds: float) -> str:
        total_seconds = int(max(0.0, round(float(seconds))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'

    def _save_training_log(self) -> Path:
        log_path = self._save_path.parent / 'training_log.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_text = '\n'.join(str(line) for line in self._training_log_lines)
        if log_text and not log_text.endswith('\n'):
            log_text += '\n'
        log_path.write_text(log_text, encoding='utf-8')
        return log_path

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    @staticmethod
    def _safe_loader_len(loader: DataLoader | None) -> int:
        if loader is None:
            return 0
        try:
            return int(len(loader))
        except Exception:
            return 0

    @staticmethod
    def _build_distributed_loader(base_loader: DataLoader, rank: int, world_size: int, shuffle: bool) -> DataLoader:
        dataset = base_loader.dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=bool(getattr(base_loader, 'drop_last', False)),
        )
        loader_kwargs: dict[str, Any] = {
            'dataset': dataset,
            'batch_size': int(getattr(base_loader, 'batch_size', 1) or 1),
            'sampler': sampler,
            'num_workers': int(getattr(base_loader, 'num_workers', 0)),
            'pin_memory': bool(getattr(base_loader, 'pin_memory', False)),
            'drop_last': bool(getattr(base_loader, 'drop_last', False)),
        }
        collate_fn = getattr(base_loader, 'collate_fn', None)
        if collate_fn is not None:
            loader_kwargs['collate_fn'] = collate_fn
        if loader_kwargs['num_workers'] > 0:
            prefetch_factor = getattr(base_loader, 'prefetch_factor', None)
            if prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = prefetch_factor
            loader_kwargs['persistent_workers'] = bool(getattr(base_loader, 'persistent_workers', False))
        return DataLoader(**loader_kwargs)

    def _has_multi_gpu_cuda(self) -> bool:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.device_count() > 1

    def _log_multi_gpu_unavailable(self, requested_mode: str) -> None:
        if requested_mode == 'off':
            self._bus.put(['logging', 'Multi-GPU mode is disabled in settings.'])
            return
        if not torch.cuda.is_available():
            self._bus.put(['logging', f'multi-GPU mode "{requested_mode}" skipped: CUDA is unavailable.'])
            return
        gpu_count = int(torch.cuda.device_count())
        if gpu_count <= 1:
            self._bus.put(['logging', f'multi-GPU mode "{requested_mode}" skipped: found {gpu_count} GPU.'])
            return
        if requested_mode == 'distributeddataparallel' and os.name == 'nt':
            self._bus.put(['logging', 'DDP requested but disabled on Windows in this build.'])
            return
        self._bus.put(['logging', f'multi-GPU mode "{requested_mode}" unavailable; using single GPU.'])

    def _should_use_ddp(self) -> bool:
        if self._multi_gpu_mode == 'off':
            self._bus.put(['logging', 'Режим multi GPU отключен в настройках.'])
            return False
        if self._multi_gpu_mode != 'distributeddataparallel':
            return False
        if not self._has_multi_gpu_cuda():
            return False
        if os.name == 'nt':
            # On Windows, DDP+gloo can hang during rendezvous on some setups.
            return False
        return True

    def _should_use_data_parallel(self) -> bool:
        if self._multi_gpu_mode != 'dataparallel':
            return False
        return self._has_multi_gpu_cuda()

    def _run_ddp(self, world_size: int) -> None:
        master_port = self._find_free_port()
        self._bus.put(['logging', f'Включен режим multi GPU: DistributedDataParallel на {world_size} GPU.'])
        torch_mp.spawn(
            _ddp_worker_entry,
            args=(self, world_size, master_port),
            nprocs=world_size,
            join=True,
        )

    def _run_ddp_worker(self, rank: int, world_size: int, master_port: int) -> None:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)
        backend = 'nccl'
        if os.name == 'nt' or not dist.is_nccl_available():
            backend = 'gloo'
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://127.0.0.1:{master_port}',
            timeout=datetime.timedelta(seconds=120),
        )
        try:
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
            if rank != 0:
                self._bus = _NoOpQueue()
                self._show_batch_preview = False

            if self._train_dataloader is None:
                raise RuntimeError('Train dataloader is not initialized for DDP.')

            sampler_obj = getattr(self._train_dataloader, 'sampler', None)
            train_shuffle = bool(getattr(self._train_dataloader, 'shuffle', False))
            if (not train_shuffle) and isinstance(sampler_obj, RandomSampler):
                train_shuffle = True
            self._train_dataloader = self._build_distributed_loader(
                self._train_dataloader,
                rank=rank,
                world_size=world_size,
                shuffle=train_shuffle,
            )
            if self._val_dataloader is not None:
                self._val_dataloader = self._build_distributed_loader(
                    self._val_dataloader,
                    rank=rank,
                    world_size=world_size,
                    shuffle=False,
                )

            self._model.to(device)
            self._try_compile_model(is_main_process=(rank == 0), device=device)
            self._model = DDP(self._model, device_ids=[rank], output_device=rank)
            self._run_impl(device=device, rank=rank, world_size=world_size, distributed=True)
        finally:
            dist.destroy_process_group()

    def _checkpoint_path(self) -> Path:
        return self._save_path.with_suffix(CHECKPOINT_SUFFIX)

    def _save_checkpoint(
        self,
        completed_epoch: int,
        optimizer,
        scaler,
        warmup_scheduler,
        lr_scheduler,
        early_stopping_best_loss: float | None = None,
        early_stopping_bad_epochs: int = 0,
        early_stopping_best_epoch: int = 0,
        early_stopping_best_model_state: dict[str, torch.Tensor] | None = None,
        early_stopping_best_threshold: float = 0.5,
    ) -> None:
        checkpoint = {
            'version': 1,
            'completed_epoch': completed_epoch,
            'epochs': self._epochs,
            'model_state_dict': self._base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'optimizer_name': self._optimizer_params.name.value,
            'learning_rate': self._optimizer_params.learning_rate,
            'weight_decay': self._optimizer_params.weight_decay,
            'warmup_scheduler_state_dict': warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'scheduler_state_dict': warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
            'early_stopping_best_loss': early_stopping_best_loss,
            'early_stopping_bad_epochs': early_stopping_bad_epochs,
            'early_stopping_best_epoch': early_stopping_best_epoch,
            'early_stopping_best_model_state': early_stopping_best_model_state,
            'early_stopping_best_threshold': float(min(max(early_stopping_best_threshold, 0.0), 1.0)),
            'recommended_inference_threshold': float(
                min(max(getattr(self, '_recommended_inference_threshold', 0.5), 0.0), 1.0)
            ),
        }
        torch.save(checkpoint, self._checkpoint_path())

    def _load_checkpoint_if_available(
        self,
        optimizer,
        scaler,
        warmup_scheduler,
        lr_scheduler,
    ) -> tuple[int, dict[str, Any]]:
        if not self._resume_from_checkpoint:
            return 0, {}

        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            self._bus.put(['logging', 'Файл контрольной точки не найден. Обучение начнется с первой эпохи.'])
            return 0, {}

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model_state = checkpoint.get('model_state_dict')
            optimizer_state = checkpoint.get('optimizer_state_dict')
            scaler_state = checkpoint.get('scaler_state_dict')
            legacy_scheduler_state = checkpoint.get('scheduler_state_dict')
            warmup_scheduler_state = checkpoint.get('warmup_scheduler_state_dict', legacy_scheduler_state)
            lr_scheduler_state = checkpoint.get('lr_scheduler_state_dict')
            completed_epoch = int(checkpoint.get('completed_epoch', 0))
            if model_state:
                self._base_model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            if warmup_scheduler is not None and warmup_scheduler_state:
                warmup_scheduler.load_state_dict(warmup_scheduler_state)
            if lr_scheduler is not None and lr_scheduler_state:
                lr_scheduler.load_state_dict(lr_scheduler_state)
            self._recommended_inference_threshold = float(
                min(
                    max(float(checkpoint.get('recommended_inference_threshold', 0.5)), 0.0),
                    1.0,
                )
            )

            start_epoch = max(0, min(completed_epoch, self._epochs))
            self._bus.put([
                'logging',
                f'Параметры восстановлены из контрольной точки {checkpoint_path.name}. Последняя завершенная эпоха: {completed_epoch}.',
            ])
            return start_epoch, checkpoint
        except Exception as error:
            self._bus.put(['logging', f'Ошибка загрузки контрольной точки: {error}. Обучение начнется с первой эпохи.'])
            return 0, {}

    def _resolved_scheduler_name(self) -> SchedulerName:
        scheduler_params = getattr(self, '_scheduler_params', None)
        scheduler_name = getattr(scheduler_params, 'name', SchedulerName.off)
        if isinstance(scheduler_name, SchedulerName):
            return scheduler_name
        try:
            return SchedulerName(str(scheduler_name or SchedulerName.off.value).strip().lower())
        except ValueError:
            return SchedulerName.off

    def _warmup_supported_with_scheduler(self) -> bool:
        return self._resolved_scheduler_name() != SchedulerName.one_cycle

    def _create_warmup_scheduler(self, optimizer, train_steps_per_epoch: int) -> tuple[Any, int]:
        warmup = self._warmup_params
        if not warmup.enabled:
            return None, 0
        if not self._warmup_supported_with_scheduler():
            self._bus.put([
                'logging',
                'Warmup disabled for OneCycleLR: use OneCycle pct_start to control the initial ramp phase.',
            ])
            return None, 0
        if train_steps_per_epoch <= 0:
            return None, 0

        warmup_epochs = max(1, int(warmup.epochs))
        warmup_steps = warmup_epochs * train_steps_per_epoch
        start_factor = float(min(max(warmup.start_factor, 0.0), 1.0))

        def lr_lambda(step: int) -> float:
            if step >= warmup_steps:
                return 1.0
            progress = (step + 1) / warmup_steps
            return start_factor + (1.0 - start_factor) * progress

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps

    @staticmethod
    def _resolve_one_cycle_anneal_strategy(value: str) -> str:
        normalized = str(value or 'cos').strip().lower()
        return normalized if normalized in {'cos', 'linear'} else 'cos'

    def _create_lr_scheduler(
        self,
        optimizer,
        *,
        train_steps_per_epoch: int,
    ) -> tuple[Any, str]:
        scheduler_params = getattr(self, '_scheduler_params', None) or SchedulerParameters()
        scheduler_name = self._resolved_scheduler_name()
        if scheduler_name == SchedulerName.off:
            return None, 'none'
        if scheduler_name == SchedulerName.reduce_on_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(min(max(getattr(scheduler_params, 'plateau_factor', 0.5), 1e-4), 0.9999)),
                patience=max(0, int(getattr(scheduler_params, 'plateau_patience', 3))),
                threshold=max(0.0, float(getattr(scheduler_params, 'plateau_threshold', 1e-4))),
                min_lr=max(0.0, float(getattr(scheduler_params, 'plateau_min_lr', 1e-6))),
                cooldown=max(0, int(getattr(scheduler_params, 'plateau_cooldown', 0))),
            )
            return scheduler, 'plateau'
        if scheduler_name == SchedulerName.cosine_annealing:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(getattr(scheduler_params, 'cosine_t_max', 10))),
                eta_min=max(0.0, float(getattr(scheduler_params, 'cosine_eta_min', 1e-6))),
            )
            return scheduler, 'epoch'
        if scheduler_name == SchedulerName.one_cycle:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max(0.0, float(getattr(scheduler_params, 'one_cycle_max_lr', 1e-3))),
                epochs=max(1, int(self._epochs)),
                steps_per_epoch=max(1, int(train_steps_per_epoch)),
                pct_start=float(
                    min(max(getattr(scheduler_params, 'one_cycle_pct_start', 0.3), 0.0), 1.0)
                ),
                anneal_strategy=self._resolve_one_cycle_anneal_strategy(
                    getattr(scheduler_params, 'one_cycle_anneal_strategy', 'cos')
                ),
                div_factor=max(1.0, float(getattr(scheduler_params, 'one_cycle_div_factor', 25.0))),
                final_div_factor=max(
                    1.0,
                    float(getattr(scheduler_params, 'one_cycle_final_div_factor', 10000.0)),
                ),
                three_phase=bool(getattr(scheduler_params, 'one_cycle_three_phase', False)),
            )
            return scheduler, 'batch'
        if scheduler_name == SchedulerName.step_lr:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, int(getattr(scheduler_params, 'step_lr_step_size', 10))),
                gamma=float(min(max(getattr(scheduler_params, 'step_lr_gamma', 0.1), 1e-4), 1.0)),
            )
            return scheduler, 'epoch'
        return None, 'none'

    def _step_batch_schedulers(self, run_context: _RunContext) -> None:
        warmup_scheduler = run_context.warmup_scheduler
        if warmup_scheduler is not None and int(run_context.warmup_total_steps) > 0:
            completed_steps = int(getattr(warmup_scheduler, 'last_epoch', -1)) + 1
            if completed_steps < int(run_context.warmup_total_steps):
                warmup_scheduler.step()
                return
        if run_context.scheduler is not None and run_context.scheduler_step_mode == 'batch':
            run_context.scheduler.step()

    def _step_epoch_scheduler(
        self,
        *,
        run_context: _RunContext,
        validation_result: dict[str, float] | None,
        train_loss: float | None,
        distributed: bool,
        device: torch.device,
        is_main_process: bool,
    ) -> None:
        scheduler = run_context.scheduler
        if scheduler is None:
            return
        if run_context.scheduler_step_mode == 'epoch':
            scheduler.step()
            return
        if run_context.scheduler_step_mode != 'plateau':
            return

        metric = validation_result.get('loss') if validation_result is not None else train_loss
        try:
            metric_value = float(metric) if metric is not None else None
        except (TypeError, ValueError):
            metric_value = None
        if metric_value is None or not math.isfinite(metric_value):
            return
        if distributed:
            metric_tensor = torch.tensor(
                [metric_value if is_main_process else 0.0],
                device=device,
                dtype=torch.float64,
            )
            dist.broadcast(metric_tensor, src=0)
            metric_value = float(metric_tensor.item())
        scheduler.step(metric_value)

    def _resolve_target_epochs(self, start_epoch: int) -> int:
        """
        In fine-tuning mode the UI epoch value is treated as "additional epochs".
        For fresh training it is treated as total epochs.
        """
        if self._resume_from_checkpoint and start_epoch > 0:
            return start_epoch + int(self._epochs)
        return int(self._epochs)

    def _resolved_loss_function(self) -> str:
        if self._loss_function in (
            'bce',
            'dice',
            'cldice',
            'bce_dice',
            'iou',
            'bce_iou',
            'ce',
            'ce_dice',
            'focal_bce',
            'focal_dice',
            'focal_iou',
            'boundary',
            'focal_tversky',
        ):
            return self._loss_function
        return 'bce'

    def _resolved_dice_weight(self) -> float:
        return float(min(max(self._dice_loss_weight, 0.0), 1.0))

    def _resolved_iou_weight(self) -> float:
        return float(min(max(self._iou_loss_weight, 0.0), 1.0))

    def _resolved_loss_term_weights(self) -> dict[str, float]:
        return resolve_loss_term_weights(
            getattr(self, '_loss_term_weights', None),
            fallback_loss_function=self._resolved_loss_function(),
        )

    def _resolved_hard_pixel_keep_ratio(self) -> float:
        params = getattr(self, '_hard_mining_params', None)
        ratio = float(getattr(params, 'pixel_keep_ratio', 0.25))
        return float(min(max(ratio, 0.01), 1.0))

    def _hard_pixel_mining_enabled(self) -> bool:
        params = getattr(self, '_hard_mining_params', None)
        return bool(getattr(params, 'pixel_enabled', False))

    def _loss_supports_hard_pixel_mining(self, loss_mode: str) -> bool:
        return loss_mode not in ('dice', 'cldice', 'iou', 'boundary', 'focal_tversky')

    @staticmethod
    def _is_finite_tensor(tensor: torch.Tensor) -> bool:
        return bool(torch.isfinite(tensor).all())

    @staticmethod
    def _sanitize_outputs_for_loss(outputs: torch.Tensor) -> torch.Tensor:
        sanitized = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        sanitized = torch.where(torch.isposinf(sanitized), torch.full_like(sanitized, 20.0), sanitized)
        sanitized = torch.where(torch.isneginf(sanitized), torch.full_like(sanitized, -20.0), sanitized)
        return torch.clamp(sanitized, min=-20.0, max=20.0)

    @staticmethod
    def _sanitize_labels_for_loss(label: torch.Tensor) -> torch.Tensor:
        sanitized = torch.nan_to_num(label, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(sanitized, min=0.0, max=1.0)

    def _resolved_cutout_parameters(self) -> tuple[bool, float, int, float]:
        params = getattr(self, '_cutout_params', None)
        enabled = bool(getattr(params, 'enabled', False))
        probability = float(getattr(params, 'probability', 1.0))
        holes = max(1, int(getattr(params, 'holes', 1)))
        size_ratio = float(getattr(params, 'size_ratio', 0.25))
        probability = float(min(max(probability, 0.0), 1.0))
        size_ratio = float(min(max(size_ratio, 0.0), 1.0))
        return enabled, probability, holes, size_ratio

    def _resolved_random_artifact_parameters(self) -> tuple[bool, float, int, float, tuple[str, ...]]:
        params = getattr(self, '_random_artifacts_params', None)
        enabled = bool(getattr(params, 'enabled', False))
        probability = float(getattr(params, 'probability', 1.0))
        count = max(1, int(getattr(params, 'count', 1)))
        size_ratio = float(getattr(params, 'size_ratio', 0.25))
        artifact_types = (
            tuple(params.enabled_types())
            if params is not None and hasattr(params, 'enabled_types')
            else ('dust', 'resist_residue', 'etch_residue', 'particle_cluster', 'flake')
        )
        probability = float(min(max(probability, 0.0), 1.0))
        size_ratio = float(min(max(size_ratio, 0.0), 1.0))
        return enabled, probability, count, size_ratio, artifact_types

    @staticmethod
    def _random_artifact_size_bounds(
        *,
        image_height: int,
        image_width: int,
        size_ratio: float,
    ) -> tuple[int, int, int, int]:
        max_artifact_height = max(1, min(int(image_height), int(round(int(image_height) * float(size_ratio)))))
        max_artifact_width = max(1, min(int(image_width), int(round(int(image_width) * float(size_ratio)))))
        min_artifact_height = 1 if max_artifact_height <= 2 else max(2, int(round(max_artifact_height * 0.35)))
        min_artifact_width = 1 if max_artifact_width <= 2 else max(2, int(round(max_artifact_width * 0.35)))
        min_artifact_height = min(max_artifact_height, min_artifact_height)
        min_artifact_width = min(max_artifact_width, min_artifact_width)
        return (
            int(min_artifact_height),
            int(max_artifact_height),
            int(min_artifact_width),
            int(max_artifact_width),
        )

    def _resolve_random_artifact_prewarm_buckets(
        self,
        *,
        image_height: int,
        image_width: int,
        size_ratio: float,
    ) -> list[tuple[int, int]]:
        min_h, max_h, min_w, max_w = self._random_artifact_size_bounds(
            image_height=image_height,
            image_width=image_width,
            size_ratio=size_ratio,
        )
        levels_h = [min_h, max_h]
        levels_w = [min_w, max_w]
        if max_h > min_h:
            levels_h.extend([
                int(round(min_h + ((max_h - min_h) / 3.0))),
                int(round(min_h + (((max_h - min_h) * 2.0) / 3.0))),
            ])
        if max_w > min_w:
            levels_w.extend([
                int(round(min_w + ((max_w - min_w) / 3.0))),
                int(round(min_w + (((max_w - min_w) * 2.0) / 3.0))),
            ])
        buckets: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for bucket in zip(sorted(set(levels_h)), sorted(set(levels_w))):
            normalized = (
                max(1, int(bucket[0])),
                max(1, int(bucket[1])),
            )
            if normalized in seen:
                continue
            seen.add(normalized)
            buckets.append(normalized)
        return buckets

    def _ensure_random_artifact_bank(
        self,
        *,
        image: torch.Tensor,
        artifact_types: tuple[str, ...],
    ) -> _RandomArtifactBank | None:
        if not artifact_types:
            return None
        if not hasattr(self, '_random_artifact_bank'):
            return None
        channels = int(image.shape[1])
        bank = getattr(self, '_random_artifact_bank', None)
        if bank is not None and bank.matches(channels=channels, artifact_types=artifact_types):
            return bank
        self._stop_random_artifact_bank()
        _enabled, _probability, count, size_ratio, _artifact_types = self._resolved_random_artifact_parameters()
        bank = _RandomArtifactBank(
            channels=channels,
            artifact_types=artifact_types,
            target_per_bucket=max(_RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET, int(count) * 2),
        )
        prewarm_buckets = self._resolve_random_artifact_prewarm_buckets(
            image_height=int(image.shape[2]),
            image_width=int(image.shape[3]),
            size_ratio=size_ratio,
        )
        bank.start(prewarm_buckets=prewarm_buckets)
        bank.wait_until_ready(timeout=_RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC)
        self._random_artifact_bank = bank
        return bank

    def _stop_random_artifact_bank(self) -> None:
        bank = getattr(self, '_random_artifact_bank', None)
        self._random_artifact_bank = None
        if bank is not None:
            bank.stop()

    def _resolved_mixup_parameters(self) -> tuple[bool, float, float]:
        params = getattr(self, '_mixup_params', None)
        enabled = bool(getattr(params, 'enabled', False))
        probability = float(getattr(params, 'probability', 1.0))
        alpha = float(getattr(params, 'alpha', 0.2))
        probability = float(min(max(probability, 0.0), 1.0))
        alpha = float(max(alpha, 0.0))
        return enabled, probability, alpha

    def _has_training_batch_augmentations(self) -> bool:
        mixup_enabled, mixup_probability, mixup_alpha = self._resolved_mixup_parameters()
        if mixup_enabled and mixup_probability > 0.0 and mixup_alpha > 0.0:
            return True
        cutout_enabled, cutout_probability, _cutout_holes, cutout_size_ratio = self._resolved_cutout_parameters()
        if cutout_enabled and cutout_probability > 0.0 and cutout_size_ratio > 0.0:
            return True
        artifacts_enabled, artifacts_probability, _artifacts_count, artifacts_size_ratio, artifact_types = (
            self._resolved_random_artifact_parameters()
        )
        return bool(
            artifacts_enabled
            and artifacts_probability > 0.0
            and artifacts_size_ratio > 0.0
            and artifact_types
        )

    @staticmethod
    def _resolved_ignore_index() -> int:
        raw = str(os.getenv('NEURALIMAGE_IGNORE_INDEX', '-100')).strip()
        try:
            return int(raw)
        except ValueError:
            return -100

    def _reduce_loss_map_per_sample(
        self,
        loss_map: torch.Tensor,
        *,
        loss_mode: str,
        apply_pixel_mining: bool,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_flat = loss_map.view(loss_map.shape[0], -1)
        valid_flat = valid_mask.view(valid_mask.shape[0], -1).to(dtype=torch.bool) if valid_mask is not None else None
        use_pixel_mining = (
            apply_pixel_mining
            and self._hard_pixel_mining_enabled()
            and self._loss_supports_hard_pixel_mining(loss_mode)
        )
        keep_ratio = self._resolved_hard_pixel_keep_ratio()
        reduced_losses: list[torch.Tensor] = []
        for batch_index in range(loss_flat.shape[0]):
            sample_losses = loss_flat[batch_index]
            if valid_flat is not None:
                sample_losses = sample_losses[valid_flat[batch_index]]
            if sample_losses.numel() == 0:
                reduced_losses.append(loss_flat.new_tensor(0.0))
                continue
            if use_pixel_mining and keep_ratio < 1.0:
                keep_count = max(1, int(math.ceil(sample_losses.numel() * keep_ratio)))
                if keep_count < sample_losses.numel():
                    sample_losses = torch.topk(sample_losses, keep_count, largest=True, sorted=False).values
            reduced_losses.append(sample_losses.mean())
        return torch.stack(reduced_losses, dim=0)

    @staticmethod
    def _compute_focal_bce_loss_map(
        outputs: torch.Tensor,
        label: torch.Tensor,
        bce_loss_map: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(outputs)
        pt = (probs * label) + ((1.0 - probs) * (1.0 - label))
        pt = torch.clamp(pt, min=0.0, max=1.0)
        alpha_t = (FOCAL_LOSS_ALPHA * label) + ((1.0 - FOCAL_LOSS_ALPHA) * (1.0 - label))
        focal_factor = torch.pow(1.0 - pt, FOCAL_LOSS_GAMMA)
        focal_map = alpha_t * focal_factor * bce_loss_map
        focal_map = torch.nan_to_num(focal_map, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(focal_map, min=0.0, max=50.0)

    @staticmethod
    def _compute_soft_boundary_map(
        mask: torch.Tensor,
        *,
        kernel_size: int = BOUNDARY_LOSS_KERNEL_SIZE,
    ) -> torch.Tensor:
        kernel_size = max(1, int(kernel_size))
        if kernel_size <= 1:
            return torch.clamp(torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)

        pad = kernel_size // 2
        padded_mask = F.pad(mask, (pad, pad, pad, pad), mode='constant', value=0.0)
        dilation = F.max_pool2d(padded_mask, kernel_size=kernel_size, stride=1)
        erosion = -F.max_pool2d(-padded_mask, kernel_size=kernel_size, stride=1)
        boundary_map = dilation - erosion
        boundary_map = torch.nan_to_num(boundary_map, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(boundary_map, min=0.0, max=1.0)

    def _compute_boundary_loss_per_sample(
        self,
        outputs: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(outputs)
        label_bin = (label >= 0.5).to(dtype=probs.dtype)
        pred_boundary = self._compute_soft_boundary_map(probs)
        target_boundary = self._compute_soft_boundary_map(label_bin)
        pred_boundary_flat = pred_boundary.view(pred_boundary.shape[0], -1)
        target_boundary_flat = target_boundary.view(target_boundary.shape[0], -1)
        eps = 1e-6
        intersection = (pred_boundary_flat * target_boundary_flat).sum(dim=1)
        denom = pred_boundary_flat.sum(dim=1) + target_boundary_flat.sum(dim=1)
        boundary_loss = 1.0 - ((2.0 * intersection + eps) / (denom + eps))
        boundary_loss = torch.nan_to_num(boundary_loss, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(boundary_loss, min=0.0, max=50.0)

    @staticmethod
    def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
        vertical = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
        horizontal = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
        eroded = torch.minimum(vertical, horizontal)
        eroded = torch.nan_to_num(eroded, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(eroded, min=0.0, max=1.0)

    @staticmethod
    def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        dilated = torch.nan_to_num(dilated, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(dilated, min=0.0, max=1.0)

    def _soft_open(self, mask: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(mask))

    def _soft_skeletonize(
        self,
        mask: torch.Tensor,
        *,
        iterations: int = CLDICE_SKELETON_ITERATIONS,
    ) -> torch.Tensor:
        work = torch.clamp(torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
        skeleton = F.relu(work - self._soft_open(work))
        for _ in range(max(0, int(iterations))):
            work = self._soft_erode(work)
            delta = F.relu(work - self._soft_open(work))
            skeleton = skeleton + F.relu(delta - (skeleton * delta))
        skeleton = torch.nan_to_num(skeleton, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(skeleton, min=0.0, max=1.0)

    def _compute_cldice_loss_per_sample(
        self,
        outputs: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(outputs)
        target = (label >= 0.5).to(dtype=probs.dtype)
        pred_skeleton = self._soft_skeletonize(probs)
        target_skeleton = self._soft_skeletonize(target)
        pred_flat = probs.view(probs.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        pred_skeleton_flat = pred_skeleton.view(pred_skeleton.shape[0], -1)
        target_skeleton_flat = target_skeleton.view(target_skeleton.shape[0], -1)
        eps = 1e-6
        topology_precision = ((pred_skeleton_flat * target_flat).sum(dim=1) + eps) / (
            pred_skeleton_flat.sum(dim=1) + eps
        )
        topology_sensitivity = ((target_skeleton_flat * pred_flat).sum(dim=1) + eps) / (
            target_skeleton_flat.sum(dim=1) + eps
        )
        cldice = (2.0 * topology_precision * topology_sensitivity + eps) / (
            topology_precision + topology_sensitivity + eps
        )
        cldice_loss = 1.0 - cldice
        cldice_loss = torch.nan_to_num(cldice_loss, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(cldice_loss, min=0.0, max=50.0)

    @staticmethod
    def _compute_focal_tversky_loss_per_sample(
        outputs: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(outputs)
        probs_flat = probs.view(probs.shape[0], -1)
        label_flat = label.view(label.shape[0], -1)
        eps = 1e-6
        true_positive = (probs_flat * label_flat).sum(dim=1)
        false_positive = (probs_flat * (1.0 - label_flat)).sum(dim=1)
        false_negative = ((1.0 - probs_flat) * label_flat).sum(dim=1)
        tversky = (true_positive + eps) / (
            true_positive
            + (FOCAL_TVERSKY_ALPHA * false_positive)
            + (FOCAL_TVERSKY_BETA * false_negative)
            + eps
        )
        focal_tversky_loss = torch.pow(1.0 - tversky, FOCAL_LOSS_GAMMA)
        focal_tversky_loss = torch.nan_to_num(focal_tversky_loss, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(focal_tversky_loss, min=0.0, max=50.0)

    def _compute_single_per_sample_loss(
        self,
        loss_mode: str,
        outputs: torch.Tensor,
        label: torch.Tensor,
        bce_criterion: nn.Module,
        *,
        apply_pixel_mining: bool = False,
    ) -> torch.Tensor:
        if loss_mode in ('ce', 'ce_dice'):
            logits_two_class = torch.cat([-outputs, outputs], dim=1)
            soft_label = label[:, 0, :, :]
            target_probs = torch.stack((1.0 - soft_label, soft_label), dim=1)
            log_probs = F.log_softmax(logits_two_class, dim=1)
            ce_map = -(target_probs * log_probs).sum(dim=1)
            valid_mask = torch.ones_like(soft_label, dtype=ce_map.dtype)
            ce_per_sample = self._reduce_loss_map_per_sample(
                ce_map,
                loss_mode=loss_mode,
                apply_pixel_mining=apply_pixel_mining,
                valid_mask=valid_mask,
            )
            ce_per_sample = torch.nan_to_num(ce_per_sample, nan=1.0, posinf=50.0, neginf=0.0)
            if loss_mode == 'ce':
                return torch.clamp(ce_per_sample, min=0.0, max=50.0)

            probs_fg = torch.softmax(logits_two_class, dim=1)[:, 1, :, :]
            probs_flat = probs_fg.view(probs_fg.shape[0], -1)
            label_flat = soft_label.view(soft_label.shape[0], -1)
            valid_flat = valid_mask.view(valid_mask.shape[0], -1)
            probs_flat = probs_flat * valid_flat
            label_flat = label_flat * valid_flat
            eps = 1e-6
            intersection = (probs_flat * label_flat).sum(dim=1)
            denom = probs_flat.sum(dim=1) + label_flat.sum(dim=1)
            dice_per_sample = 1.0 - ((2.0 * intersection + eps) / (denom + eps))
            dice_weight = self._resolved_dice_weight()
            mixed = ((1.0 - dice_weight) * ce_per_sample) + (dice_weight * dice_per_sample)
            mixed = torch.nan_to_num(mixed, nan=1.0, posinf=50.0, neginf=0.0)
            return torch.clamp(mixed, min=0.0, max=50.0)

        if loss_mode == 'boundary':
            return self._compute_boundary_loss_per_sample(outputs, label)

        if loss_mode == 'cldice':
            return self._compute_cldice_loss_per_sample(outputs, label)

        if loss_mode == 'focal_tversky':
            return self._compute_focal_tversky_loss_per_sample(outputs, label)

        bce_loss_map = cast(torch.Tensor, bce_criterion(outputs, label))
        pointwise_loss_map = bce_loss_map
        if loss_mode in ('focal_bce', 'focal_dice', 'focal_iou'):
            pointwise_loss_map = self._compute_focal_bce_loss_map(outputs, label, bce_loss_map)
        pointwise_per_sample = self._reduce_loss_map_per_sample(
            pointwise_loss_map,
            loss_mode=loss_mode,
            apply_pixel_mining=apply_pixel_mining,
        )
        pointwise_per_sample = torch.nan_to_num(pointwise_per_sample, nan=1.0, posinf=50.0, neginf=0.0)
        if loss_mode in ('bce', 'focal_bce'):
            return torch.clamp(pointwise_per_sample, min=0.0, max=50.0)

        probs = torch.sigmoid(outputs)
        probs_flat = probs.view(probs.shape[0], -1)
        label_flat = label.view(label.shape[0], -1)
        eps = 1e-6
        intersection = (probs_flat * label_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + label_flat.sum(dim=1)
        dice = (2.0 * intersection + eps) / (denom + eps)
        dice_per_sample = 1.0 - dice
        union = probs_flat.sum(dim=1) + label_flat.sum(dim=1) - intersection
        iou = (intersection + eps) / (union + eps)
        iou_per_sample = 1.0 - iou
        if loss_mode == 'dice':
            return dice_per_sample
        if loss_mode == 'iou':
            return torch.clamp(torch.nan_to_num(iou_per_sample, nan=1.0, posinf=50.0, neginf=0.0), min=0.0, max=50.0)

        if loss_mode in ('bce_dice', 'focal_dice'):
            dice_weight = self._resolved_dice_weight()
            mixed = ((1.0 - dice_weight) * pointwise_per_sample) + (dice_weight * dice_per_sample)
            mixed = torch.nan_to_num(mixed, nan=1.0, posinf=50.0, neginf=0.0)
            return torch.clamp(mixed, min=0.0, max=50.0)

        iou_weight = self._resolved_iou_weight()
        mixed = ((1.0 - iou_weight) * pointwise_per_sample) + (iou_weight * iou_per_sample)
        mixed = torch.nan_to_num(mixed, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(mixed, min=0.0, max=50.0)

    def _compute_per_sample_loss(
        self,
        outputs: torch.Tensor,
        label: torch.Tensor,
        bce_criterion: nn.Module,
        *,
        apply_pixel_mining: bool = False,
    ) -> torch.Tensor:
        outputs = self._sanitize_outputs_for_loss(outputs)
        label = self._sanitize_labels_for_loss(label)
        combined_loss: torch.Tensor | None = None
        for loss_mode, coefficient in self._resolved_loss_term_weights().items():
            single_loss = self._compute_single_per_sample_loss(
                loss_mode,
                outputs,
                label,
                bce_criterion,
                apply_pixel_mining=apply_pixel_mining,
            )
            weighted_loss = single_loss * float(coefficient)
            combined_loss = weighted_loss if combined_loss is None else (combined_loss + weighted_loss)
        if combined_loss is None:
            fallback_loss = self._compute_single_per_sample_loss(
                self._resolved_loss_function(),
                outputs,
                label,
                bce_criterion,
                apply_pixel_mining=apply_pixel_mining,
            )
            return torch.clamp(torch.nan_to_num(fallback_loss, nan=1.0, posinf=50.0, neginf=0.0), min=0.0, max=50.0)
        combined_loss = torch.nan_to_num(combined_loss, nan=1.0, posinf=50.0, neginf=0.0)
        return torch.clamp(combined_loss, min=0.0, max=50.0)

    @staticmethod
    def _compute_binary_metrics(
        *,
        true_positive: float,
        false_positive: float,
        false_negative: float,
        correct: int,
        total: int,
    ) -> dict[str, float]:
        accuracy = (float(correct) / float(total)) if total else 0.0
        iou_denom = true_positive + false_positive + false_negative
        dice_denom = (2.0 * true_positive) + false_positive + false_negative
        if iou_denom == 0.0:
            return {
                'accuracy': float(accuracy),
                'iou': 1.0,
                'dice': 1.0,
                'f1': 1.0,
            }

        iou = true_positive / iou_denom
        dice = (2.0 * true_positive) / dice_denom if dice_denom > 0.0 else 0.0
        precision_denom = true_positive + false_positive
        recall_denom = true_positive + false_negative
        precision = true_positive / precision_denom if precision_denom > 0.0 else 0.0
        recall = true_positive / recall_denom if recall_denom > 0.0 else 0.0
        f1_denom = precision + recall
        f1 = (2.0 * precision * recall) / f1_denom if f1_denom > 0.0 else 0.0
        return {
            'accuracy': float(accuracy),
            'iou': float(iou),
            'dice': float(dice),
            'f1': float(f1),
        }

    @staticmethod
    def _pick_best_validation_threshold(
        threshold_metrics: dict[float, dict[str, float]],
    ) -> tuple[float, dict[str, float]]:
        if not threshold_metrics:
            return 0.5, {'accuracy': 0.0, 'iou': 0.0, 'dice': 0.0, 'f1': 0.0}

        def _sort_key(item: tuple[float, dict[str, float]]) -> tuple[float, float, float]:
            threshold, metrics = item
            return (
                float(metrics.get('dice', 0.0)),
                float(metrics.get('iou', 0.0)),
                -abs(float(threshold) - 0.5),
            )

        best_threshold, best_metrics = max(threshold_metrics.items(), key=_sort_key)
        return float(best_threshold), dict(best_metrics)

    def _resolve_validation_dataset(self) -> Any:
        if self._val_dataloader is None:
            return None
        return getattr(self._val_dataloader, 'dataset', None)

    @staticmethod
    def _unwrap_validation_dataset(dataset: Any) -> Any:
        current = dataset
        seen_ids: set[int] = set()
        while current is not None and id(current) not in seen_ids:
            seen_ids.add(id(current))
            base_dataset = getattr(current, '_base_dataset', None)
            if base_dataset is None:
                return current
            current = base_dataset
        return current

    def _resolve_validation_export_items(self) -> list[dict[str, Any]]:
        dataset = self._resolve_validation_dataset()
        base_dataset = self._unwrap_validation_dataset(dataset)
        if isinstance(base_dataset, NoCutDataset):
            cut_settings = getattr(base_dataset, '_cut_settings', None)
            segment_size = tuple(getattr(cut_settings, 'segment_size', (0, 0)))
            if len(segment_size) != 2:
                return []
            step = int(getattr(cut_settings, 'step', segment_size[0]))
            overlap = max(0, int(segment_size[0]) - step)
            channels = int(getattr(base_dataset, 'colors', 1))
            return [
                {
                    'image_path': Path(image_path),
                    'segment_shape': (channels, int(segment_size[0]), int(segment_size[1])),
                    'overlap': overlap,
                    'use_context_branch': bool(getattr(base_dataset, '_use_context_branch', False)),
                    'context_crop_size': getattr(base_dataset, '_context_crop_size', None),
                    'context_input_size': getattr(base_dataset, '_context_input_size', None),
                }
                for image_path, _label_path in getattr(base_dataset, 'samples', [])
            ]

        if isinstance(base_dataset, CustomDataset):
            export_items: list[dict[str, Any]] = []
            channels = int(getattr(base_dataset, 'channels', 1))
            for image_path, _label_path in getattr(base_dataset, 'samples', []):
                image_file = Path(image_path)
                try:
                    with Image.open(image_file) as source_image:
                        width, height = source_image.size
                except OSError:
                    continue
                export_items.append(
                    {
                        'image_path': image_file,
                        'segment_shape': (channels, int(width), int(height)),
                        'overlap': 0,
                        'use_context_branch': False,
                        'context_crop_size': None,
                        'context_input_size': None,
                    }
                )
            return export_items

        return []

    @staticmethod
    def _resolve_validation_sample_indices(
        sample_indices: Any,
        *,
        fallback_start: int,
        batch_size: int,
    ) -> list[int]:
        if torch.is_tensor(sample_indices):
            return [int(index) for index in sample_indices.detach().cpu().tolist()]
        if sample_indices is None:
            return list(range(int(fallback_start), int(fallback_start) + int(batch_size)))
        return [int(index) for index in sample_indices]

    @staticmethod
    def _can_cache_no_cut_validation_export(base_dataset: NoCutDataset) -> bool:
        cut_settings = getattr(base_dataset, '_cut_settings', None)
        if bool(getattr(cut_settings, 'random_crop', False)):
            return False
        if bool(getattr(base_dataset, '_skip_uniform_labels', False)):
            return False
        if bool(getattr(base_dataset, '_rare_patch_oversampling_enabled', False)):
            return False
        return True

    @staticmethod
    def _decode_no_cut_export_item(base_dataset: NoCutDataset, item: int) -> tuple[int, int, int, int]:
        cut_settings = getattr(base_dataset, '_cut_settings', None)
        loc = int(item)
        augmentation_variant = 0
        if bool(getattr(cut_settings, 'additional_augmentation', False)):
            loc, augmentation_variant = divmod(loc, 2)
        scale_variant = 0
        if bool(getattr(cut_settings, 'scale_augmentation', False)):
            loc, scale_variant = divmod(loc, 2)

        if bool(getattr(cut_settings, 'vertical_rotation', False)) and bool(getattr(cut_settings, 'horizontal_rotation', False)):
            location = loc // 4
            rotation_index = loc % 4
        elif bool(getattr(cut_settings, 'horizontal_rotation', False)):
            location = loc // 3
            rotation_index = 2 - (loc % 3)
        elif bool(getattr(cut_settings, 'vertical_rotation', False)):
            location = loc // 2
            rotation_index = 2 * (loc % 2)
        else:
            location = loc
            rotation_index = 0
        return int(location), int(rotation_index), int(scale_variant), int(augmentation_variant)

    @staticmethod
    def _build_no_cut_frame_part_lookup(
        base_dataset: NoCutDataset,
        *,
        frame_index: int,
        parts_count: int,
    ) -> list[int] | None:
        if not bool(getattr(base_dataset, 'shuffle_patches_in_frame', False)):
            return None
        parts = list(range(int(parts_count)))
        randomizer = random.Random(int(getattr(base_dataset, '_frame_seed')(frame_index)))
        randomizer.shuffle(parts)
        return [int(item) for item in parts]

    def _create_validation_export_cache(self) -> _ValidationExportCache | None:
        if (not self._save_validation_binary_images) or self._val_dataloader is None:
            return None

        dataset = self._resolve_validation_dataset()
        base_dataset = self._unwrap_validation_dataset(dataset)
        if dataset is None or base_dataset is None:
            return None

        if isinstance(base_dataset, CustomDataset):
            estimated_bytes = 0
            for image_path, _label_path in getattr(base_dataset, 'samples', []):
                image_file = Path(image_path)
                try:
                    with Image.open(image_file) as source_image:
                        width, height = source_image.size
                except OSError:
                    return None
                estimated_bytes += int(width) * int(height) * 2
                if estimated_bytes > VALIDATION_EXPORT_CACHE_MAX_BYTES:
                    return None
            return _ValidationExportCache(
                mode='custom',
                dataset=dataset,
                sample_predictions={},
            )

        if not isinstance(base_dataset, NoCutDataset):
            return None
        if not self._can_cache_no_cut_validation_export(base_dataset):
            return None

        cut_settings = getattr(base_dataset, '_cut_settings', None)
        segment_size = tuple(getattr(cut_settings, 'segment_size', (0, 0)))
        if len(segment_size) != 2:
            return None

        frame_predictions: dict[int, _ValidationNoCutFrameExportCache] = {}
        estimated_bytes = 0
        for frame_index, (image_path, _label_path) in enumerate(getattr(base_dataset, 'samples', [])):
            prepared_size = ImagePreparator(Path(image_path), getattr(base_dataset, '_prep_settings')).size
            if len(prepared_size) != 2:
                return None
            base_width, base_height = int(prepared_size[0]), int(prepared_size[1])
            sample_calculator = SampleCalculator((base_height, base_width), cut_settings)
            _ = len(sample_calculator)
            width_steps, height_steps = sample_calculator.size
            parts_count = max(0, int(width_steps * height_steps))
            if parts_count <= 0:
                continue

            estimated_bytes += parts_count * int(segment_size[0]) * int(segment_size[1]) * 2
            if estimated_bytes > VALIDATION_EXPORT_CACHE_MAX_BYTES:
                return None

            frame_length = int(getattr(base_dataset, '_frame_lengths', [parts_count])[frame_index])
            frame_predictions[frame_index] = _ValidationNoCutFrameExportCache(
                frame_index=int(frame_index),
                image_path=Path(image_path),
                baseim_size=(base_width, base_height),
                overlap=max(0, int(segment_size[0]) - int(getattr(cut_settings, 'step', segment_size[0]))),
                parts_count=int(parts_count),
                part_lookup=self._build_no_cut_frame_part_lookup(
                    base_dataset,
                    frame_index=int(frame_index),
                    parts_count=int(frame_length),
                ),
                patches={},
            )

        if not frame_predictions:
            return None
        return _ValidationExportCache(
            mode='no_cut',
            dataset=dataset,
            frame_predictions=frame_predictions,
        )

    def _collect_validation_export_batch(
        self,
        export_cache: _ValidationExportCache | None,
        *,
        sample_indices: Any,
        probs: torch.Tensor,
        saved_images: int,
    ) -> None:
        if export_cache is None or probs.ndim < 4:
            return

        resolved_indices = self._resolve_validation_sample_indices(
            sample_indices,
            fallback_start=saved_images,
            batch_size=int(probs.shape[0]),
        )
        if not resolved_indices:
            return

        cpu_probs = probs.detach().cpu().to(dtype=torch.float16).numpy()
        if export_cache.mode == 'custom':
            sample_predictions = export_cache.sample_predictions
            if sample_predictions is None:
                return
            for batch_offset, sample_index in enumerate(resolved_indices):
                if batch_offset >= int(cpu_probs.shape[0]):
                    break
                sample_predictions[int(sample_index)] = np.ascontiguousarray(cpu_probs[batch_offset])
            return

        if export_cache.mode != 'no_cut':
            return

        frame_predictions = export_cache.frame_predictions
        base_dataset = self._unwrap_validation_dataset(export_cache.dataset)
        if frame_predictions is None or not isinstance(base_dataset, NoCutDataset):
            return

        lookup_lengths = getattr(base_dataset, '_lookup_len_list', None)
        if not lookup_lengths:
            return

        for batch_offset, sample_index in enumerate(resolved_indices):
            if batch_offset >= int(cpu_probs.shape[0]):
                break
            try:
                frame_index, local_part = index_in_list(int(sample_index), list(lookup_lengths))
            except (IndexError, ValueError):
                continue
            frame_cache = frame_predictions.get(int(frame_index))
            if frame_cache is None:
                continue
            actual_part = int(local_part)
            if frame_cache.part_lookup is not None:
                if actual_part < 0 or actual_part >= len(frame_cache.part_lookup):
                    continue
                actual_part = int(frame_cache.part_lookup[actual_part])
            location, rotation_index, scale_variant, augmentation_variant = self._decode_no_cut_export_item(
                base_dataset,
                actual_part,
            )
            if rotation_index != 0 or scale_variant != 0 or augmentation_variant != 0:
                continue
            if location < 0 or location >= frame_cache.parts_count or location in frame_cache.patches:
                continue
            frame_cache.patches[int(location)] = np.ascontiguousarray(cpu_probs[batch_offset])

    def _save_validation_binary_predictions_from_cache(
        self,
        *,
        epoch_dir: Path,
        threshold: float,
        export_cache: _ValidationExportCache | None,
    ) -> int | None:
        if export_cache is None:
            return None

        dataset = export_cache.dataset
        if export_cache.mode == 'custom':
            sample_predictions = export_cache.sample_predictions or {}
            if not sample_predictions:
                return None
            if isinstance(dataset, Sized) and len(sample_predictions) < int(len(dataset)):
                return None

            saved_images = 0
            for sample_index in sorted(sample_predictions):
                sample_name = self._describe_validation_sample(dataset, int(sample_index), saved_images)
                sample_path = epoch_dir / f'{saved_images:06d}_{sample_name}.png'
                sample_tensor = np.asarray(sample_predictions[int(sample_index)], dtype=np.float32)
                if sample_tensor.ndim == 3:
                    sample_tensor = sample_tensor[0]
                sample_array = (sample_tensor >= float(threshold)).astype(np.uint8) * 255
                Image.fromarray(sample_array, mode='L').save(sample_path)
                saved_images += 1
            return int(saved_images)

        if export_cache.mode != 'no_cut':
            return None

        frame_predictions = export_cache.frame_predictions or {}
        if not frame_predictions:
            return None
        for frame_cache in frame_predictions.values():
            if len(frame_cache.patches) < int(frame_cache.parts_count):
                return None

        saved_images = 0
        for frame_cache in frame_predictions.values():
            sample_patch = next(iter(frame_cache.patches.values()), None)
            if sample_patch is None:
                continue
            predicted = np.zeros(
                (
                    int(frame_cache.parts_count),
                    int(sample_patch.shape[0]),
                    int(sample_patch.shape[1]),
                    int(sample_patch.shape[2]),
                ),
                dtype=np.float32,
            )
            for location, patch in frame_cache.patches.items():
                predicted[int(location)] = np.asarray(patch, dtype=np.float32)
            _sew(
                epoch_dir,
                {
                    'name': frame_cache.image_path.name,
                    'baseim_size': frame_cache.baseim_size,
                    'overlap': int(frame_cache.overlap),
                    'predicted_image': predicted,
                },
                jpeg_quality=95,
                threshold=float(threshold),
                postprocess_kernel_size=0,
            )
            saved_images += 1
        return int(saved_images)

    def _describe_validation_sample(self, dataset: Any, sample_index: int, fallback_index: int) -> str:
        describe_fn = getattr(dataset, 'describe_sample', None)
        if callable(describe_fn):
            try:
                return self._sanitize_artifact_name(str(describe_fn(int(sample_index))), fallback='sample')
            except Exception:
                pass
        return f'sample_{int(fallback_index):06d}'

    def _save_validation_binary_predictions(
        self,
        *,
        epoch: int,
        device: torch.device,
        autocast_ctx: Callable[[], ContextManager[Any]],
        threshold: float,
        export_cache: _ValidationExportCache | None = None,
    ) -> None:
        if (not self._save_validation_binary_images) or self._val_dataloader is None:
            return

        epoch_dir = self._save_path.parent / 'validation_binary_predictions' / f'epoch_{int(epoch + 1):04d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        saved_images = self._save_validation_binary_predictions_from_cache(
            epoch_dir=epoch_dir,
            threshold=float(threshold),
            export_cache=export_cache,
        )
        if saved_images is not None:
            self._bus.put([
                'logging',
                (
                    f'Validation predictions saved in recognition-style stitched form: {int(saved_images)} file(s) '
                    f'in {epoch_dir}.'
                ),
            ])
            return

        saved_images = 0
        export_items = self._resolve_validation_export_items()
        batch_size = max(1, int(getattr(self._val_dataloader, 'batch_size', 1) or 1))

        if export_items:
            with torch.no_grad():
                for export_item in export_items:
                    predicted = _gpu_predict(
                        _cut_image_prepare(
                            Path(export_item['image_path']),
                            tuple(export_item['segment_shape']),
                            int(export_item['overlap']),
                            use_context_branch=bool(export_item['use_context_branch']),
                            context_crop_size=export_item['context_crop_size'],
                            context_input_size=export_item['context_input_size'],
                        ),
                        self._model,
                        device,
                        batch_size,
                    )
                    _sew(
                        epoch_dir,
                        predicted,
                        jpeg_quality=95,
                        threshold=float(threshold),
                        postprocess_kernel_size=0,
                    )
                    saved_images += 1
        else:
            dataset = self._resolve_validation_dataset()
            if dataset is None:
                return
            with torch.no_grad():
                for batch in self._val_dataloader:
                    data, _target, sample_indices = self._split_batch(batch)
                    inputs = self._move_batch_input_to_device(data, device)
                    with autocast_ctx():
                        outputs = self._forward_model(inputs)
                    probs = torch.sigmoid(self._sanitize_outputs_for_loss(outputs))
                    binary_predictions = (probs >= float(threshold)).detach().cpu()
                    resolved_indices = self._resolve_validation_sample_indices(
                        sample_indices,
                        fallback_start=saved_images,
                        batch_size=int(binary_predictions.shape[0]),
                    )

                    for batch_offset in range(int(binary_predictions.shape[0])):
                        sample_index = (
                            int(resolved_indices[batch_offset]) if batch_offset < len(resolved_indices) else saved_images
                        )
                        sample_name = self._describe_validation_sample(dataset, sample_index, saved_images)
                        sample_path = epoch_dir / f'{saved_images:06d}_{sample_name}.png'
                        sample_tensor = binary_predictions[batch_offset]
                        if sample_tensor.ndim == 3:
                            sample_tensor = sample_tensor[0]
                        sample_array = sample_tensor.numpy().astype(np.uint8) * 255
                        Image.fromarray(sample_array, mode='L').save(sample_path)
                        saved_images += 1

        self._bus.put([
            'logging',
            (
                f'Validation predictions saved in recognition-style stitched form: {saved_images} file(s) '
                f'in {epoch_dir}.'
            ),
        ])

    def _run_validation_epoch(
        self,
        epoch: int,
        device: torch.device,
        bce_criterion: nn.Module,
        autocast_ctx: Callable[[], ContextManager[Any]],
    ) -> dict[str, float] | None:
        if self._val_dataloader is None:
            return None
        val_dataset_len = len(cast(Sized, self._val_dataloader.dataset))
        if val_dataset_len == 0:
            return None

        self._model.eval()
        val_loss = 0.0
        val_loss_samples = 0
        correct = 0
        total = 0
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0
        threshold_counts = {
            float(threshold): {
                'correct': 0,
                'total': 0,
                'tp': 0.0,
                'fp': 0.0,
                'fn': 0.0,
            }
            for threshold in VALIDATION_THRESHOLD_CANDIDATES
        }
        skipped_non_finite_batches = 0
        validation_export_cache = self._create_validation_export_cache()
        export_saved_images = 0
        with torch.no_grad():
            for batch in self._val_dataloader:
                data, target, _sample_indices = self._split_batch(batch)
                inputs = self._move_batch_input_to_device(data, device)
                image = self._extract_local_image(inputs)
                label = target.to(device, non_blocking=True)
                with autocast_ctx():
                    outputs = self._forward_model(inputs)
                    per_sample_loss = self._compute_per_sample_loss(outputs, label, bce_criterion)
                    loss = per_sample_loss.mean()

                if not self._is_finite_tensor(loss):
                    skipped_non_finite_batches += 1
                    continue

                val_loss += loss.item() * image.size(0)
                val_loss_samples += int(image.size(0))
                probs = torch.sigmoid(self._sanitize_outputs_for_loss(outputs))
                self._collect_validation_export_batch(
                    validation_export_cache,
                    sample_indices=_sample_indices,
                    probs=probs,
                    saved_images=export_saved_images,
                )
                export_saved_images += int(probs.shape[0])
                preds = probs >= 0.5
                label_bin = self._sanitize_labels_for_loss(label) >= 0.5
                correct += (preds == label_bin).sum().item()
                total += label_bin.numel()
                preds_f = preds.float()
                label_f = label_bin.float()
                true_positive += float((preds_f * label_f).sum().item())
                false_positive += float((preds_f * (1.0 - label_f)).sum().item())
                false_negative += float(((1.0 - preds_f) * label_f).sum().item())
                for threshold, counts in threshold_counts.items():
                    threshold_preds = probs >= float(threshold)
                    counts['correct'] += int((threshold_preds == label_bin).sum().item())
                    counts['total'] += int(label_bin.numel())
                    threshold_preds_f = threshold_preds.float()
                    counts['tp'] += float((threshold_preds_f * label_f).sum().item())
                    counts['fp'] += float((threshold_preds_f * (1.0 - label_f)).sum().item())
                    counts['fn'] += float(((1.0 - threshold_preds_f) * label_f).sum().item())

        if skipped_non_finite_batches > 0:
            self._bus.put([
                'logging',
                (
                    'Validation warning: '
                    f'skipped {skipped_non_finite_batches} batch(es) with non-finite loss.'
                ),
            ])
            validation_export_cache = None
        if val_loss_samples <= 0:
            self._bus.put(['logging', 'Validation warning: all batches were skipped due to non-finite loss.'])
            return None

        avg_val_loss = val_loss / val_loss_samples
        base_metrics = self._compute_binary_metrics(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            correct=correct,
            total=total,
        )
        threshold_metrics = {
            float(threshold): self._compute_binary_metrics(
                true_positive=float(counts['tp']),
                false_positive=float(counts['fp']),
                false_negative=float(counts['fn']),
                correct=int(counts['correct']),
                total=int(counts['total']),
            )
            for threshold, counts in threshold_counts.items()
        }
        best_threshold, best_threshold_metrics = self._pick_best_validation_threshold(threshold_metrics)
        self._recommended_inference_threshold = float(best_threshold)
        self._save_validation_binary_predictions(
            epoch=epoch,
            device=device,
            autocast_ctx=autocast_ctx,
            threshold=float(best_threshold),
            export_cache=validation_export_cache,
        )

        return {
            'loss': float(avg_val_loss),
            'accuracy': float(base_metrics['accuracy']),
            'iou': float(base_metrics['iou']),
            'dice': float(base_metrics['dice']),
            'f1': float(base_metrics['f1']),
            'best_threshold': float(best_threshold),
            'best_threshold_accuracy': float(best_threshold_metrics['accuracy']),
            'best_threshold_iou': float(best_threshold_metrics['iou']),
            'best_threshold_dice': float(best_threshold_metrics['dice']),
            'best_threshold_f1': float(best_threshold_metrics['f1']),
        }

    def _create_optimizer(self):
        params = self._optimizer_params
        if params.name == OptimizerName.adam:
            adam_kwargs: dict[str, Any] = {'lr': params.learning_rate, 'weight_decay': params.weight_decay}
            if self._can_use_fused_optim():
                adam_kwargs['fused'] = True
            try:
                return optim.Adam(self._model.parameters(), **adam_kwargs)
            except TypeError:
                adam_kwargs.pop('fused', None)
                return optim.Adam(self._model.parameters(), **adam_kwargs)
        if params.name == OptimizerName.adamw:
            return self._create_adamw_optimizer(params)
        if params.name == OptimizerName.adamw_muon:
            return self._create_adamw_muon_optimizer(params)
        raise ValueError(f'Unsupported optimizer: {params.name}')

    def _can_use_fused_optim(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            return any(p.is_cuda for p in self._model.parameters())
        except Exception:
            return False

    def _resolve_mixed_precision(self, device: torch.device) -> tuple[MixedPrecisionMode, torch.dtype | None, bool]:
        if device.type != 'cuda':
            return MixedPrecisionMode.off, None, False

        mode = self._mixed_precision
        if mode == MixedPrecisionMode.off:
            return mode, None, False
        if mode == MixedPrecisionMode.fp16:
            return mode, torch.float16, True
        if mode == MixedPrecisionMode.bf16:
            is_bf16_supported = False
            try:
                is_bf16_supported = bool(torch.cuda.is_bf16_supported())
            except Exception:
                is_bf16_supported = False
            if is_bf16_supported:
                return mode, torch.bfloat16, False
            return MixedPrecisionMode.fp16, torch.float16, True
        return MixedPrecisionMode.off, None, False

    def _build_adamw_param_groups(self, params: OptimizerParameters) -> list[dict[str, Any]]:
        norm_param_ids: set[int] = set()
        batch_norm_base = getattr(nn.modules.batchnorm, '_BatchNorm', ())
        norm_types = (nn.GroupNorm, batch_norm_base) if batch_norm_base else (nn.GroupNorm,)

        for module in self._base_model.modules():
            if isinstance(module, norm_types):
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        norm_param_ids.add(id(p))

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for param in self._model.parameters():
            if not param.requires_grad:
                continue
            if id(param) in norm_param_ids:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append({'params': decay_params, 'weight_decay': params.weight_decay})
        if no_decay_params:
            param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})
        return param_groups

    def _create_adamw_optimizer(self, params: OptimizerParameters):
        use_fused = self._can_use_fused_optim()
        param_groups = self._build_adamw_param_groups(params)
        if param_groups:
            self._bus.put([
                'logging',
                f'AdamW: исключены norm-параметры из weight decay (групп: {len(param_groups)}).',
            ])
            adamw_kwargs: dict[str, Any] = {'lr': params.learning_rate}
            if use_fused:
                adamw_kwargs['fused'] = True
            try:
                return optim.AdamW(param_groups, **adamw_kwargs)
            except TypeError:
                adamw_kwargs.pop('fused', None)
                return optim.AdamW(param_groups, **adamw_kwargs)
        adamw_kwargs = {'lr': params.learning_rate, 'weight_decay': params.weight_decay}
        if use_fused:
            adamw_kwargs['fused'] = True
        try:
            return optim.AdamW(self._model.parameters(), **adamw_kwargs)
        except TypeError:
            adamw_kwargs.pop('fused', None)
            return optim.AdamW(self._model.parameters(), **adamw_kwargs)

    @staticmethod
    def _load_optional_attribute(module_name: str, attribute_name: str) -> Any | None:
        try:
            module_spec = importlib.util.find_spec(module_name)
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
            return None
        if module_spec is None:
            return None
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return None
        return getattr(module, attribute_name, None)

    def _create_adamw_muon_optimizer(self, params: OptimizerParameters):
        """
        Prepared hook for Muon integration.
        Tries to use optional Muon package; if unavailable falls back to AdamW.
        """
        muon_cls = getattr(optim, 'Muon', None)
        if muon_cls is None:
            for module_name in ('muon', 'optimizers.muon', 'torch_optimizer'):
                muon_cls = self._load_optional_attribute(module_name, 'Muon')
                if muon_cls is not None:
                    break

        if muon_cls is None:
            self._bus.put([
                'logging',
                'Muon optimizer is unavailable. Using AdamW.',
            ])
            return self._create_adamw_optimizer(params)

        try:
            return muon_cls(
                self._model.parameters(),
                lr=params.learning_rate,
                weight_decay=params.weight_decay,
            )
        except Exception as error:
            self._bus.put([
                'logging',
                f'Muon optimizer initialization failed ({error}). Using AdamW.',
            ])
            return self._create_adamw_optimizer(params)

    @staticmethod
    def _tensor_to_preview_array(tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().cpu().float().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 3:
            arr = arr[:, :, 0]

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        min_val = float(arr.min()) if arr.size else 0.0
        max_val = float(arr.max()) if arr.size else 0.0
        if 0.0 <= min_val and max_val <= 1.0:
            arr = arr * 255.0
        elif max_val > min_val:
            arr = (arr - min_val) * (255.0 / (max_val - min_val))
        else:
            arr = np.zeros_like(arr)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _try_compile_model(self, is_main_process: bool = True, device: torch.device | None = None) -> None:
        if bool(getattr(sys, 'frozen', False)):
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', 'torch.compile отключен в сборке PyInstaller.'])
            return
        if isinstance(self._model, nn.DataParallel):
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', 'torch.compile skipped: nn.DataParallel model wrapper is active.'])
            return
        if not self._env_bool('NEURALIMAGE_TORCH_COMPILE', True):
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', 'torch.compile disabled by NEURALIMAGE_TORCH_COMPILE=0.'])
            return
        compile_fn = getattr(torch, 'compile', None)
        if compile_fn is None:
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', 'torch.compile недоступен в этой версии PyTorch.'])
            return
        target_device_type = device.type if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        compile_unavailable_reason = _get_torch_compile_unavailable_reason(target_device_type)
        if compile_unavailable_reason is not None:
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', f'torch.compile disabled: {compile_unavailable_reason}'])
            return
        try:
            self._uncompiled_model = self._model
            compile_mode, mode_reason = _resolve_torch_compile_mode(target_device_type, device)
            self._model = compile_fn(self._model, mode=compile_mode, dynamic=False)
            self._torch_compile_active = True
            if is_main_process:
                self._bus.put(['logging', f'torch.compile enabled (mode={compile_mode}, reason={mode_reason}).'])
        except Exception as error:
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', f'torch.compile отключен (fallback): {error}'])

    @staticmethod
    def _is_torch_compile_runtime_failure(error: Exception) -> bool:
        current: BaseException | None = error
        for _ in range(6):
            if current is None:
                break
            message = str(current)
            if type(current).__name__ == 'BackendCompilerFailed':
                return True
            if "backend='inductor'" in message:
                return True
            if 'triton_key' in message:
                return True
            current = current.__cause__ if current.__cause__ is not None else current.__context__
        return False

    def _disable_torch_compile_runtime_fallback(self, error: Exception) -> bool:
        if not self._torch_compile_active:
            return False
        if self._uncompiled_model is None:
            return False
        if isinstance(self._model, (DDP, nn.DataParallel)):
            return False
        if not self._is_torch_compile_runtime_failure(error):
            return False
        self._model = self._uncompiled_model
        self._torch_compile_active = False
        self._bus.put(['logging', f'torch.compile disabled at runtime (fallback): {error}'])
        return True

    def _forward_model(self, image: Any) -> torch.Tensor:
        try:
            return self._model(image)
        except Exception as error:
            if self._disable_torch_compile_runtime_fallback(error):
                return self._model(image)
            raise

    @staticmethod
    def _create_grad_scaler(device_type: str, enabled: bool) -> Any:
        amp_module = getattr(torch, 'amp', None)
        grad_scaler_cls = getattr(amp_module, 'GradScaler', None) if amp_module is not None else None
        if grad_scaler_cls is not None:
            try:
                return grad_scaler_cls(device_type, enabled=enabled)
            except TypeError:
                try:
                    return grad_scaler_cls(enabled=enabled)
                except TypeError:
                    pass
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def run(self):
        try:
            requested_multi_gpu_mode = self._multi_gpu_mode
            if self._should_use_ddp():
                self._run_ddp(world_size=torch.cuda.device_count())
                return

            if (
                requested_multi_gpu_mode == 'distributeddataparallel'
                and self._has_multi_gpu_cuda()
                and os.name == 'nt'
            ):
                gpu_count = int(torch.cuda.device_count())
                self._bus.put([
                    'logging',
                    f'DDP requested on Windows; using nn.DataParallel on {gpu_count} GPU as fallback.',
                ])
                self._model = nn.DataParallel(self._model)
                device = torch.device('cuda:0')
            elif self._should_use_data_parallel():
                gpu_count = int(torch.cuda.device_count())
                self._bus.put([
                    'logging',
                    f'Включен режим multi GPU: nn.DataParallel на {gpu_count} GPU.',
                ])
                self._model = nn.DataParallel(self._model)
                device = torch.device('cuda:0')
            else:
                if requested_multi_gpu_mode != 'off':
                    self._log_multi_gpu_unavailable(requested_multi_gpu_mode)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model.to(device)
            self._run_impl(device=device, rank=0, world_size=1, distributed=False)
        except Exception as error:
            self._bus.put(['error', f'Критическая ошибка обучения: {error}'])
            raise

    @staticmethod
    def _build_train_loop_strides(train_size: int, log_update_frequency: int = 0) -> _TrainLoopStrides:
        if log_update_frequency > 0:
            metric_stride = int(log_update_frequency)
            preview_stride = 2*int(log_update_frequency)
            log_stride = int(log_update_frequency)
        else:
             metric_stride = 10
             preview_stride = 20
             log_stride = 10
        return _TrainLoopStrides(
            metric=metric_stride,
            progress=metric_stride,
            log=log_stride,
            preview=preview_stride,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = str(os.getenv(name, '1' if default else '0')).strip().lower()
        return value in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _env_int(name: str, default: int, minimum: int = 0) -> int:
        raw = os.getenv(name)
        if raw is None:
            return max(minimum, int(default))
        try:
            return max(minimum, int(raw))
        except Exception:
            return max(minimum, int(default))

    def _resolve_training_profiler_config(self) -> _TrainingProfilerConfig:
        enabled = bool(TRAINING_PROFILER_ENABLED)
        max_batches = self._env_int('NEURALIMAGE_TRAIN_PROFILE_STEPS', 40, minimum=1)
        row_limit = self._env_int('NEURALIMAGE_TRAIN_PROFILE_ROW_LIMIT', 15, minimum=5)
        output_dir_name = str(os.getenv('NEURALIMAGE_TRAIN_PROFILE_DIR', 'profiles')).strip() or 'profiles'
        return _TrainingProfilerConfig(
            enabled=enabled,
            max_batches=max_batches,
            record_shapes=self._env_bool('NEURALIMAGE_TRAIN_PROFILE_RECORD_SHAPES', True),
            profile_memory=self._env_bool('NEURALIMAGE_TRAIN_PROFILE_MEMORY', True),
            with_stack=self._env_bool('NEURALIMAGE_TRAIN_PROFILE_WITH_STACK', False),
            row_limit=row_limit,
            output_dir_name=output_dir_name,
        )

    def _start_training_profiler(
        self,
        *,
        device: torch.device,
        train_size: int,
        is_main_process: bool,
        distributed: bool,
    ) -> _ActiveTrainProfiler | None:
        cfg = self._training_profiler_config
        if not cfg.enabled or not is_main_process:
            return None
        if distributed:
            self._bus.put(['logging', 'Профилирование отключено в режиме DDP (distributed).'])
            return None
        if train_size <= 0:
            return None
        if not hasattr(torch, 'profiler'):
            self._bus.put(['logging', 'torch.profiler недоступен, профилирование пропущено.'])
            return None

        steps = max(1, min(cfg.max_batches, int(train_size)))
        warmup_steps = 1 if steps > 1 else 0
        active_steps = max(1, steps - warmup_steps)
        wait_steps = 0
        profile_dir = self._save_path.parent / cfg.output_dir_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        trace_path = profile_dir / f'train_profile_{timestamp}.json'

        activities: list[Any] = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        def _on_trace_ready(prof: Any) -> None:
            try:
                prof.export_chrome_trace(str(trace_path))
            except Exception as export_error:
                self._bus.put(['logging', f'Не удалось сохранить trace профайлера: {export_error}'])

        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1),
            on_trace_ready=_on_trace_ready,
            record_shapes=cfg.record_shapes,
            profile_memory=cfg.profile_memory,
            with_stack=cfg.with_stack,
        )
        profiler.__enter__()
        sort_key = 'self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
        self._bus.put([
            'logging',
            (
                f'Профилирование обучения включено: шагов={steps}, warmup={warmup_steps}, active={active_steps}, '
                f'trace={trace_path.name}'
            ),
        ])
        return _ActiveTrainProfiler(
            profiler=profiler,
            steps_left=steps,
            trace_path=trace_path,
            summary_sort_key=sort_key,
            row_limit=cfg.row_limit,
        )

    def _step_training_profiler(self, active_profiler: _ActiveTrainProfiler | None) -> None:
        if active_profiler is None or active_profiler.is_closed:
            return
        if active_profiler.steps_left <= 0:
            self._stop_training_profiler(active_profiler)
            return
        try:
            active_profiler.profiler.step()
        finally:
            active_profiler.steps_left -= 1
            if active_profiler.steps_left <= 0:
                self._stop_training_profiler(active_profiler)

    def _stop_training_profiler(self, active_profiler: _ActiveTrainProfiler | None) -> None:
        if active_profiler is None or active_profiler.is_closed:
            return
        try:
            summary = active_profiler.profiler.key_averages().table(
                sort_by=active_profiler.summary_sort_key,
                row_limit=active_profiler.row_limit,
            )
            self._bus.put(['logging', f'Сводка профайлера (top {active_profiler.row_limit}):\n{summary}'])
            self._bus.put(['logging', f'Профиль сохранен: {active_profiler.trace_path}'])
        except Exception as profile_error:
            self._bus.put(['logging', f'Не удалось получить сводку профайлера: {profile_error}'])
        finally:
            try:
                active_profiler.profiler.__exit__(None, None, None)
            finally:
                active_profiler.is_closed = True

    def _prepare_training_device(self, device: torch.device, *, is_main_process: bool, distributed: bool) -> None:
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if not distributed:
            self._try_compile_model(is_main_process=is_main_process, device=device)

    def _resolve_train_loader_context(self) -> tuple[int, Any, bool, _TrainLoopStrides]:
        train_size = self._safe_loader_len(self._train_dataloader)
        train_sampler = getattr(self._train_dataloader, 'sampler', None) if self._train_dataloader is not None else None
        supports_loss_aware_sampling = hasattr(train_sampler, 'update_batch_losses')
        strides = self._build_train_loop_strides(train_size, self._log_update_frequency)
        return train_size, train_sampler, supports_loss_aware_sampling, strides

    def _publish_training_start_metrics(self, train_size: int) -> None:
        self._bus.put(['metrics', {'type': 'train_epoch_progress', 'current': 0, 'total': int(self._epochs)}])
        self._bus.put(['metrics', {'type': 'train_batch_progress', 'current': 0, 'total': int(train_size)}])
        memory_payload = _collect_memory_metrics()
        if memory_payload is not None:
            self._bus.put(['metrics', {'type': 'system_memory', **memory_payload}])

    def _log_training_start(self, *, is_main_process: bool, device: torch.device) -> None:
        if is_main_process:
            self._bus.put(['logging', f'Training device detected: {device}'])
        self._bus.put(['logging', 'Preparing data and starting the training loop'])

    def _log_mixed_precision_mode(
        self,
        *,
        resolved_mode: MixedPrecisionMode,
        device: torch.device,
    ) -> None:
        if resolved_mode != self._mixed_precision:
            self._bus.put([
                'logging',
                (
                    f'Mixed precision mode "{self._mixed_precision.value}" is unavailable on {device.type}. '
                    f'Using mode "{resolved_mode.value}".'
                ),
            ])
            return
        self._bus.put(['logging', f'Mixed precision mode: {resolved_mode.value}.'])

    def _log_loss_configuration(self) -> None:
        resolved_loss_mode = self._resolved_loss_function()
        resolved_loss_weights = self._resolved_loss_term_weights()
        if (not getattr(self, '_loss_term_weights', {})) and resolved_loss_mode != self._loss_function:
            self._bus.put(['logging', f'Unknown loss "{self._loss_function}". Using "bce".'])
        self._bus.put(['logging', f'Loss formula: {format_loss_formula(resolved_loss_weights)}'])
        for loss_mode, coefficient in resolved_loss_weights.items():
            self._bus.put(['logging', f'Loss term: {coefficient:.2f} * {loss_mode}.'])
        if self._hard_pixel_mining_enabled():
            supported_terms = [
                loss_mode for loss_mode in resolved_loss_weights
                if self._loss_supports_hard_pixel_mining(loss_mode)
            ]
            if supported_terms:
                self._bus.put([
                    'logging',
                    (
                        f'Hard pixel mining is enabled: keep_ratio={self._resolved_hard_pixel_keep_ratio():.2f}. '
                        f'Applied only to terms with pointwise loss: {", ".join(supported_terms)}.'
                    ),
                ])
            else:
                self._bus.put([
                    'logging',
                    (
                        'Hard pixel mining is enabled but ignored because all active loss terms '
                        'lack a pointwise pixel term.'
                    ),
                ])
        self._bus.put(['logging', 'Loss normalization: per-sample mean (batch-size invariant).'])

    def _log_sampling_configuration(self, *, train_sampler: Any, supports_loss_aware_sampling: bool) -> None:
        if self._skip_uniform_labels:
            self._bus.put(['logging', 'Skipping uniform 0/1 labels is enabled for training batches.'])
        if not supports_loss_aware_sampling:
            return
        self._bus.put([
            'logging',
            (
                f'Loss-aware sampling is enabled: '
                f'strength={float(getattr(train_sampler, "strength", 0.0)):.2f}, '
                f'ema_alpha={float(getattr(train_sampler, "ema_alpha", 0.0)):.2f}.'
            ),
        ])

    @staticmethod
    def _build_autocast_context(
        *,
        device_type: str,
        autocast_dtype: torch.dtype | None,
    ) -> Callable[[], ContextManager[Any]]:
        if autocast_dtype is None:
            return lambda: nullcontext()
        return lambda: torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=True)

    def _log_warmup_configuration(self, scheduler: Any) -> None:
        if scheduler is None:
            return
        self._bus.put([
            'logging',
            (
                f'Warmup enabled: epochs={int(self._warmup_params.epochs)}, '
                f'start_factor={float(self._warmup_params.start_factor):.4f}.'
            ),
        ])

    def _log_scheduler_configuration(self, scheduler: Any, scheduler_step_mode: str) -> None:
        if scheduler is None:
            return
        scheduler_name = self._resolved_scheduler_name()
        scheduler_params = getattr(self, '_scheduler_params', None) or SchedulerParameters()
        if scheduler_name == SchedulerName.reduce_on_plateau:
            self._bus.put([
                'logging',
                (
                    'LR scheduler enabled: ReduceLROnPlateau '
                    f'(factor={float(getattr(scheduler_params, "plateau_factor", 0.5)):.4f}, '
                    f'patience={int(getattr(scheduler_params, "plateau_patience", 3))}, '
                    f'threshold={float(getattr(scheduler_params, "plateau_threshold", 1e-4)):.6f}, '
                    f'min_lr={float(getattr(scheduler_params, "plateau_min_lr", 1e-6)):.6f}, '
                    f'cooldown={int(getattr(scheduler_params, "plateau_cooldown", 0))}).'
                ),
            ])
            return
        if scheduler_name == SchedulerName.cosine_annealing:
            self._bus.put([
                'logging',
                (
                    'LR scheduler enabled: CosineAnnealingLR '
                    f'(T_max={int(getattr(scheduler_params, "cosine_t_max", 10))}, '
                    f'eta_min={float(getattr(scheduler_params, "cosine_eta_min", 1e-6)):.6f}).'
                ),
            ])
            return
        if scheduler_name == SchedulerName.one_cycle:
            self._bus.put([
                'logging',
                (
                    'LR scheduler enabled: OneCycleLR '
                    f'(max_lr={float(getattr(scheduler_params, "one_cycle_max_lr", 1e-3)):.6f}, '
                    f'pct_start={float(getattr(scheduler_params, "one_cycle_pct_start", 0.3)):.3f}, '
                    f'anneal={self._resolve_one_cycle_anneal_strategy(getattr(scheduler_params, "one_cycle_anneal_strategy", "cos"))}, '
                    f'div_factor={float(getattr(scheduler_params, "one_cycle_div_factor", 25.0)):.2f}, '
                    f'final_div_factor={float(getattr(scheduler_params, "one_cycle_final_div_factor", 10000.0)):.2f}, '
                    f'three_phase={bool(getattr(scheduler_params, "one_cycle_three_phase", False))}).'
                ),
            ])
            return
        if scheduler_name == SchedulerName.step_lr:
            self._bus.put([
                'logging',
                (
                    'LR scheduler enabled: StepLR '
                    f'(step_size={int(getattr(scheduler_params, "step_lr_step_size", 10))}, '
                    f'gamma={float(getattr(scheduler_params, "step_lr_gamma", 0.1)):.4f}, '
                    f'step_mode={scheduler_step_mode}).'
                ),
            ])

    def _create_run_context(self, device: torch.device, is_main_process: bool, distributed: bool) -> _RunContext:
        self._prepare_training_device(device, is_main_process=is_main_process, distributed=distributed)

        bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = self._create_optimizer()
        train_size, train_sampler, supports_loss_aware_sampling, strides = self._resolve_train_loader_context()
        self._publish_training_start_metrics(train_size)
        self._log_training_start(is_main_process=is_main_process, device=device)

        resolved_mp_mode, autocast_dtype, scaler_enabled = self._resolve_mixed_precision(device)
        self._log_mixed_precision_mode(resolved_mode=resolved_mp_mode, device=device)
        self._log_loss_configuration()
        self._log_sampling_configuration(
            train_sampler=train_sampler,
            supports_loss_aware_sampling=supports_loss_aware_sampling,
        )

        scaler = self._create_grad_scaler(device_type=device.type, enabled=scaler_enabled)
        autocast_ctx = self._build_autocast_context(device_type=device.type, autocast_dtype=autocast_dtype)
        warmup_scheduler, warmup_total_steps = self._create_warmup_scheduler(optimizer, train_size)
        scheduler, scheduler_step_mode = self._create_lr_scheduler(
            optimizer,
            train_steps_per_epoch=train_size,
        )
        self._log_warmup_configuration(warmup_scheduler)
        self._log_scheduler_configuration(scheduler, scheduler_step_mode)

        return _RunContext(
            bce_criterion=bce_criterion,
            optimizer=optimizer,
            scaler=scaler,
            autocast_ctx=autocast_ctx,
            scheduler=scheduler,
            train_size=train_size,
            train_sampler=train_sampler,
            supports_loss_aware_sampling=supports_loss_aware_sampling,
            strides=strides,
            scheduler_step_mode=scheduler_step_mode,
            warmup_scheduler=warmup_scheduler,
            warmup_total_steps=warmup_total_steps,
        )

    def _restore_training_state(self, run_context: _RunContext) -> tuple[int, _EarlyStoppingState, _EarlyStoppingConfig]:
        start_epoch, checkpoint = self._load_checkpoint_if_available(
            run_context.optimizer,
            run_context.scaler,
            run_context.warmup_scheduler,
            run_context.scheduler,
        )
        early_stopping_state = _EarlyStoppingState(
            best_loss=(
                float(checkpoint['early_stopping_best_loss'])
                if checkpoint.get('early_stopping_best_loss') is not None
                else None
            ),
            bad_epochs=int(checkpoint.get('early_stopping_bad_epochs', 0)),
            best_epoch=int(checkpoint.get('early_stopping_best_epoch', 0)),
            best_model_state=checkpoint.get('early_stopping_best_model_state'),
            best_threshold=float(checkpoint.get('early_stopping_best_threshold', self._recommended_inference_threshold)),
        )

        has_validation = bool(self._val_dataloader is not None)
        if has_validation and self._val_dataloader is not None:
            has_validation = len(cast(Sized, self._val_dataloader.dataset)) > 0
        early_stopping_config = _EarlyStoppingConfig(
            enabled=bool(self._early_stopping_params.enabled and has_validation),
            patience=max(0, int(self._early_stopping_params.patience)),
            min_delta=max(0.0, float(self._early_stopping_params.min_delta)),
        )
        if self._early_stopping_params.enabled and not has_validation:
            self._bus.put([
                'logging',
                'Early stopping включен, но валидационный датасет отсутствует. Early stopping отключен.',
            ])
        elif early_stopping_config.enabled:
            self._bus.put([
                'logging',
                (
                    f'Early stopping включен: patience={early_stopping_config.patience}, '
                    f'min_delta={early_stopping_config.min_delta:.6f}.'
                ),
            ])

        target_epochs = self._resolve_target_epochs(start_epoch)
        if target_epochs != self._epochs:
            self._bus.put([
                'logging',
                (
                    f'Режим дообучения: продолжение с эпохи {start_epoch}. '
                    f'Будет добавлено {self._epochs} эпох (до {target_epochs}).'
                ),
            ])
            self._epochs = target_epochs
            self._bus.put([
                'metrics',
                {'type': 'train_epoch_progress', 'current': int(start_epoch), 'total': int(self._epochs)},
            ])

        return start_epoch, early_stopping_state, early_stopping_config

    def _publish_epoch_start(self, epoch: int, run_context: _RunContext, distributed: bool) -> None:
        self._model.train()
        if distributed:
            sampler = getattr(self._train_dataloader, 'sampler', None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
        self._bus.put(['logging', f'Начало эпохи [{epoch + 1}/{self._epochs}]'])
        current_lr = float(run_context.optimizer.param_groups[0]['lr'])
        self._bus.put(['logging', f'Текущий learning rate: {current_lr:.8f}'])
        self._bus.put([
            'metrics',
            {'type': 'train_epoch_progress', 'current': int(epoch + 1), 'total': int(self._epochs)},
        ])
        self._bus.put([
            'metrics',
            {'type': 'train_batch_progress', 'current': 0, 'total': int(run_context.train_size)},
        ])
        memory_payload = _collect_memory_metrics()
        if memory_payload is not None:
            self._bus.put(['metrics', {'type': 'system_memory', **memory_payload, 'epoch': int(epoch + 1)}])
        self._bus.put(['logging', ' '])

    @staticmethod
    def _split_batch(batch: Any) -> tuple[Any, Any, Any]:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            data, target, sample_indices = batch
            return data, target, sample_indices
        data, target = batch
        return data, target, None

    @staticmethod
    def _move_batch_input_to_device(data: Any, device: torch.device) -> Any:
        if isinstance(data, Mapping):
            moved: dict[str, Any] = {}
            for key, value in data.items():
                if torch.is_tensor(value):
                    moved[str(key)] = value.to(device, non_blocking=True)
                else:
                    moved[str(key)] = value
            return moved
        if torch.is_tensor(data):
            return data.to(device, non_blocking=True)
        raise TypeError(f'Unsupported batch input type: {type(data)!r}')

    @staticmethod
    def _create_cuda_timing_events(device: torch.device, count: int) -> tuple[Any, ...] | None:
        if device.type != 'cuda':
            return None
        return tuple(torch.cuda.Event(enable_timing=True) for _ in range(max(0, int(count))))

    @staticmethod
    def _record_cuda_timing_event(event: Any, device: torch.device) -> None:
        if event is None or device.type != 'cuda':
            return
        event.record(torch.cuda.current_stream(device=device))

    @staticmethod
    def _elapsed_cuda_event_ms(start_event: Any, end_event: Any) -> float:
        return float(start_event.elapsed_time(end_event))

    @staticmethod
    def _loss_value_to_float(loss: torch.Tensor | float) -> float:
        if torch.is_tensor(loss):
            return float(loss.detach().to(device='cpu', dtype=torch.float32).item())
        return float(loss)

    @staticmethod
    def _extract_local_image(data: Any) -> torch.Tensor:
        if isinstance(data, Mapping):
            local_image = data.get('local_image')
            if not torch.is_tensor(local_image):
                raise TypeError('Expected "local_image" tensor in model input mapping.')
            return local_image
        if not torch.is_tensor(data):
            raise TypeError(f'Unsupported local batch input type: {type(data)!r}')
        return data

    @staticmethod
    def _extract_context_image(data: Any) -> torch.Tensor | None:
        if not isinstance(data, Mapping):
            return None
        context_image = data.get('context_image')
        if context_image is None:
            return None
        if not torch.is_tensor(context_image):
            raise TypeError('Expected "context_image" tensor in model input mapping.')
        return context_image

    @staticmethod
    def _filter_batch_input(data: Any, valid_mask: torch.Tensor) -> Any:
        if isinstance(data, Mapping):
            filtered: dict[str, Any] = {}
            for key, value in data.items():
                if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(valid_mask.shape[0]):
                    filtered[str(key)] = value[valid_mask.detach().to(device=value.device)]
                else:
                    filtered[str(key)] = value
            return filtered
        if torch.is_tensor(data):
            return data[valid_mask]
        raise TypeError(f'Unsupported batch input type: {type(data)!r}')

    def _filter_uniform_batch_samples(
        self,
        image: Any,
        label: torch.Tensor,
        sample_indices: Any,
    ) -> tuple[Any, torch.Tensor, Any, int, bool]:
        if not self._skip_uniform_labels:
            return image, label, sample_indices, 0, True
        # Binarize before uniformity detection so labels normalized as 255/256
        # are still treated as "all ones".
        label_flat = self._sanitize_labels_for_loss(label).reshape(label.shape[0], -1)
        label_bin = label_flat >= 0.5
        is_all_zero = (~label_bin).all(dim=1)
        is_all_one = label_bin.all(dim=1)
        valid_mask = ~(is_all_zero | is_all_one)
        skipped_here = int((~valid_mask).sum().item())
        if not bool(valid_mask.any()):
            return image, label, sample_indices, skipped_here, False

        image = self._filter_batch_input(image, valid_mask)
        label = label[valid_mask]
        if sample_indices is None:
            return image, label, None, skipped_here, True
        if torch.is_tensor(sample_indices):
            sample_indices = sample_indices[valid_mask.detach().to(device=sample_indices.device)]
        else:
            sample_indices_tensor = torch.as_tensor(sample_indices)
            sample_indices = sample_indices_tensor[valid_mask.detach().to(device='cpu')]
        return image, label, sample_indices, skipped_here, True

    @staticmethod
    def _permute_sample_indices(sample_indices: Any, permutation: torch.Tensor) -> Any:
        if sample_indices is None:
            return None
        if torch.is_tensor(sample_indices):
            return sample_indices[permutation.detach().to(device=sample_indices.device)]
        sample_indices_tensor = torch.as_tensor(sample_indices, dtype=torch.long)
        return sample_indices_tensor[permutation.detach().to(device='cpu')]

    def _apply_mixup_to_batch(
        self,
        image: Any,
        label: torch.Tensor,
        sample_indices: Any,
    ) -> tuple[Any, torch.Tensor, Any, torch.Tensor | None]:
        enabled, probability, alpha = self._resolved_mixup_parameters()
        local_image = self._extract_local_image(image)
        batch_size = int(local_image.size(0))
        if (not enabled) or batch_size <= 1 or probability <= 0.0 or alpha <= 0.0:
            return image, label, None, None
        if float(np.random.random()) > probability:
            return image, label, None, None

        lambda_value = float(np.random.beta(alpha, alpha))
        lambda_value = float(min(max(lambda_value, 0.0), 1.0))
        permutation = torch.randperm(batch_size, device=local_image.device)
        order = torch.arange(batch_size, device=local_image.device)
        if bool((permutation == order).any()):
            permutation = torch.roll(order, shifts=1)

        lambda_tensor = local_image.new_full((batch_size,), lambda_value)
        lambda_view = lambda_tensor.view(batch_size, 1, 1, 1)
        mixed_image = self._mixup_batch_input(image, permutation, lambda_view)
        mixed_label = (lambda_view * label) + ((1.0 - lambda_view) * label[permutation])
        pair_indices = self._permute_sample_indices(sample_indices, permutation)
        return mixed_image, mixed_label, pair_indices, lambda_tensor

    @staticmethod
    def _mixup_batch_input(data: Any, permutation: torch.Tensor, lambda_view: torch.Tensor) -> Any:
        if isinstance(data, Mapping):
            mixed: dict[str, Any] = {}
            for key, value in data.items():
                if torch.is_tensor(value) and value.ndim >= 4 and int(value.shape[0]) == int(permutation.shape[0]):
                    mixed[str(key)] = (lambda_view * value) + ((1.0 - lambda_view) * value[permutation])
                else:
                    mixed[str(key)] = value
            return mixed
        if torch.is_tensor(data):
            return (lambda_view * data) + ((1.0 - lambda_view) * data[permutation])
        raise TypeError(f'Unsupported batch input type: {type(data)!r}')

    def _apply_cutout_to_batch(self, image: torch.Tensor) -> torch.Tensor:
        enabled, probability, holes, size_ratio = self._resolved_cutout_parameters()
        if (not enabled) or probability <= 0.0 or size_ratio <= 0.0:
            return image

        batch_size, _channels, height, width = image.shape
        max_cutout_height = max(1, min(height, int(round(height * size_ratio))))
        max_cutout_width = max(1, min(width, int(round(width * size_ratio))))
        if max_cutout_height <= 0 or max_cutout_width <= 0:
            return image

        cutout_image = image.clone()
        for sample_index in range(batch_size):
            if float(np.random.random()) > probability:
                continue
            for _ in range(holes):
                cutout_height = (
                    1
                    if max_cutout_height == 1
                    else int(torch.randint(1, max_cutout_height + 1, (1,), device=image.device).item())
                )
                cutout_width = (
                    1
                    if max_cutout_width == 1
                    else int(torch.randint(1, max_cutout_width + 1, (1,), device=image.device).item())
                )
                max_top = max(0, height - cutout_height)
                max_left = max(0, width - cutout_width)
                top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,), device=image.device).item())
                left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,), device=image.device).item())
                fill_color = torch.rand((_channels, 1, 1), device=image.device, dtype=image.dtype)
                cutout_image[sample_index, :, top:top + cutout_height, left:left + cutout_width] = fill_color
        return cutout_image

    def _apply_random_artifacts_to_batch(self, image: torch.Tensor) -> torch.Tensor:
        enabled, probability, count, size_ratio, artifact_types = self._resolved_random_artifact_parameters()
        if (not enabled) or probability <= 0.0 or size_ratio <= 0.0 or not artifact_types:
            return image

        batch_size, channels, height, width = image.shape
        min_artifact_height, max_artifact_height, min_artifact_width, max_artifact_width = (
            self._random_artifact_size_bounds(
                image_height=int(height),
                image_width=int(width),
                size_ratio=size_ratio,
            )
        )
        if max_artifact_height <= 0 or max_artifact_width <= 0:
            return image

        artifact_bank = self._ensure_random_artifact_bank(image=image, artifact_types=artifact_types)
        artifact_image = image.clone()
        for sample_index in range(batch_size):
            if float(np.random.random()) > probability:
                continue
            for _ in range(count):
                artifact_height = (
                    int(min_artifact_height)
                    if max_artifact_height == min_artifact_height
                    else int(np.random.randint(min_artifact_height, max_artifact_height + 1))
                )
                artifact_width = (
                    int(min_artifact_width)
                    if max_artifact_width == min_artifact_width
                    else int(np.random.randint(min_artifact_width, max_artifact_width + 1))
                )
                max_top = max(0, height - artifact_height)
                max_left = max(0, width - artifact_width)
                top = 0 if max_top == 0 else int(np.random.randint(0, max_top + 1))
                left = 0 if max_left == 0 else int(np.random.randint(0, max_left + 1))

                if artifact_bank is not None:
                    overlay, alpha = artifact_bank.acquire(
                        height=int(artifact_height),
                        width=int(artifact_width),
                    )
                else:
                    overlay, alpha = generate_random_artifact_patch(
                        int(channels),
                        int(artifact_height),
                        int(artifact_width),
                        device=torch.device('cpu'),
                        dtype=torch.float32,
                        artifact_types=artifact_types,
                    )
                overlay = overlay.to(device=image.device, dtype=image.dtype, non_blocking=True)
                alpha = alpha.to(device=image.device, dtype=image.dtype, non_blocking=True)
                patch = artifact_image[sample_index, :, top:top + artifact_height, left:left + artifact_width]
                artifact_image[sample_index, :, top:top + artifact_height, left:left + artifact_width] = torch.clamp(
                    (patch * (1.0 - alpha)) + (overlay * alpha),
                    min=0.0,
                    max=1.0,
                )
        return artifact_image

    def _apply_cutout_to_input(self, image: Any) -> Any:
        if not isinstance(image, Mapping):
            return self._apply_cutout_to_batch(image)
        local_image = image.get('local_image')
        if not torch.is_tensor(local_image):
            return image
        updated = dict(image)
        updated['local_image'] = self._apply_cutout_to_batch(local_image)
        return updated

    def _apply_random_artifacts_to_input(self, image: Any) -> Any:
        if not isinstance(image, Mapping):
            return self._apply_random_artifacts_to_batch(image)
        local_image = image.get('local_image')
        if not torch.is_tensor(local_image):
            return image
        updated = dict(image)
        updated['local_image'] = self._apply_random_artifacts_to_batch(local_image)
        return updated

    def _apply_training_batch_augmentations(
        self,
        image: Any,
        label: torch.Tensor,
        sample_indices: Any,
    ) -> tuple[Any, torch.Tensor, Any, torch.Tensor | None]:
        image, label, mixup_pair_indices, mixup_lambdas = self._apply_mixup_to_batch(
            image,
            label,
            sample_indices,
        )
        image = self._apply_cutout_to_input(image)
        image = self._apply_random_artifacts_to_input(image)
        return image, label, mixup_pair_indices, mixup_lambdas

    def _publish_batch_preview(
        self,
        *,
        epoch: int,
        batch_index: int,
        train_size: int,
        data: Any,
        target: Any,
        outputs: torch.Tensor,
        preview_stride: int,
    ) -> None:
        if not self._show_batch_preview:
            return
        if not ((batch_index % preview_stride == 0) or (batch_index == train_size - 1)):
            return
        preview_image = self._tensor_to_preview_array(data[0])
        preview_label = self._tensor_to_preview_array(target[0])
        preview_outputs = self._tensor_to_preview_array(torch.sigmoid(outputs[0].detach()))
        self._bus.put([
            'metrics',
            {
                'type': 'train_batch_preview',
                'epoch': int(epoch + 1),
                'batch_index': int(batch_index + 1),
                'image': preview_image,
                'label': preview_label,
                'outputs': preview_outputs,
            },
        ])

    def _update_loss_aware_sampling(
        self,
        *,
        run_context: _RunContext,
        sample_indices: Any,
        per_sample_loss: torch.Tensor,
        mixup_pair_indices: Any = None,
        mixup_lambdas: torch.Tensor | None = None,
    ) -> None:
        if not run_context.supports_loss_aware_sampling or sample_indices is None:
            return
        if torch.is_tensor(sample_indices):
            sample_idx_tensor = sample_indices.detach().to(device='cpu', dtype=torch.long).flatten()
        else:
            sample_idx_tensor = torch.as_tensor(sample_indices, dtype=torch.long).flatten()
        sample_loss_tensor = per_sample_loss.detach().to(device='cpu', dtype=torch.float32).flatten()
        sample_loss_tensor = torch.nan_to_num(sample_loss_tensor, nan=1.0, posinf=50.0, neginf=0.0)
        if sample_idx_tensor.numel() != sample_loss_tensor.numel():
            return

        sampler = cast(_SupportsLossAwareSampling, run_context.train_sampler)
        if mixup_pair_indices is None or mixup_lambdas is None:
            sampler.update_batch_losses(sample_idx_tensor, sample_loss_tensor)
            return

        lambda_tensor = mixup_lambdas.detach().to(device='cpu', dtype=torch.float32).flatten()
        if lambda_tensor.numel() == 1 and sample_loss_tensor.numel() > 1:
            lambda_tensor = lambda_tensor.expand_as(sample_loss_tensor)
        if lambda_tensor.numel() != sample_loss_tensor.numel():
            sampler.update_batch_losses(sample_idx_tensor, sample_loss_tensor)
            return

        if torch.is_tensor(mixup_pair_indices):
            pair_idx_tensor = mixup_pair_indices.detach().to(device='cpu', dtype=torch.long).flatten()
        else:
            pair_idx_tensor = torch.as_tensor(mixup_pair_indices, dtype=torch.long).flatten()
        if pair_idx_tensor.numel() != sample_loss_tensor.numel():
            sampler.update_batch_losses(sample_idx_tensor, sample_loss_tensor)
            return

        sampler.update_batch_losses(sample_idx_tensor, sample_loss_tensor * lambda_tensor)
        sampler.update_batch_losses(pair_idx_tensor, sample_loss_tensor * (1.0 - lambda_tensor))

    def _has_non_finite_gradients(self) -> bool:
        for param in self._model.parameters():
            gradient = param.grad
            if gradient is None:
                continue
            if not self._is_finite_tensor(gradient):
                return True
        return False

    def _prepare_train_batch(
        self,
        *,
        batch: Any,
        device: torch.device,
        data_wait_started_at: float,
    ) -> tuple[_PreparedTrainBatch | None, int]:
        data, target, sample_indices = self._split_batch(batch)
        inputs = self._move_batch_input_to_device(data, device)
        label = target.to(device, non_blocking=True)
        inputs, label, sample_indices, skipped_here, has_valid_samples = self._filter_uniform_batch_samples(
            inputs,
            label,
            sample_indices,
        )
        batch_start = data_wait_started_at
        data_wait_ms = (time.perf_counter() - data_wait_started_at) * 1000.0
        if not has_valid_samples:
            return None, skipped_here
        has_batch_augmentations = self._has_training_batch_augmentations()
        augmentation_ms = 0.0
        augmentation_events = None
        if has_batch_augmentations:
            augmentation_start = time.perf_counter()
            augmentation_events = self._create_cuda_timing_events(device, 2)
            if augmentation_events is not None:
                augmentation_start_event, augmentation_end_event = augmentation_events
                self._record_cuda_timing_event(augmentation_start_event, device)
        inputs, label, mixup_pair_indices, mixup_lambdas = self._apply_training_batch_augmentations(
            inputs,
            label,
            sample_indices,
        )
        if augmentation_events is not None:
            self._record_cuda_timing_event(augmentation_end_event, device)
            augmentation_end_event.synchronize()
            augmentation_ms = self._elapsed_cuda_event_ms(augmentation_start_event, augmentation_end_event)
        elif has_batch_augmentations:
            augmentation_ms = (time.perf_counter() - augmentation_start) * 1000.0
        image = self._extract_local_image(inputs)
        context_image = self._extract_context_image(inputs)
        return (
            _PreparedTrainBatch(
                data=data,
                target=target,
                sample_indices=sample_indices,
                mixup_pair_indices=mixup_pair_indices,
                mixup_lambdas=mixup_lambdas,
                inputs=inputs,
                image=image,
                context_image=context_image,
                label=label,
                batch_start=batch_start,
                data_wait_ms=data_wait_ms,
                augmentation_ms=augmentation_ms,
            ),
            skipped_here,
        )

    def _run_train_step(
        self,
        *,
        run_context: _RunContext,
        batch: _PreparedTrainBatch,
    ) -> _TrainStepResult | None:
        run_context.optimizer.zero_grad(set_to_none=True)

        step_device = batch.image.device
        forward_start = time.perf_counter()
        step_events = self._create_cuda_timing_events(step_device, 4)
        if step_events is not None:
            forward_start_event, forward_end_event, backward_end_event, optimizer_end_event = step_events
            self._record_cuda_timing_event(forward_start_event, step_device)
        backward_denominator = float(max(1, int(batch.image.size(0))))
        with run_context.autocast_ctx():
            outputs = self._sanitize_outputs_for_loss(self._forward_model(batch.inputs))
            per_sample_loss = self._compute_per_sample_loss(
                outputs,
                batch.label,
                run_context.bce_criterion,
                apply_pixel_mining=True,
            )
            metric_loss = per_sample_loss.mean()
            loss = per_sample_loss.sum() / backward_denominator
        if (not self._is_finite_tensor(loss)) or (not self._is_finite_tensor(metric_loss)):
            # Retry in full precision to avoid occasional AMP overflow/underflow instability.
            with nullcontext():
                retry_inputs = (
                    batch.image.float()
                    if not isinstance(batch.inputs, Mapping)
                    else {
                        key: value.float() if torch.is_tensor(value) else value
                        for key, value in batch.inputs.items()
                    }
                )
                outputs = self._sanitize_outputs_for_loss(self._forward_model(retry_inputs))
                per_sample_loss = self._compute_per_sample_loss(
                    outputs,
                    batch.label,
                    run_context.bce_criterion,
                    apply_pixel_mining=True,
                )
                metric_loss = per_sample_loss.mean()
                loss = per_sample_loss.sum() / backward_denominator
            if (not self._is_finite_tensor(loss)) or (not self._is_finite_tensor(metric_loss)):
                self._bus.put(['logging', 'Training warning: non-finite loss detected, batch skipped.'])
                run_context.optimizer.zero_grad(set_to_none=True)
                return None
        if step_events is not None:
            self._record_cuda_timing_event(forward_end_event, step_device)
            forward_ms = 0.0
        else:
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

        backward_start = time.perf_counter()
        run_context.scaler.scale(loss).backward()
        run_context.scaler.unscale_(run_context.optimizer)
        if self._has_non_finite_gradients():
            self._bus.put(['logging', 'Training warning: non-finite gradients detected, optimizer step skipped.'])
            run_context.optimizer.zero_grad(set_to_none=True)
            run_context.scaler.update()
            if step_events is not None:
                self._record_cuda_timing_event(backward_end_event, step_device)
                backward_end_event.synchronize()
                forward_ms = self._elapsed_cuda_event_ms(forward_start_event, forward_end_event)
                backward_ms = self._elapsed_cuda_event_ms(forward_end_event, backward_end_event)
            else:
                backward_ms = (time.perf_counter() - backward_start) * 1000.0
            return _TrainStepResult(
                outputs=outputs,
                per_sample_loss=per_sample_loss,
                metric_loss=metric_loss.detach(),
                batch_samples=int(batch.image.size(0)),
                forward_ms=forward_ms,
                backward_ms=backward_ms,
                optimizer_ms=0.0,
            )
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        if step_events is not None:
            self._record_cuda_timing_event(backward_end_event, step_device)
            backward_ms = 0.0
        else:
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

        optimizer_start = time.perf_counter()
        run_context.scaler.step(run_context.optimizer)
        run_context.scaler.update()
        self._step_batch_schedulers(run_context)
        if step_events is not None:
            self._record_cuda_timing_event(optimizer_end_event, step_device)
            optimizer_end_event.synchronize()
            forward_ms = self._elapsed_cuda_event_ms(forward_start_event, forward_end_event)
            backward_ms = self._elapsed_cuda_event_ms(forward_end_event, backward_end_event)
            optimizer_ms = self._elapsed_cuda_event_ms(backward_end_event, optimizer_end_event)
        else:
            optimizer_ms = (time.perf_counter() - optimizer_start) * 1000.0

        return _TrainStepResult(
            outputs=outputs,
            per_sample_loss=per_sample_loss,
            metric_loss=metric_loss.detach(),
            batch_samples=int(batch.image.size(0)),
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            optimizer_ms=optimizer_ms,
        )

    def _publish_train_batch_runtime(
        self,
        *,
        epoch: int,
        batch_index: int,
        run_context: _RunContext,
        batch: _PreparedTrainBatch,
        step_result: _TrainStepResult,
        batch_total_ms: float,
    ) -> None:
        self._publish_batch_preview(
            epoch=epoch,
            batch_index=batch_index,
            train_size=run_context.train_size,
            data=batch.image,
            target=batch.label,
            outputs=step_result.outputs,
            preview_stride=run_context.strides.preview,
        )

        if (batch_index % run_context.strides.metric == 0) or (batch_index == run_context.train_size - 1):
            batch_loss = self._loss_value_to_float(cast(torch.Tensor | float, step_result.metric_loss))
            epoch_points = self._batch_points_by_epoch.setdefault(int(epoch + 1), [])
            epoch_points.append((float(batch_index + 1), batch_loss))
            self._bus.put([
                'metrics',
                {
                    'type': 'train_batch',
                    'epoch': epoch + 1,
                    'batch_index': batch_index + 1,
                    'loss': batch_loss,
                },
            ])
            self._bus.put([
                'metrics',
                {
                    'type': 'train_perf',
                    'epoch': epoch + 1,
                    'batch_index': batch_index + 1,
                    'data_wait_ms': float(batch.data_wait_ms),
                    'augmentation_ms': float(batch.augmentation_ms),
                    'forward_ms': float(step_result.forward_ms),
                    'backward_ms': float(step_result.backward_ms),
                    'optimizer_ms': float(step_result.optimizer_ms),
                    'total_ms': float(batch_total_ms),
                },
            ])

        if (batch_index % run_context.strides.progress == 0) or (batch_index == run_context.train_size - 1):
            self._bus.put([
                'metrics',
                {
                    'type': 'train_batch_progress',
                    'current': int(batch_index + 1),
                    'total': int(run_context.train_size),
                },
            ])

    def _run_train_epoch(
        self,
        epoch: int,
        device: torch.device,
        run_context: _RunContext,
        active_profiler: _ActiveTrainProfiler | None = None,
    ) -> _EpochStats:
        epoch_stats = _EpochStats()

        train_dataset = self._train_dataloader.dataset
        if hasattr(train_dataset, 'set_epoch'):
            cast(_SupportsSetEpoch, train_dataset).set_epoch()
        train_sampler = getattr(self._train_dataloader, 'sampler', None)
        if hasattr(train_sampler, 'resize'):
            base_dataset = getattr(train_dataset, '_base_dataset', train_dataset)
            reset_sampling_state = bool(
                getattr(base_dataset, 'shuffle_frames', False)
                or getattr(base_dataset, '_dynamic_frame_lengths', False)
            )
            cast(_SupportsLossAwareSampling, train_sampler).resize(
                len(train_dataset),
                reset=reset_sampling_state,
            )

        train_iterator = iter(self._train_dataloader)
        batch_index = 0
        while True:
            data_wait_started_at = time.perf_counter()
            try:
                batch = next(train_iterator)
            except StopIteration:
                break
            prepared_batch, skipped_here = self._prepare_train_batch(
                batch=batch,
                device=device,
                data_wait_started_at=data_wait_started_at,
            )
            if skipped_here > 0:
                epoch_stats.skipped_uniform_count += skipped_here

            if prepared_batch is None:
                self._step_training_profiler(active_profiler)
                batch_index += 1
                continue

            step_result = self._run_train_step(run_context=run_context, batch=prepared_batch)
            if step_result is None:
                epoch_stats.skipped_non_finite_count += 1
                self._step_training_profiler(active_profiler)
                batch_index += 1
                continue
            batch_total_ms = (time.perf_counter() - prepared_batch.batch_start) * 1000.0

            epoch_stats.add_batch(
                batch_samples=step_result.batch_samples,
                batch_loss=step_result.metric_loss,
                data_wait_ms=prepared_batch.data_wait_ms,
                augmentation_ms=prepared_batch.augmentation_ms,
                forward_ms=step_result.forward_ms,
                backward_ms=step_result.backward_ms,
                optimizer_ms=step_result.optimizer_ms,
                total_ms=batch_total_ms,
            )
            self._update_loss_aware_sampling(
                run_context=run_context,
                sample_indices=prepared_batch.sample_indices,
                per_sample_loss=step_result.per_sample_loss,
                mixup_pair_indices=prepared_batch.mixup_pair_indices,
                mixup_lambdas=prepared_batch.mixup_lambdas,
            )
            self._publish_train_batch_runtime(
                epoch=epoch,
                batch_index=batch_index,
                run_context=run_context,
                batch=prepared_batch,
                step_result=step_result,
                batch_total_ms=batch_total_ms,
            )
            self._step_training_profiler(active_profiler)
            batch_index += 1

        return epoch_stats

    @staticmethod
    def _reduce_epoch_train_stats(
        epoch_stats: _EpochStats,
        *,
        distributed: bool,
        device: torch.device,
    ) -> tuple[float, float]:
        loss_sum_tensor = epoch_stats.train_loss_sum
        if distributed:
            reduce_tensor = torch.zeros(2, dtype=torch.float64, device=device)
            if loss_sum_tensor is not None:
                reduce_tensor[0] = loss_sum_tensor.detach().to(device=device, dtype=torch.float64)
            reduce_tensor[1] = float(epoch_stats.train_samples_count)
            dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
            return float(reduce_tensor[0].item()), max(1.0, float(reduce_tensor[1].item()))
        local_loss_sum = 0.0
        if loss_sum_tensor is not None:
            local_loss_sum = float(loss_sum_tensor.detach().to(device='cpu', dtype=torch.float64).item())
        return local_loss_sum, float(max(1, epoch_stats.train_samples_count))

    def _publish_epoch_train_metrics(
        self,
        *,
        epoch: int,
        run_context: _RunContext,
        epoch_stats: _EpochStats,
        global_train_loss: float,
        global_train_samples: float,
    ) -> None:
        if global_train_samples <= 0:
            return
        avg_train_loss = global_train_loss / global_train_samples
        self._train_epoch_history.append((float(epoch + 1), float(avg_train_loss)))
        self._bus.put(['metrics', {'type': 'train_epoch', 'epoch': epoch + 1, 'loss': float(avg_train_loss)}])
        avg_denom = max(1, run_context.train_size)
        self._bus.put([
            'metrics',
            {
                'type': 'train_perf_epoch',
                'epoch': epoch + 1,
                'data_wait_ms': float(epoch_stats.data_wait_ms / avg_denom),
                'augmentation_ms': float(epoch_stats.augmentation_ms / avg_denom),
                'forward_ms': float(epoch_stats.forward_ms / avg_denom),
                'backward_ms': float(epoch_stats.backward_ms / avg_denom),
                'optimizer_ms': float(epoch_stats.optimizer_ms / avg_denom),
                'total_ms': float(epoch_stats.total_ms / avg_denom),
            },
        ])
        if self._skip_uniform_labels and epoch_stats.skipped_uniform_count > 0:
            self._bus.put([
                'logging',
                f'Пропущено сэмплов из-за uniform label (0/1): {epoch_stats.skipped_uniform_count}',
            ])
        if epoch_stats.skipped_non_finite_count > 0:
            self._bus.put([
                'logging',
                (
                    'Training warning: '
                    f'skipped {epoch_stats.skipped_non_finite_count} batch(es) with non-finite loss/gradients.'
                ),
            ])

    def _publish_epoch_load_breakdown(
        self,
        *,
        epoch: int,
        epoch_stats: _EpochStats,
        train_stage_ms: float,
        validation_ms: float,
        checkpoint_ms: float,
    ) -> None:
        total_epoch_ms = max(1e-6, float(train_stage_ms + validation_ms + checkpoint_ms))
        parts: list[tuple[str, float]] = [
            ('data_wait', float(epoch_stats.data_wait_ms)),
            ('augmentation', float(epoch_stats.augmentation_ms)),
            ('forward', float(epoch_stats.forward_ms)),
            ('backward', float(epoch_stats.backward_ms)),
            ('optimizer', float(epoch_stats.optimizer_ms)),
            ('validation', float(validation_ms)),
            ('checkpoint', float(checkpoint_ms)),
        ]
        dominant_name, dominant_ms = max(parts, key=lambda x: x[1])
        breakdown = ' | '.join(
            f'{name}: {ms/1000.0:.2f}s ({(ms / total_epoch_ms) * 100.0:.1f}%)'
            for name, ms in parts
        )
        self._bus.put([
            'logging',
            (
                f'Профилирование по времени, эпоха {epoch + 1}: {breakdown}. '
                f'Наибольшая нагрузка: {dominant_name} ({dominant_ms/1000.0:.2f}s).'
            ),
        ])
        self._bus.put([
            'metrics',
            {
                'type': 'epoch_time_breakdown',
                'epoch': int(epoch + 1),
                'train_total_ms': float(train_stage_ms),
                'data_wait_ms': float(epoch_stats.data_wait_ms),
                'augmentation_ms': float(epoch_stats.augmentation_ms),
                'forward_ms': float(epoch_stats.forward_ms),
                'backward_ms': float(epoch_stats.backward_ms),
                'optimizer_ms': float(epoch_stats.optimizer_ms),
                'validation_ms': float(validation_ms),
                'checkpoint_ms': float(checkpoint_ms),
                'dominant_stage': dominant_name,
            },
        ])

    def _run_validation_if_main_process(
        self,
        *,
        epoch: int,
        is_main_process: bool,
        device: torch.device,
        run_context: _RunContext,
    ) -> dict[str, float] | None:
        if not is_main_process:
            return None
        return self._run_validation_epoch(
            epoch,
            device,
            run_context.bce_criterion,
            run_context.autocast_ctx,
        )

    def _publish_validation_metrics(self, *, epoch: int, validation_result: dict[str, float]) -> None:
        avg_val_loss = validation_result['loss']
        val_accuracy = validation_result['accuracy']
        val_iou = validation_result['iou']
        val_dice = validation_result['dice']
        val_f1 = validation_result['f1']
        best_threshold = float(validation_result.get('best_threshold', 0.5))
        best_threshold_iou = float(validation_result.get('best_threshold_iou', val_iou))
        best_threshold_dice = float(validation_result.get('best_threshold_dice', val_dice))
        best_threshold_f1 = float(validation_result.get('best_threshold_f1', val_f1))
        self._val_epoch_history.append((float(epoch + 1), float(avg_val_loss)))
        self._val_iou_history.append((float(epoch + 1), float(val_iou)))
        self._val_dice_history.append((float(epoch + 1), float(val_dice)))

        self._bus.put([
            'logging',
            (
                f'Epoch [{epoch + 1}/{self._epochs}] '
                f'Validation accuracy: {val_accuracy:.4%} | '
                f'IoU: {val_iou:.4%} | Dice: {val_dice:.4%} | F1: {val_f1:.4%} | '
                f'Best threshold: {best_threshold:.2f} '
                f'(IoU {best_threshold_iou:.4%}, Dice {best_threshold_dice:.4%}, F1 {best_threshold_f1:.4%})'
            ),
        ])
        self._bus.put([
            'metrics',
            {
                'type': 'val_epoch',
                'epoch': epoch + 1,
                'loss': float(avg_val_loss),
                'accuracy': float(val_accuracy),
                'iou': float(val_iou),
                'dice': float(val_dice),
                'f1': float(val_f1),
                'best_threshold': float(best_threshold),
                'best_threshold_iou': float(best_threshold_iou),
                'best_threshold_dice': float(best_threshold_dice),
                'best_threshold_f1': float(best_threshold_f1),
            },
        ])

    def _update_early_stopping_from_validation(
        self,
        *,
        epoch: int,
        validation_result: dict[str, float],
        early_stopping_state: _EarlyStoppingState,
        early_stopping_config: _EarlyStoppingConfig,
    ) -> None:
        if not early_stopping_config.enabled:
            return

        avg_val_loss = float(validation_result['loss'])
        improved = (
            early_stopping_state.best_loss is None
            or (early_stopping_state.best_loss - avg_val_loss) > early_stopping_config.min_delta
        )
        if improved:
            early_stopping_state.best_loss = avg_val_loss
            early_stopping_state.bad_epochs = 0
            early_stopping_state.best_epoch = int(epoch + 1)
            early_stopping_state.best_threshold = float(validation_result.get('best_threshold', self._recommended_inference_threshold))
            if self._early_stopping_params.restore_best_weights:
                early_stopping_state.best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self._base_model.state_dict().items()
                }
            self._bus.put([
                'logging',
                f'Early stopping: new best result at epoch {epoch + 1} (val_loss={avg_val_loss:.6f}).',
            ])
            return

        early_stopping_state.bad_epochs += 1
        self._bus.put([
            'logging',
            f'Early stopping: no improvement {early_stopping_state.bad_epochs}/{early_stopping_config.patience}.',
        ])

    def _handle_validation(
        self,
        *,
        epoch: int,
        device: torch.device,
        run_context: _RunContext,
        early_stopping_state: _EarlyStoppingState,
        early_stopping_config: _EarlyStoppingConfig,
        is_main_process: bool,
    ) -> dict[str, float] | None:
        validation_result = self._run_validation_if_main_process(
            epoch=epoch,
            is_main_process=is_main_process,
            device=device,
            run_context=run_context,
        )
        if validation_result is None:
            return None

        self._publish_validation_metrics(epoch=epoch, validation_result=validation_result)
        self._update_early_stopping_from_validation(
            epoch=epoch,
            validation_result=validation_result,
            early_stopping_state=early_stopping_state,
            early_stopping_config=early_stopping_config,
        )
        return validation_result

    def _save_epoch_artifacts(
        self,
        *,
        epoch: int,
        run_context: _RunContext,
        early_stopping_state: _EarlyStoppingState,
        is_main_process: bool,
    ) -> None:
        if not is_main_process:
            return
        self._save_model_artifact()
        self._save_checkpoint(
            epoch + 1,
            run_context.optimizer,
            run_context.scaler,
            run_context.warmup_scheduler,
            run_context.scheduler,
            early_stopping_best_loss=early_stopping_state.best_loss,
            early_stopping_bad_epochs=early_stopping_state.bad_epochs,
            early_stopping_best_epoch=early_stopping_state.best_epoch,
            early_stopping_best_model_state=early_stopping_state.best_model_state,
            early_stopping_best_threshold=early_stopping_state.best_threshold,
        )

    @staticmethod
    def _sync_early_stopping_signal(
        should_stop: bool,
        *,
        is_main_process: bool,
        distributed: bool,
        device: torch.device,
    ) -> bool:
        if not distributed:
            return bool(should_stop)
        stop_tensor = torch.tensor([1 if (should_stop and is_main_process) else 0], device=device, dtype=torch.int32)
        dist.broadcast(stop_tensor, src=0)
        return bool(int(stop_tensor.item()))

    def _initialize_training_runtime(
        self,
        *,
        device: torch.device,
        rank: int,
        distributed: bool,
    ) -> _TrainingRuntimeState:
        is_main_process = (rank == 0)
        run_context = self._create_run_context(device, is_main_process, distributed)
        start_epoch, early_stopping_state, early_stopping_config = self._restore_training_state(run_context)
        active_profiler = self._start_training_profiler(
            device=device,
            train_size=run_context.train_size,
            is_main_process=is_main_process,
            distributed=distributed,
        )
        return _TrainingRuntimeState(
            is_main_process=is_main_process,
            run_context=run_context,
            start_epoch=start_epoch,
            early_stopping_state=early_stopping_state,
            early_stopping_config=early_stopping_config,
            active_profiler=active_profiler,
        )

    def _run_single_training_epoch(
        self,
        *,
        epoch: int,
        device: torch.device,
        distributed: bool,
        runtime_state: _TrainingRuntimeState,
    ) -> None:
        run_context = runtime_state.run_context
        epoch_started_at = time.perf_counter()
        self._publish_epoch_start(epoch, run_context, distributed)

        train_stage_start = time.perf_counter()
        epoch_stats = self._run_train_epoch(
            epoch,
            device,
            run_context,
            active_profiler=runtime_state.active_profiler,
        )
        train_stage_ms = (time.perf_counter() - train_stage_start) * 1000.0

        global_train_loss, global_train_samples = self._reduce_epoch_train_stats(
            epoch_stats,
            distributed=distributed,
            device=device,
        )
        average_train_loss = (
            float(global_train_loss) / float(global_train_samples)
            if int(global_train_samples) > 0
            else None
        )
        if runtime_state.is_main_process:
            self._publish_epoch_train_metrics(
                epoch=epoch,
                run_context=run_context,
                epoch_stats=epoch_stats,
                global_train_loss=global_train_loss,
                global_train_samples=global_train_samples,
            )

        validation_start = time.perf_counter()
        validation_result = self._handle_validation(
            epoch=epoch,
            device=device,
            run_context=run_context,
            early_stopping_state=runtime_state.early_stopping_state,
            early_stopping_config=runtime_state.early_stopping_config,
            is_main_process=runtime_state.is_main_process,
        )
        self._step_epoch_scheduler(
            run_context=run_context,
            validation_result=validation_result,
            train_loss=average_train_loss,
            distributed=distributed,
            device=device,
            is_main_process=runtime_state.is_main_process,
        )
        validation_ms = (time.perf_counter() - validation_start) * 1000.0

        checkpoint_start = time.perf_counter()
        self._save_epoch_artifacts(
            epoch=epoch,
            run_context=run_context,
            early_stopping_state=runtime_state.early_stopping_state,
            is_main_process=runtime_state.is_main_process,
        )
        checkpoint_ms = (time.perf_counter() - checkpoint_start) * 1000.0

        if runtime_state.is_main_process:
            self._publish_epoch_load_breakdown(
                epoch=epoch,
                epoch_stats=epoch_stats,
                train_stage_ms=train_stage_ms,
                validation_ms=validation_ms,
                checkpoint_ms=checkpoint_ms,
            )
            self._bus.put([
                'logging',
                (
                    f'Epoch [{epoch + 1}/{self._epochs}] completed in '
                    f'{self._format_elapsed_duration(time.perf_counter() - epoch_started_at)}.'
                ),
            ])

        return None

    def _should_stop_after_epoch(
        self,
        *,
        epoch: int,
        device: torch.device,
        distributed: bool,
        runtime_state: _TrainingRuntimeState,
    ) -> bool:
        should_stop = bool(
            runtime_state.early_stopping_config.enabled
            and runtime_state.early_stopping_state.bad_epochs > 0
            and runtime_state.early_stopping_state.bad_epochs >= runtime_state.early_stopping_config.patience
        )
        should_stop = self._sync_early_stopping_signal(
            should_stop,
            is_main_process=runtime_state.is_main_process,
            distributed=distributed,
            device=device,
        )
        if not should_stop:
            return False

        if runtime_state.is_main_process:
            self._bus.put([
                'logging',
                (
                    f'Early stopping triggered at epoch {epoch + 1}. '
                    f'Best val_loss reached at epoch {runtime_state.early_stopping_state.best_epoch}.'
                ),
            ])
            if self._early_stopping_params.restore_best_weights and runtime_state.early_stopping_state.best_model_state:
                self._base_model.load_state_dict(runtime_state.early_stopping_state.best_model_state)
                self._recommended_inference_threshold = float(runtime_state.early_stopping_state.best_threshold)
                self._bus.put(['logging', 'Best validation weights restored.'])
        return True

    def _publish_epoch_end_memory(self, *, epoch: int, is_main_process: bool) -> None:
        if not is_main_process:
            return
        memory_payload = _collect_memory_metrics()
        if memory_payload is not None:
            self._bus.put(['metrics', {'type': 'system_memory', **memory_payload, 'epoch': int(epoch + 1)}])

    def _finalize_training_success(self, *, is_main_process: bool) -> None:
        if is_main_process:
            self._save_model_artifact()
            try:
                self._save_metric_charts()
            except Exception as error:
                self._bus.put(['logging', f'Failed to save training metric charts: {error}'])
            try:
                log_path = self._save_path.parent / 'training_log.txt'
                self._bus.put(['logging', f'Training log saved: {log_path}'])
                self._save_training_log()
            except Exception as error:
                self._bus.put(['logging', f'Failed to save training log: {error}'])
        if self.callback is not None and is_main_process:
            self.callback()
        if is_main_process:
            print('Model saved successfully!')

    def _run_impl(self, device: torch.device, rank: int, world_size: int, distributed: bool):
        del world_size
        runtime_state = self._initialize_training_runtime(
            device=device,
            rank=rank,
            distributed=distributed,
        )

        if runtime_state.start_epoch >= self._epochs:
            self._bus.put(['logging', 'All epochs from checkpoint are already completed. Additional training is not required.'])

        try:
            for epoch in range(runtime_state.start_epoch, self._epochs):
                self._run_single_training_epoch(
                    epoch=epoch,
                    device=device,
                    distributed=distributed,
                    runtime_state=runtime_state,
                )
                if self._should_stop_after_epoch(
                    epoch=epoch,
                    device=device,
                    distributed=distributed,
                    runtime_state=runtime_state,
                ):
                    break

                self._publish_epoch_end_memory(epoch=epoch, is_main_process=runtime_state.is_main_process)
                if distributed:
                    dist.barrier()
        finally:
            self._stop_random_artifact_bank()
            self._stop_training_profiler(runtime_state.active_profiler)

        self._finalize_training_success(is_main_process=runtime_state.is_main_process)

@dataclass(frozen=True)
class RecognitionRuntimePlan:
    threads: int
    devices: list[torch.device]
    gpu_count: int
    use_multiprocessing: bool
    cut_workers: int
    predict_workers: int
    sew_workers: int


class NeuralRecognizer(threading.Thread):

    def __init__(self, recognition_parameters:RecognitionParameters,
                 message_bus:AbstractMessageBus, callback:Callable[..., None]|None = None  ):
        super().__init__()
        self.number_of_threads:int = 1
        self.devices_list: list[torch.device] = []
        self._parameters = recognition_parameters
        self._bus = message_bus
        self.callback = callback
        result_folder = self._parameters.result_folder
        result_folder.mkdir(parents=True, exist_ok=True)
        self.colors = 1
        self.model:nn.Module|None = None
        self._thread_stop_event = threading.Event()
        self.stop_event = mp.Event()
        self._resolved_output_threshold: float | None = 0.5
        self._use_context_branch = False
        self._context_crop_size: tuple[int, int] | None = None
        self._context_input_size: tuple[int, int] | None = None

    def _ensure_source_files_indexed(self) -> None:
        if self._parameters.source_files:
            self._bus.publish('logging', f'Images queued for recognition: {len(self._parameters.source_files)}')
            return

        source_folder = getattr(self._parameters, 'source_folder', None)
        if source_folder is None or not str(source_folder).strip():
            self._parameters.source_files = []
            self._bus.publish('logging', 'Images queued for recognition: 0')
            return

        self._bus.publish('logging', 'Индексация файлов для распознавания...')
        self._parameters.source_files = filter_images(Path(source_folder))
        self._bus.publish('logging', f'Images queued for recognition: {len(self._parameters.source_files)}')

    def run(self, multithreading: bool | None = None):
        try:
            self._ensure_source_files_indexed()
            self.prepare_model()
            if multithreading is None:
                multithreading = bool(getattr(self._parameters, 'recognition_multiprocessing_enabled', True))
            runtime_plan = self._build_runtime_plan(multithreading=multithreading)
            self.number_of_threads = runtime_plan.threads
            self.devices_list = runtime_plan.devices

            if runtime_plan.use_multiprocessing:
                self.run_multiprocessing(runtime_plan)
            else:
                if not multithreading:
                    self._bus.publish(
                        'logging',
                        'Recognition multiprocessing disabled by settings; using single-thread mode.',
                    )
                elif multithreading and not isinstance(self._parameters.model, (str, Path)):
                    self._bus.publish(
                        'logging',
                        'Многопроцессное распознавание доступно только при загрузке модели из файла. Переход в однопоточный режим.',
                    )
                elif len(self._parameters.source_files) < 2:
                    self._bus.publish(
                        'logging',
                        'Recognition multiprocessing requires at least 2 source images; falling back to single-thread mode.',
                    )
                self.run_one_thread()
        finally:
            self.model = None
            _release_torch_memory()
            if self.callback is not None:
                self.callback()

    def _build_runtime_plan(self, multithreading: bool) -> RecognitionRuntimePlan:
        source_count = len(self._parameters.source_files)
        cpu_threads = mp.cpu_count()
        devices, gpu_count = self._resolve_devices()

        can_use_multiprocessing_model = isinstance(self._parameters.model, (str, Path))
        enough_inputs_for_pipeline = source_count >= 2
        use_multiprocessing = bool(multithreading and can_use_multiprocessing_model and enough_inputs_for_pipeline)
        if gpu_count > 0:
            predict_workers = int(gpu_count)
            cut_workers = int(RECOGNITION_AUX_WORKERS_PER_GPU * gpu_count)
            sew_workers = int(RECOGNITION_AUX_WORKERS_PER_GPU * gpu_count)
            threads = int(predict_workers + cut_workers + sew_workers)
        else:
            threads = min(source_count, cpu_threads)
            predict_workers = max(1, len(devices))
            remaining_workers = max(2, threads - predict_workers)
            cut_workers = max(1, remaining_workers // 2)
            sew_workers = max(1, remaining_workers - cut_workers)

        return RecognitionRuntimePlan(
            threads=threads,
            devices=devices,
            gpu_count=gpu_count,
            use_multiprocessing=use_multiprocessing,
            cut_workers=cut_workers,
            predict_workers=predict_workers,
            sew_workers=sew_workers,
        )

    @staticmethod
    def _resolve_devices() -> tuple[list[torch.device], int]:
        if not torch.cuda.is_available():
            return [torch.device('cpu')], 0
        gpu_count = System.check_gpu_availability()
        if gpu_count <= 0:
            return [torch.device('cpu')], 0
        return [torch.device(f'cuda:{gpu}') for gpu in range(gpu_count)], gpu_count

    @staticmethod
    def _normalize_output_threshold(value: Any, *, fallback: float = 0.5) -> float:
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            threshold = float(fallback)
        return float(min(max(threshold, 0.0), 1.0))

    @staticmethod
    def _extract_recommended_threshold(model: nn.Module) -> float | None:
        metadata = getattr(model, '_neuralimage_artifact_metadata', None)
        if not isinstance(metadata, dict):
            return None
        inference = metadata.get('inference')
        if not isinstance(inference, dict):
            return None
        threshold = inference.get('recommended_threshold')
        if threshold is None:
            return None
        return NeuralRecognizer._normalize_output_threshold(threshold, fallback=0.5)

    def _resolve_output_threshold(self, model: nn.Module) -> None:
        manual_threshold = self._normalize_output_threshold(
            getattr(self._parameters, 'threshold', 0.5),
            fallback=0.5,
        )
        if not bool(getattr(self._parameters, 'binarize_output', True)):
            self._resolved_output_threshold = None
            self._bus.publish(
                'logging',
                'Recognition output binarization disabled: saving probability maps.',
            )
            return

        if bool(getattr(self._parameters, 'use_auto_threshold', True)):
            recommended_threshold = self._extract_recommended_threshold(model)
            if recommended_threshold is not None:
                self._resolved_output_threshold = recommended_threshold
                self._bus.publish(
                    'logging',
                    f'Recognition threshold: using recommended model threshold {recommended_threshold:.2f}.',
                )
                return
            self._bus.publish(
                'logging',
                (
                    'Recognition threshold: model metadata does not contain a recommended '
                    f'threshold, using fallback {manual_threshold:.2f}.'
                ),
            )
        else:
            self._bus.publish(
                'logging',
                f'Recognition threshold: using manual value {manual_threshold:.2f}.',
            )

        self._resolved_output_threshold = manual_threshold

    def _resolve_context_branch_settings(self, model: nn.Module) -> None:
        model_kwargs = getattr(model, '_neuralimage_model_kwargs', {})
        if not isinstance(model_kwargs, dict):
            model_kwargs = {}

        use_context_override = getattr(self._parameters, 'use_context_branch', None)
        requested_context_branch = (
            bool(model_kwargs.get('use_context_branch', False))
            if use_context_override is None
            else bool(use_context_override)
        )
        if not requested_context_branch:
            self._use_context_branch = False
            self._context_crop_size = None
            self._context_input_size = None
            return

        fallback_local = tuple(getattr(self._parameters, 'part_size', (256, 256)))
        context_crop_raw = getattr(self._parameters, 'context_crop_size', None)
        if context_crop_raw is None:
            context_crop_raw = model_kwargs.get('context_crop_size')
        context_input_raw = getattr(self._parameters, 'context_input_size', None)
        if context_input_raw is None:
            context_input_raw = model_kwargs.get('context_input_size')

        if context_crop_raw is None or context_input_raw is None:
            self._use_context_branch = False
            self._context_crop_size = None
            self._context_input_size = None
            self._bus.publish(
                'logging',
                'Recognition context branch disabled: model/context sizes are missing.',
            )
            return

        self._use_context_branch = True
        self._context_crop_size = normalize_size_pair(
            context_crop_raw,
            fallback=(int(fallback_local[0]) * 2, int(fallback_local[1]) * 2),
        )
        self._context_input_size = normalize_size_pair(
            context_input_raw,
            fallback=(int(fallback_local[0]), int(fallback_local[1])),
        )
        self._bus.publish(
            'logging',
            (
                'Recognition context branch enabled: '
                f'crop={self._context_crop_size}, input={self._context_input_size}.'
            ),
        )

    def prepare_model(self):
        loaded_model: str | Path | nn.Module = self._parameters.model
        if isinstance(loaded_model, (str, Path)):
            loaded_model = load_model_artifact(loaded_model, map_location='cpu')
        if not isinstance(loaded_model, nn.Module):
            raise TypeError('Recognition model must be a torch.nn.Module or a model path.')
        self._resolve_output_threshold(loaded_model)
        if not bool(getattr(sys, 'frozen', False)):
            compile_enabled = str(os.getenv('NEURALIMAGE_TORCH_COMPILE', '1')).strip().lower() in {'1', 'true', 'yes', 'on'}
            compile_fn = getattr(torch, 'compile', None)
            target_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            compile_unavailable_reason = _get_torch_compile_unavailable_reason(target_device_type)
            can_compile = (
                compile_enabled
                and compile_fn is not None
                and compile_unavailable_reason is None
            )
            if can_compile:
                try:
                    compile_mode, mode_reason = _resolve_torch_compile_mode(target_device_type)
                    compiled_model = compile_fn(loaded_model, mode=compile_mode, dynamic=False)
                    for attr_name in (
                        '_neuralimage_model_name',
                        '_neuralimage_input_channels',
                        '_neuralimage_model_kwargs',
                        '_neuralimage_artifact_metadata',
                    ):
                        if hasattr(loaded_model, attr_name):
                            setattr(compiled_model, attr_name, getattr(loaded_model, attr_name))
                    loaded_model = compiled_model
                    self._bus.publish(
                        'logging',
                        f'Распознавание: torch.compile enabled (mode={compile_mode}, reason={mode_reason}).',
                    )
                except Exception as error:
                    self._bus.publish('logging', f'Распознавание: torch.compile отключен (fallback): {error}')
            elif not compile_enabled:
                self._bus.publish('logging', 'Распознавание: torch.compile отключен переменной NEURALIMAGE_TORCH_COMPILE=0.')
            elif compile_fn is None:
                self._bus.publish('logging', 'Распознавание: torch.compile недоступен в этой версии PyTorch.')
            else:
                self._bus.publish(
                    'logging',
                    f'Распознавание: torch.compile отключен, {compile_unavailable_reason}',
                )
        self.model = loaded_model
        self.colors = get_input_channels(loaded_model)
        self._resolve_context_branch_settings(loaded_model)

    def run_multiprocessing(self, runtime_plan: RecognitionRuntimePlan | None = None):
        if runtime_plan is None:
            runtime_plan = self._build_runtime_plan(multithreading=True)
        workload = RecognitionWorkload(
            source_files=list(self._parameters.source_files),
            result_folder=self._parameters.result_folder,
            part_size=self._parameters.part_size,
            overlap=self._parameters.overlap,
            batch_size=self._parameters.batch_size,
            colors=self.colors,
            jpeg_quality=int(getattr(self._parameters, 'jpeg_quality', 95)),
            binarize_output=bool(getattr(self._parameters, 'binarize_output', True)),
            threshold=self._resolved_output_threshold,
            postprocess_enabled=bool(getattr(self._parameters, 'postprocess_enabled', False)),
            postprocess_kernel_size=max(1, int(getattr(self._parameters, 'postprocess_kernel_size', 3))),
            devices=list(self.devices_list),
            model_source=cast(str | Path, self._parameters.model),
            use_context_branch=bool(self._use_context_branch),
            context_crop_size=self._context_crop_size,
            context_input_size=self._context_input_size,
        )
        run_multiprocessing_recognition(
            workload=workload,
            worker_counts=WorkerCounts(
                cut=runtime_plan.cut_workers,
                predict=runtime_plan.predict_workers,
                sew=runtime_plan.sew_workers,
            ),
            stop_event=self.stop_event,
            publish=self._bus.publish,
            stop_token=STOP_TOKEN,
        )

    def run_one_thread(self):
        model = self.model
        if model is None:
            raise RuntimeError('Model is not prepared before recognition start.')
        if not self.devices_list:
            self.devices_list = [torch.device('cpu')]
        device = self.devices_list[0]
        run_single_thread_recognition(
            source_files=list(self._parameters.source_files),
            result_folder=self._parameters.result_folder,
            part_size=self._parameters.part_size,
            overlap=self._parameters.overlap,
            batch_size=self._parameters.batch_size,
            colors=self.colors,
            model=model,
            device=device,
            stop_event=self._thread_stop_event,
            publish=self._bus.publish,
            collect_memory_metrics=_collect_memory_metrics,
            jpeg_quality=int(getattr(self._parameters, 'jpeg_quality', 95)),
            binarize_output=bool(getattr(self._parameters, 'binarize_output', True)),
            threshold=self._resolved_output_threshold,
            postprocess_enabled=bool(getattr(self._parameters, 'postprocess_enabled', False)),
            postprocess_kernel_size=max(1, int(getattr(self._parameters, 'postprocess_kernel_size', 3))),
            use_context_branch=bool(self._use_context_branch),
            context_crop_size=self._context_crop_size,
            context_input_size=self._context_input_size,
        )

    def stop(self):
        self._thread_stop_event.set()
        self.stop_event.set()


class _QueueMessageBus:
    def __init__(self, queue: mp.Queue):
        self._queue = queue

    def publish(self, topic: str, payload: Any) -> None:
        self._queue.put([topic, payload])


class RecognizerProcess(mp.Process):
    def __init__(
        self,
        recognition_parameters: RecognitionParameters,
        message_bus: mp.Queue,
        stop_event: Any,
        multithreading: bool = False,
    ):
        super().__init__()
        self._recognition_parameters = recognition_parameters
        self._bus = message_bus
        self._stop_event = stop_event
        self._multithreading = bool(multithreading)

    def _start_stop_watcher(self, recognizer: NeuralRecognizer) -> threading.Thread:
        def _watch_stop_signal() -> None:
            self._stop_event.wait()
            recognizer.stop()

        watcher = threading.Thread(target=_watch_stop_signal, daemon=True)
        watcher.start()
        return watcher

    def run(self):
        recognizer = NeuralRecognizer(
            self._recognition_parameters,
            message_bus=cast(AbstractMessageBus, _QueueMessageBus(self._bus)),
            callback=None,
        )
        self._start_stop_watcher(recognizer)
        try:
            recognizer.run(multithreading=self._multithreading)
        except Exception as error:
            self._bus.put(['error', f'Recognition error: {error}'])
            raise
        finally:
            recognizer.stop()


class ModelRecognizer(threading.Thread):
    def __init__(
        self,
        recognition_parameters: RecognitionParameters,
        message_bus: AbstractMessageBus,
        callback: Callable[..., None] | None = None,
        multithreading: bool | None = None,
    ):
        super().__init__()
        self._parameters = recognition_parameters
        self._bus = message_bus
        self.callback = callback
        if multithreading is None:
            multithreading = bool(getattr(recognition_parameters, 'recognition_multiprocessing_enabled', True))
        self._multithreading = bool(multithreading)
        self._stop_event = threading.Event()
        self._process_stop_event = mp.Event()
        self.message_queue = mp.Queue()
        self._inline_recognizer: NeuralRecognizer | None = None
        self.succeeded = False
        self.error_message: str | None = None

    @staticmethod
    def _elapsed_suffix(since: float) -> str:
        elapsed_seconds = int(max(0.0, time.perf_counter() - since))
        elapsed_hours, remainder = divmod(elapsed_seconds, 3600)
        elapsed_minutes, elapsed_secs = divmod(remainder, 60)
        return f' Прошло: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}'

    def _publish_recognition_message(
        self,
        message: Any,
        *,
        append_elapsed_suffix: bool,
        started_at: float,
    ) -> None:
        topic, payload = message[0], message[1]
        if append_elapsed_suffix and isinstance(payload, str) and topic != 'error':
            payload = payload + self._elapsed_suffix(started_at)
        if topic == 'error' and isinstance(payload, str):
            self.error_message = payload
        self._bus.publish(topic, payload)

    def _drain_recognition_queue(self, *, append_elapsed_suffix: bool, started_at: float) -> None:
        def _publish(queued_message: Any) -> None:
            self._publish_recognition_message(
                queued_message,
                append_elapsed_suffix=append_elapsed_suffix,
                started_at=started_at,
            )
        _drain_process_queue(self.message_queue, _publish)

    def _finalize_recognition_result(self, recognition_process: Any) -> bool:
        if self._stop_event.is_set():
            self.succeeded = False
            return False
        exit_code = recognition_process.exitcode if recognition_process is not None else 1
        if exit_code not in (0, None):
            if self.error_message is None:
                self.error_message = f'Recognition error: process exited with code {exit_code}.'
                self._bus.publish('error', self.error_message)
            self.succeeded = False
            return False
        self.succeeded = True
        return True

    def run(self):
        recognition_process: Any = None
        try:
            recognition_process = RecognizerProcess(
                self._parameters,
                self.message_queue,
                self._process_stop_event,
                multithreading=self._multithreading,
            )
            recognition_process.start()
            started_at = time.perf_counter()
            while recognition_process.is_alive():
                if self._stop_event.is_set():
                    self._process_stop_event.set()
                    _join_process_with_escalation(recognition_process)
                    break
                self._drain_recognition_queue(
                    append_elapsed_suffix=True,
                    started_at=started_at,
                )
                time.sleep(1)

            _join_process_with_escalation(recognition_process)
            self._drain_recognition_queue(append_elapsed_suffix=False, started_at=started_at)
            self._finalize_recognition_result(recognition_process)
        finally:
            self._inline_recognizer = None
            if recognition_process is not None and recognition_process.is_alive():
                _join_process_with_escalation(recognition_process)
            try:
                self.message_queue.close()
            except Exception:
                pass
            _release_torch_memory()
            if self.callback is not None:
                self.callback()

    def stop(self):
        self._stop_event.set()
        self._process_stop_event.set()
        inline_recognizer = self._inline_recognizer
        if inline_recognizer is not None:
            inline_recognizer.stop()


class NeuralRecognitioner(NeuralRecognizer):
    """Backward-compatible alias for the legacy misspelled class name."""

def cut_image_process(
    cut_queue,
    cutted_queue,
    size,
    overlap,
    stop_event,
    use_context_branch: bool = False,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
):
    return _cut_image_process(
        cut_queue,
        cutted_queue,
        size,
        overlap,
        stop_event,
        use_context_branch=use_context_branch,
        context_crop_size=context_crop_size,
        context_input_size=context_input_size,
        stop_token=STOP_TOKEN,
    )


def cut_image_prepare(
    img_path: Path,
    segment_size: tuple[int, int, int],
    overlap: int,
    *,
    use_context_branch: bool = False,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
):
    return _cut_image_prepare(
        img_path,
        segment_size,
        overlap,
        use_context_branch=use_context_branch,
        context_crop_size=context_crop_size,
        context_input_size=context_input_size,
    )


def get_array_from_image(path, channels):
    return _get_array_from_image(path, channels)


def imgpredict(prediction_queue, predicted_queue, model_path, gpu, batch_size, stop_event):
    return _imgpredict(
        prediction_queue,
        predicted_queue,
        model_path,
        gpu,
        batch_size,
        stop_event,
        stop_token=STOP_TOKEN,
    )


def gpu_predict(img, model, device, batch_size):
    return _gpu_predict(img, model, device, batch_size)


def create_batches(tensor_data, batch_size):
    return _create_batches(tensor_data, batch_size)


def imgsew(
    outputDir,
    sew_queue,
    sewed_queue,
    stop_event,
    jpeg_quality=95,
    threshold=None,
    postprocess_kernel_size=0,
):
    return _imgsew(
        outputDir,
        sew_queue,
        sewed_queue,
        jpeg_quality,
        stop_event,
        threshold=threshold,
        postprocess_kernel_size=postprocess_kernel_size,
        stop_token=STOP_TOKEN,
    )


def sew_from_queue(
    output_dir,
    sew_queue,
    sewed_queue,
    jpeg_quality=95,
    threshold=None,
    postprocess_kernel_size=0,
):
    return _sew_from_queue(
        output_dir,
        sew_queue,
        sewed_queue,
        jpeg_quality=jpeg_quality,
        threshold=threshold,
        postprocess_kernel_size=postprocess_kernel_size,
    )


def sew(save_dir, item, jpeg_quality=95, *, threshold=None, postprocess_kernel_size=0):
    return _sew(
        save_dir,
        item,
        jpeg_quality=jpeg_quality,
        threshold=threshold,
        postprocess_kernel_size=postprocess_kernel_size,
    )
