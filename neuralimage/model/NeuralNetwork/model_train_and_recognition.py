import os
import datetime
import time
import socket
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from collections.abc import Callable, Sized
from contextlib import nullcontext
from typing import Any, ContextManager, Protocol, cast

import multiprocessing as mp
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from lib import System
from model.NeuralNetwork.dataset import CustomDataset
from lib.data_interfaces import (
    RecognitionParameters,
    OptimizerParameters,
    OptimizerName,
    MixedPrecisionMode,
    EarlyStoppingParameters,
    WarmupParameters,
)
from lib.file_func import filter_images
from lib.func import get_input_channels
from lib.message_bus import AbstractMessageBus
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


class _NoOpQueue:
    def put(self, item: Any) -> None:
        return


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


@dataclass(frozen=True)
class _TrainLoopStrides:
    metric: int
    progress: int
    log: int
    preview: int


@dataclass
class _EpochStats:
    train_loss: float = 0.0
    train_samples_count: int = 0
    skipped_uniform_count: int = 0
    data_wait_ms: float = 0.0
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    total_ms: float = 0.0

    def add_batch(
        self,
        *,
        batch_samples: int,
        batch_loss: float,
        data_wait_ms: float,
        forward_ms: float,
        backward_ms: float,
        optimizer_ms: float,
        total_ms: float,
    ) -> None:
        self.train_loss += batch_loss * batch_samples
        self.train_samples_count += int(batch_samples)
        self.data_wait_ms += data_wait_ms
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
    image: torch.Tensor
    label: torch.Tensor
    batch_start: float
    data_wait_ms: float


@dataclass
class _TrainStepResult:
    outputs: torch.Tensor
    per_sample_loss: torch.Tensor
    batch_loss: float
    batch_samples: int
    forward_ms: float
    backward_ms: float
    optimizer_ms: float


@dataclass(frozen=True)
class _TrainingRuntimeState:
    is_main_process: bool
    run_context: _RunContext
    start_epoch: int
    early_stopping_state: _EarlyStoppingState
    early_stopping_config: _EarlyStoppingConfig
    active_profiler: _ActiveTrainProfiler | None


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
                 dice_loss_weight: float = 0.5,
                 iou_loss_weight: float = 0.5,
                 early_stopping_params: EarlyStoppingParameters | None = None,
                 warmup_params: WarmupParameters | None = None,
                 skip_uniform_labels: bool = False,
                 resume_from_checkpoint: bool = False,
                 use_multi_gpu: bool = True,
                 show_batch_preview: bool = True,
                 log_update_frequency: int = 0):
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
        self._dice_loss_weight = float(dice_loss_weight)
        self._iou_loss_weight = float(iou_loss_weight)
        self._early_stopping_params = early_stopping_params or EarlyStoppingParameters()
        self._warmup_params = warmup_params or WarmupParameters()
        self._skip_uniform_labels = bool(skip_uniform_labels)
        self._resume_from_checkpoint = resume_from_checkpoint
        self._use_multi_gpu = use_multi_gpu
        self._show_batch_preview = show_batch_preview
        self._log_update_frequency = max(0, int(log_update_frequency))
        self._stop_event = threading.Event()
        self.message_queue = mp.Queue()
        self.succeeded = False
        self.error_message: str | None = None


    def _validate_training_inputs(self) -> bool:
        if self._train_dataloader is not None and self._model is not None:
            return True
        self.error_message = 'Error: training data or model is missing for process startup.'
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
            dice_loss_weight=self._dice_loss_weight,
            iou_loss_weight=self._iou_loss_weight,
            early_stopping_params=self._early_stopping_params,
            warmup_params=self._warmup_params,
            skip_uniform_labels=self._skip_uniform_labels,
            resume_from_checkpoint=self._resume_from_checkpoint,
            use_multi_gpu=self._use_multi_gpu,
            show_batch_preview=self._show_batch_preview,
            log_update_frequency=self._log_update_frequency,
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
        last_message_at: float,
    ) -> float:
        topic, payload = message[0], message[1]
        now = time.perf_counter()
        if append_elapsed_suffix and isinstance(payload, str) and topic != 'error':
            payload = payload + self._elapsed_suffix(last_message_at)
        if topic == 'error' and isinstance(payload, str):
            self.error_message = payload
        self._bus.publish(topic, payload)
        return now

    def _drain_training_queue(self, *, append_elapsed_suffix: bool, last_message_at: float) -> float:
        while not self.message_queue.empty():
            queued_message = self.message_queue.get()
            last_message_at = self._publish_training_message(
                queued_message,
                append_elapsed_suffix=append_elapsed_suffix,
                last_message_at=last_message_at,
            )
        return last_message_at

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

            training_process = self._create_training_process()
            training_process.start()

            last_message_at = time.perf_counter()
            while training_process.is_alive():
                if self._stop_event.is_set():
                    training_process.kill()
                    break
                last_message_at = self._drain_training_queue(
                    append_elapsed_suffix=True,
                    last_message_at=last_message_at,
                )
                time.sleep(1)

            training_process.join()
            self._drain_training_queue(append_elapsed_suffix=False, last_message_at=last_message_at)
            self._finalize_training_result(training_process)
        finally:
            if training_process is not None and training_process.is_alive():
                training_process.kill()
                training_process.join(timeout=5)
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
                 dice_loss_weight: float = 0.5,
                 iou_loss_weight: float = 0.5,
                 early_stopping_params: EarlyStoppingParameters | None = None,
                 warmup_params: WarmupParameters | None = None,
                 skip_uniform_labels: bool = False,
                 resume_from_checkpoint: bool = False,
                 use_multi_gpu: bool = True,
                 show_batch_preview: bool = True,
                 log_update_frequency: int = 0):
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
        self._dice_loss_weight = float(dice_loss_weight)
        self._iou_loss_weight = float(iou_loss_weight)
        self._early_stopping_params = early_stopping_params or EarlyStoppingParameters()
        self._warmup_params = warmup_params or WarmupParameters()
        self._skip_uniform_labels = bool(skip_uniform_labels)
        self._resume_from_checkpoint = resume_from_checkpoint
        self._use_multi_gpu = use_multi_gpu
        self._show_batch_preview = show_batch_preview
        self._log_update_frequency = max(0, int(log_update_frequency))
        self._training_profiler_config = self._resolve_training_profiler_config()

    @property
    def _base_model(self) -> nn.Module:
        """Return the original model instance even if wrapped by DDP/DataParallel."""
        if isinstance(self._model, (DDP, nn.DataParallel)):
            return self._model.module
        return self._model

    def _resolve_model_artifact_metadata(self) -> tuple[str, int]:
        base_model = self._base_model
        model_name = str(getattr(base_model, '_neuralimage_model_name', base_model.__class__.__name__))
        input_channels = getattr(base_model, '_neuralimage_input_channels', None)
        if input_channels is None:
            input_channels = get_input_channels(base_model)
        return model_name, int(input_channels)

    def _save_model_artifact(self) -> None:
        model_name, input_channels = self._resolve_model_artifact_metadata()
        save_model_artifact(
            self._base_model,
            self._save_path,
            model_name=model_name,
            input_channels=input_channels,
        )

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

    def _should_use_ddp(self) -> bool:
        if not self._use_multi_gpu:
            self._bus.put(['logging', 'Режим multi GPU отключен в настройках.'])
            return False
        if not torch.cuda.is_available():
            return False
        if torch.cuda.device_count() <= 1:
            return False
        if os.name == 'nt':
            # On Windows, DDP+gloo can hang during rendezvous on some setups.
            return False
        return True

    def _should_use_data_parallel(self) -> bool:
        if not self._use_multi_gpu:
            return False
        if not torch.cuda.is_available():
            return False
        if torch.cuda.device_count() <= 1:
            return False
        return os.name == 'nt'

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
        scheduler,
        early_stopping_best_loss: float | None = None,
        early_stopping_bad_epochs: int = 0,
        early_stopping_best_epoch: int = 0,
        early_stopping_best_model_state: dict[str, torch.Tensor] | None = None,
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
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'early_stopping_best_loss': early_stopping_best_loss,
            'early_stopping_bad_epochs': early_stopping_bad_epochs,
            'early_stopping_best_epoch': early_stopping_best_epoch,
            'early_stopping_best_model_state': early_stopping_best_model_state,
        }
        torch.save(checkpoint, self._checkpoint_path())

    def _load_checkpoint_if_available(self, optimizer, scaler, scheduler) -> tuple[int, dict[str, Any]]:
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
            scheduler_state = checkpoint.get('scheduler_state_dict')
            completed_epoch = int(checkpoint.get('completed_epoch', 0))
            if model_state:
                self._base_model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            if scheduler is not None and scheduler_state:
                scheduler.load_state_dict(scheduler_state)

            start_epoch = max(0, min(completed_epoch, self._epochs))
            self._bus.put([
                'logging',
                f'Параметры восстановлены из контрольной точки {checkpoint_path.name}. Последняя завершенная эпоха: {completed_epoch}.',
            ])
            return start_epoch, checkpoint
        except Exception as error:
            self._bus.put(['logging', f'Ошибка загрузки контрольной точки: {error}. Обучение начнется с первой эпохи.'])
            return 0, {}

    def _create_warmup_scheduler(self, optimizer, train_steps_per_epoch: int):
        warmup = self._warmup_params
        if not warmup.enabled:
            return None
        if train_steps_per_epoch <= 0:
            return None

        warmup_epochs = max(1, int(warmup.epochs))
        warmup_steps = warmup_epochs * train_steps_per_epoch
        start_factor = float(min(max(warmup.start_factor, 0.0), 1.0))

        def lr_lambda(step: int) -> float:
            if step >= warmup_steps:
                return 1.0
            progress = (step + 1) / warmup_steps
            return start_factor + (1.0 - start_factor) * progress

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _resolve_target_epochs(self, start_epoch: int) -> int:
        """
        In fine-tuning mode the UI epoch value is treated as "additional epochs".
        For fresh training it is treated as total epochs.
        """
        if self._resume_from_checkpoint and start_epoch > 0:
            return start_epoch + int(self._epochs)
        return int(self._epochs)

    def _resolved_loss_function(self) -> str:
        if self._loss_function in ('bce', 'dice', 'bce_dice', 'iou', 'bce_iou'):
            return self._loss_function
        return 'bce'

    def _resolved_dice_weight(self) -> float:
        return float(min(max(self._dice_loss_weight, 0.0), 1.0))

    def _resolved_iou_weight(self) -> float:
        return float(min(max(self._iou_loss_weight, 0.0), 1.0))

    def _compute_per_sample_loss(
        self,
        outputs: torch.Tensor,
        label: torch.Tensor,
        bce_criterion: nn.Module,
    ) -> torch.Tensor:
        loss_mode = self._resolved_loss_function()
        bce_per_sample = cast(torch.Tensor, bce_criterion(outputs, label)).view(outputs.shape[0], -1).mean(dim=1)
        if loss_mode == 'bce':
            return bce_per_sample

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
            return iou_per_sample

        if loss_mode == 'bce_dice':
            dice_weight = self._resolved_dice_weight()
            return ((1.0 - dice_weight) * bce_per_sample) + (dice_weight * dice_per_sample)

        iou_weight = self._resolved_iou_weight()
        return ((1.0 - iou_weight) * bce_per_sample) + (iou_weight * iou_per_sample)

    def _run_validation_epoch(
        self,
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
        correct = 0
        total = 0
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0
        with torch.no_grad():
            for data, target in self._val_dataloader:
                image = data.to(device, non_blocking=True)
                label = target.to(device, non_blocking=True)
                with autocast_ctx():
                    outputs = self._model(image)
                    per_sample_loss = self._compute_per_sample_loss(outputs, label, bce_criterion)
                    loss = per_sample_loss.mean()

                val_loss += loss.item() * image.size(0)
                probs = torch.sigmoid(outputs)
                preds = probs >= 0.5
                label_bin = label >= 0.5
                correct += (preds == label_bin).sum().item()
                total += label_bin.numel()
                preds_f = preds.float()
                label_f = label_bin.float()
                true_positive += float((preds_f * label_f).sum().item())
                false_positive += float((preds_f * (1.0 - label_f)).sum().item())
                false_negative += float(((1.0 - preds_f) * label_f).sum().item())

        avg_val_loss = val_loss / val_dataset_len
        val_accuracy = (correct / total) if total else 0.0
        iou_denom = true_positive + false_positive + false_negative
        dice_denom = (2.0 * true_positive) + false_positive + false_negative
        if iou_denom == 0.0:
            iou = 1.0
            dice = 1.0
            f1 = 1.0
        else:
            iou = true_positive / iou_denom
            dice = (2.0 * true_positive) / dice_denom if dice_denom > 0.0 else 0.0
            precision_denom = true_positive + false_positive
            recall_denom = true_positive + false_negative
            precision = true_positive / precision_denom if precision_denom > 0.0 else 0.0
            recall = true_positive / recall_denom if recall_denom > 0.0 else 0.0
            f1_denom = precision + recall
            f1 = (2.0 * precision * recall) / f1_denom if f1_denom > 0.0 else 0.0

        return {
            'loss': float(avg_val_loss),
            'accuracy': float(val_accuracy),
            'iou': float(iou),
            'dice': float(dice),
            'f1': float(f1),
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

    def _create_adamw_muon_optimizer(self, params: OptimizerParameters):
        """
        Prepared hook for Muon integration.
        Tries to use optional Muon package; if unavailable falls back to AdamW.
        """
        muon_cls = None
        for module_name in ('muon', 'optimizers.muon', 'torch_optimizer'):
            try:
                module = __import__(module_name, fromlist=['Muon'])
                muon_cls = getattr(module, 'Muon', None)
                if muon_cls is not None:
                    break
            except Exception:
                continue

        if muon_cls is None:
            self._bus.put([
                'logging',
                'Muon optimizer is unavailable. Using AdamW.',
            ])
            return self._create_adamw_optimizer(params)

        return muon_cls(self._model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

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
            if is_main_process:
                self._bus.put(['logging', 'torch.compile отключен в сборке PyInstaller.'])
            return
        if isinstance(self._model, nn.DataParallel):
            if is_main_process:
                self._bus.put(['logging', 'torch.compile skipped: nn.DataParallel model wrapper is active.'])
            return
        if not self._env_bool('NEURALIMAGE_TORCH_COMPILE', True):
            if is_main_process:
                self._bus.put(['logging', 'torch.compile disabled by NEURALIMAGE_TORCH_COMPILE=0.'])
            return
        compile_fn = getattr(torch, 'compile', None)
        if compile_fn is None:
            if is_main_process:
                self._bus.put(['logging', 'torch.compile недоступен в этой версии PyTorch.'])
            return
        target_device_type = device.type if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        if target_device_type == 'cuda' and not _is_module_available('triton'):
            if is_main_process:
                self._bus.put(['logging', 'torch.compile disabled: Triton is not installed for CUDA backend.'])
            return
        try:
            compile_mode, mode_reason = _resolve_torch_compile_mode(target_device_type, device)
            self._model = compile_fn(self._model, mode=compile_mode, dynamic=False)
            if is_main_process:
                self._bus.put(['logging', f'torch.compile enabled (mode={compile_mode}, reason={mode_reason}).'])
        except Exception as error:
            if is_main_process:
                self._bus.put(['logging', f'torch.compile отключен (fallback): {error}'])

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
            if self._should_use_ddp():
                self._run_ddp(world_size=torch.cuda.device_count())
                return

            if self._should_use_data_parallel():
                gpu_count = int(torch.cuda.device_count())
                self._bus.put([
                    'logging',
                    f'Windows fallback: using nn.DataParallel on {gpu_count} GPU instead of DDP.',
                ])
                self._model = nn.DataParallel(self._model)
                device = torch.device('cuda:0')
            else:
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
        if resolved_loss_mode != self._loss_function:
            self._bus.put(['logging', f'Unknown loss "{self._loss_function}". Using "bce".'])
        if resolved_loss_mode == 'bce_dice':
            self._bus.put(['logging', f'Loss function: bce_dice (dice_weight={self._resolved_dice_weight():.2f}).'])
            return
        if resolved_loss_mode == 'bce_iou':
            self._bus.put(['logging', f'Loss function: bce_iou (iou_weight={self._resolved_iou_weight():.2f}).'])
            return
        self._bus.put(['logging', f'Loss function: {resolved_loss_mode}.'])

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
        scheduler = self._create_warmup_scheduler(optimizer, train_size)
        self._log_warmup_configuration(scheduler)

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
        )

    def _restore_training_state(self, run_context: _RunContext) -> tuple[int, _EarlyStoppingState, _EarlyStoppingConfig]:
        start_epoch, checkpoint = self._load_checkpoint_if_available(
            run_context.optimizer,
            run_context.scaler,
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

    def _filter_uniform_batch_samples(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        sample_indices: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, Any, int, bool]:
        if not self._skip_uniform_labels:
            return image, label, sample_indices, 0, True
        label_flat = label.view(label.shape[0], -1)
        eps = 1e-6
        is_all_zero = (label_flat <= eps).all(dim=1)
        is_all_one = (label_flat >= (1.0 - eps)).all(dim=1)
        valid_mask = ~(is_all_zero | is_all_one)
        skipped_here = int((~valid_mask).sum().item())
        if not bool(valid_mask.any()):
            return image, label, sample_indices, skipped_here, False

        image = image[valid_mask]
        label = label[valid_mask]
        if sample_indices is None:
            return image, label, None, skipped_here, True
        if torch.is_tensor(sample_indices):
            sample_indices = sample_indices[valid_mask.detach().to(device=sample_indices.device)]
        else:
            sample_indices_tensor = torch.as_tensor(sample_indices)
            sample_indices = sample_indices_tensor[valid_mask.detach().to(device='cpu')]
        return image, label, sample_indices, skipped_here, True

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
    ) -> None:
        if not run_context.supports_loss_aware_sampling or sample_indices is None:
            return
        if torch.is_tensor(sample_indices):
            sample_idx_tensor = sample_indices.detach().to(device='cpu', dtype=torch.long).flatten()
        else:
            sample_idx_tensor = torch.as_tensor(sample_indices, dtype=torch.long).flatten()
        sample_loss_tensor = per_sample_loss.detach().to(device='cpu', dtype=torch.float32).flatten()
        if sample_idx_tensor.numel() == sample_loss_tensor.numel():
            cast(_SupportsLossAwareSampling, run_context.train_sampler).update_batch_losses(
                sample_idx_tensor,
                sample_loss_tensor,
            )

    def _prepare_train_batch(
        self,
        *,
        batch: Any,
        device: torch.device,
        prev_batch_end: float,
    ) -> tuple[_PreparedTrainBatch | None, int]:
        batch_start = time.perf_counter()
        data_wait_ms = (batch_start - prev_batch_end) * 1000.0
        data, target, sample_indices = self._split_batch(batch)
        image = data.to(device, non_blocking=True)
        label = target.to(device, non_blocking=True)
        image, label, sample_indices, skipped_here, has_valid_samples = self._filter_uniform_batch_samples(
            image,
            label,
            sample_indices,
        )
        if not has_valid_samples:
            return None, skipped_here
        return (
            _PreparedTrainBatch(
                data=data,
                target=target,
                sample_indices=sample_indices,
                image=image,
                label=label,
                batch_start=batch_start,
                data_wait_ms=data_wait_ms,
            ),
            skipped_here,
        )

    def _run_train_step(
        self,
        *,
        run_context: _RunContext,
        batch: _PreparedTrainBatch,
    ) -> _TrainStepResult:
        run_context.optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()
        with run_context.autocast_ctx():
            outputs = self._model(batch.image)
            per_sample_loss = self._compute_per_sample_loss(outputs, batch.label, run_context.bce_criterion)
            loss = per_sample_loss.mean()
        forward_ms = (time.perf_counter() - forward_start) * 1000.0

        backward_start = time.perf_counter()
        run_context.scaler.scale(loss).backward()
        run_context.scaler.unscale_(run_context.optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        backward_ms = (time.perf_counter() - backward_start) * 1000.0

        optimizer_start = time.perf_counter()
        run_context.scaler.step(run_context.optimizer)
        run_context.scaler.update()
        if run_context.scheduler is not None:
            run_context.scheduler.step()
        optimizer_ms = (time.perf_counter() - optimizer_start) * 1000.0

        return _TrainStepResult(
            outputs=outputs,
            per_sample_loss=per_sample_loss,
            batch_loss=float(loss.item()),
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
            data=batch.data,
            target=batch.target,
            outputs=step_result.outputs,
            preview_stride=run_context.strides.preview,
        )

        if (batch_index % run_context.strides.metric == 0) or (batch_index == run_context.train_size - 1):
            self._bus.put([
                'metrics',
                {
                    'type': 'train_batch',
                    'epoch': epoch + 1,
                    'batch_index': batch_index + 1,
                    'loss': step_result.batch_loss,
                },
            ])
            self._bus.put([
                'metrics',
                {
                    'type': 'train_perf',
                    'epoch': epoch + 1,
                    'batch_index': batch_index + 1,
                    'data_wait_ms': float(batch.data_wait_ms),
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

        if (batch_index % run_context.strides.log == 0) or (batch_index == run_context.train_size - 1):
            self._bus.put([
                'training',
                (
                    f'Epoch [{epoch + 1}/{self._epochs}] '
                    f'Loss: {step_result.batch_loss:>7f} '
                    f'Batch: [{batch_index:>5d}/{run_context.train_size:>5d}] '
                    f'| step: {batch_total_ms:.1f} ms'
                ),
            ])

    def _run_train_epoch(
        self,
        epoch: int,
        device: torch.device,
        run_context: _RunContext,
        active_profiler: _ActiveTrainProfiler | None = None,
    ) -> _EpochStats:
        epoch_stats = _EpochStats()
        prev_batch_end = time.perf_counter()

        train_dataset = self._train_dataloader.dataset
        if hasattr(train_dataset, 'set_epoch'):
            cast(_SupportsSetEpoch, train_dataset).set_epoch()

        for batch_index, batch in enumerate(self._train_dataloader):
            prepared_batch, skipped_here = self._prepare_train_batch(
                batch=batch,
                device=device,
                prev_batch_end=prev_batch_end,
            )
            if skipped_here > 0:
                epoch_stats.skipped_uniform_count += skipped_here

            if prepared_batch is None:
                self._step_training_profiler(active_profiler)
                prev_batch_end = time.perf_counter()
                continue

            step_result = self._run_train_step(run_context=run_context, batch=prepared_batch)
            batch_total_ms = (time.perf_counter() - prepared_batch.batch_start) * 1000.0

            epoch_stats.add_batch(
                batch_samples=step_result.batch_samples,
                batch_loss=step_result.batch_loss,
                data_wait_ms=prepared_batch.data_wait_ms,
                forward_ms=step_result.forward_ms,
                backward_ms=step_result.backward_ms,
                optimizer_ms=step_result.optimizer_ms,
                total_ms=batch_total_ms,
            )
            self._update_loss_aware_sampling(
                run_context=run_context,
                sample_indices=prepared_batch.sample_indices,
                per_sample_loss=step_result.per_sample_loss,
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
            prev_batch_end = time.perf_counter()

        return epoch_stats

    @staticmethod
    def _reduce_epoch_train_stats(
        epoch_stats: _EpochStats,
        *,
        distributed: bool,
        device: torch.device,
    ) -> tuple[float, float]:
        if distributed:
            reduce_tensor = torch.tensor(
                [epoch_stats.train_loss, float(epoch_stats.train_samples_count)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
            return float(reduce_tensor[0].item()), max(1.0, float(reduce_tensor[1].item()))
        return float(epoch_stats.train_loss), float(max(1, epoch_stats.train_samples_count))

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
        self._bus.put(['metrics', {'type': 'train_epoch', 'epoch': epoch + 1, 'loss': float(avg_train_loss)}])
        self._bus.put(['logging', f'Средняя потеря на обучающей выборке: {avg_train_loss}'])
        avg_denom = max(1, run_context.train_size)
        self._bus.put([
            'metrics',
            {
                'type': 'train_perf_epoch',
                'epoch': epoch + 1,
                'data_wait_ms': float(epoch_stats.data_wait_ms / avg_denom),
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
        is_main_process: bool,
        device: torch.device,
        run_context: _RunContext,
    ) -> dict[str, float] | None:
        if not is_main_process:
            return None
        return self._run_validation_epoch(
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

        self._bus.put([
            'logging',
            (
                f'Epoch [{epoch + 1}/{self._epochs}] '
                f'Validation loss: {avg_val_loss:.6f} | '
                f'Validation accuracy: {val_accuracy:.4%} | '
                f'IoU: {val_iou:.4%} | Dice: {val_dice:.4%} | F1: {val_f1:.4%}'
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
    ) -> None:
        validation_result = self._run_validation_if_main_process(
            is_main_process=is_main_process,
            device=device,
            run_context=run_context,
        )
        if validation_result is None:
            return

        self._publish_validation_metrics(epoch=epoch, validation_result=validation_result)
        self._update_early_stopping_from_validation(
            epoch=epoch,
            validation_result=validation_result,
            early_stopping_state=early_stopping_state,
            early_stopping_config=early_stopping_config,
        )

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
            run_context.scheduler,
            early_stopping_best_loss=early_stopping_state.best_loss,
            early_stopping_bad_epochs=early_stopping_state.bad_epochs,
            early_stopping_best_epoch=early_stopping_state.best_epoch,
            early_stopping_best_model_state=early_stopping_state.best_model_state,
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
        if runtime_state.is_main_process:
            self._publish_epoch_train_metrics(
                epoch=epoch,
                run_context=run_context,
                epoch_stats=epoch_stats,
                global_train_loss=global_train_loss,
                global_train_samples=global_train_samples,
            )

        validation_start = time.perf_counter()
        self._handle_validation(
            epoch=epoch,
            device=device,
            run_context=run_context,
            early_stopping_state=runtime_state.early_stopping_state,
            early_stopping_config=runtime_state.early_stopping_config,
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
        img_files = self._parameters.source_files
        names = [img_file.name for img_file in img_files]
        result_folder = self._parameters.result_folder
        result_folder.mkdir(parents=True, exist_ok=True)
        self._bus.publish('logging', f'Images queued for recognition: {len(names)}')
        self.colors = 1
        self.model:nn.Module|None = None
        self._thread_stop_event = threading.Event()
        self.stop_event = mp.Event()

    def run(self, multithreading=False):
        try:
            self.prepare_model()
            runtime_plan = self._build_runtime_plan(multithreading=multithreading)
            self.number_of_threads = runtime_plan.threads
            self.devices_list = runtime_plan.devices

            if runtime_plan.use_multiprocessing:
                self.run_multiprocessing(runtime_plan)
            else:
                if multithreading and not isinstance(self._parameters.model, (str, Path)):
                    self._bus.publish(
                        'logging',
                        'Многопроцессное распознавание доступно только при загрузке модели из файла. Переход в однопоточный режим.',
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
        threads = min(source_count, cpu_threads)
        devices, gpu_count = self._resolve_devices()

        can_use_multiprocessing_model = isinstance(self._parameters.model, (str, Path))
        enough_threads_for_pipeline = (threads - gpu_count) >= 3
        use_multiprocessing = bool(multithreading and can_use_multiprocessing_model and enough_threads_for_pipeline)

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

    def prepare_model(self):
        loaded_model: str | Path | nn.Module = self._parameters.model
        if isinstance(loaded_model, (str, Path)):
            loaded_model = load_model_artifact(loaded_model, map_location='cpu')
        if not isinstance(loaded_model, nn.Module):
            raise TypeError('Recognition model must be a torch.nn.Module or a model path.')
        if not bool(getattr(sys, 'frozen', False)):
            compile_enabled = str(os.getenv('NEURALIMAGE_TORCH_COMPILE', '1')).strip().lower() in {'1', 'true', 'yes', 'on'}
            compile_fn = getattr(torch, 'compile', None)
            target_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            can_compile = (
                compile_enabled
                and compile_fn is not None
                and (target_device_type != 'cuda' or _is_module_available('triton'))
            )
            if can_compile:
                try:
                    compile_mode, mode_reason = _resolve_torch_compile_mode(target_device_type)
                    loaded_model = compile_fn(loaded_model, mode=compile_mode, dynamic=False)
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
                self._bus.publish('logging', 'Распознавание: torch.compile отключен, Triton не установлен для CUDA backend.')
        self.model = loaded_model
        self.colors = get_input_channels(loaded_model)

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
            devices=list(self.devices_list),
            model_source=cast(str | Path, self._parameters.model),
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
        multithreading: bool = False,
    ):
        super().__init__()
        self._parameters = recognition_parameters
        self._bus = message_bus
        self.callback = callback
        self._multithreading = bool(multithreading)
        self._stop_event = threading.Event()
        self._process_stop_event = mp.Event()
        self.message_queue = mp.Queue()
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
        last_message_at: float,
    ) -> float:
        topic, payload = message[0], message[1]
        now = time.perf_counter()
        if append_elapsed_suffix and isinstance(payload, str) and topic != 'error':
            payload = payload + self._elapsed_suffix(last_message_at)
        if topic == 'error' and isinstance(payload, str):
            self.error_message = payload
        self._bus.publish(topic, payload)
        return now

    def _drain_recognition_queue(self, *, append_elapsed_suffix: bool, last_message_at: float) -> float:
        while not self.message_queue.empty():
            queued_message = self.message_queue.get()
            last_message_at = self._publish_recognition_message(
                queued_message,
                append_elapsed_suffix=append_elapsed_suffix,
                last_message_at=last_message_at,
            )
        return last_message_at

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
            last_message_at = time.perf_counter()
            while recognition_process.is_alive():
                if self._stop_event.is_set():
                    self._process_stop_event.set()
                    recognition_process.join(timeout=5)
                    if recognition_process.is_alive():
                        recognition_process.kill()
                    break
                last_message_at = self._drain_recognition_queue(
                    append_elapsed_suffix=True,
                    last_message_at=last_message_at,
                )
                time.sleep(1)

            recognition_process.join()
            self._drain_recognition_queue(append_elapsed_suffix=False, last_message_at=last_message_at)
            self._finalize_recognition_result(recognition_process)
        finally:
            if recognition_process is not None and recognition_process.is_alive():
                recognition_process.kill()
                recognition_process.join(timeout=5)
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


class NeuralRecognitioner(NeuralRecognizer):
    """Backward-compatible alias for the legacy misspelled class name."""

def cut_image_process(cut_queue, cutted_queue, size, overlap, stop_event):
    return _cut_image_process(
        cut_queue,
        cutted_queue,
        size,
        overlap,
        stop_event,
        stop_token=STOP_TOKEN,
    )


def cut_image_prepare(img_path: Path, segment_size: tuple[int, int, int], overlap: int):
    return _cut_image_prepare(img_path, segment_size, overlap)


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


def imgsew(outputDir, sew_queue, sewed_queue, stop_event):
    return _imgsew(
        outputDir,
        sew_queue,
        sewed_queue,
        stop_event,
        stop_token=STOP_TOKEN,
    )


def sew_from_queue(output_dir, sew_queue, sewed_queue):
    return _sew_from_queue(output_dir, sew_queue, sewed_queue)


def sew(save_dir, item):
    return _sew(save_dir, item)
