"""model_train_and_recognition: epoch stats."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any, ContextManager
import torch
import torch.nn as nn

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
    early_stop_requested: bool = False
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
    enable_cuda_event_timing: bool = True

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
    label: torch.Tensor | Mapping[str, torch.Tensor]
    batch_start: float
    data_wait_ms: float
    augmentation_ms: float = 0.0

@dataclass
class _TrainStepResult:
    outputs: Any
    per_sample_loss: torch.Tensor
    metric_loss: torch.Tensor | float | None = None
    batch_loss: torch.Tensor | float | None = None
    loss_components: dict[str, float] | None = None
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

@dataclass
class _TrainingRuntimeState:
    is_main_process: bool
    run_context: _RunContext
    start_epoch: int
    early_stopping_state: _EarlyStoppingState
    early_stopping_config: _EarlyStoppingConfig
    active_profiler: _ActiveTrainProfiler | None
    automatic_stop_requested: bool = False
