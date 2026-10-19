"""model_train_and_recognition: construction 2."""
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
from collections.abc import Callable
from typing import Any
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neuralimage.lib.data_interfaces import (
    CutoutParameters,
    OptimizerParameters,
    MixupParameters,
    MixedPrecisionMode,
    EarlyStoppingParameters,
    RandomArtifactsParameters,
    SchedulerParameters,
    WarmupParameters,
    HardMiningParameters,
    normalize_multi_gpu_mode,
)
from neuralimage.lib.loss_config import sanitize_loss_term_weights
from neuralimage.losses.composite import (
    HomoscedasticLossWeighter,
)
from neuralimage.model.NeuralNetwork.early_stopping import (
    CheckpointManager,
    EarlyStoppingPolicy,
    EarlyStoppingRuntimeConfig,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import _RecordingQueue


class Construction2Mixin:
    def __init__(self,train_dataloader:DataLoader, val_dataloader:DataLoader | None,
                 model:nn.Module, save_path:Path, epochs:int, message_bus:mp.Queue,
                 callback:Callable[...,None]|None = None,
                 optimizer_params: OptimizerParameters | None = None,
                 mixed_precision: MixedPrecisionMode = MixedPrecisionMode.off,
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
                 extend_epochs_on_resume: bool = True,
                 use_multi_gpu: bool = False,
                 multi_gpu_mode: str | None = None,
                 show_batch_preview: bool = True,
                 log_update_frequency: int = 0,
                 save_validation_binary_images: bool = False,
                 pause_event: Any | None = None,
                 control_dataloader: DataLoader | None = None,
                 loss_weighting_strategy: str = 'static',
                 mask_loss_weight_floor: float = 0.25,
                 topograph_enabled: bool = False,
                 topograph_loss_weight: float = 0.1,
                 topograph_debug_viz: bool = False,
                 topograph_num_processes: int = 1,
                 topograph_use_c: bool = False):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._control_dataloader = control_dataloader
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
        self._extend_epochs_on_resume = bool(extend_epochs_on_resume)
        self._multi_gpu_mode = normalize_multi_gpu_mode(
            multi_gpu_mode,
            use_multi_gpu_fallback=bool(use_multi_gpu),
        )
        self._use_multi_gpu = self._multi_gpu_mode != 'off'
        self._show_batch_preview = show_batch_preview
        self._log_update_frequency = max(0, int(log_update_frequency))
        self._save_validation_binary_images = bool(save_validation_binary_images)
        self._pause_event = pause_event if pause_event is not None else mp.Event()
        self._resume_batch = 0
        self._epoch_start_torch_rng_state: torch.Tensor | None = None
        self._resume_epoch_torch_rng_state: torch.Tensor | None = None
        self._resume_runtime_torch_rng_state: torch.Tensor | None = None
        self._training_profiler_config = self._resolve_training_profiler_config()
        self._random_patch_profile_rows: list[dict[str, int | float]] = []
        self._uncompiled_model: nn.Module | None = None
        self._torch_compile_active = False
        self._recommended_inference_threshold = 0.5
        self._loss_weighting_strategy = str(loss_weighting_strategy).strip().lower()
        self._mask_loss_weight_floor = float(mask_loss_weight_floor)
        self._topograph_enabled = bool(topograph_enabled)
        self._topograph_loss_weight = max(0.0, float(topograph_loss_weight))
        self._topograph_debug_viz = bool(topograph_debug_viz)
        self._topograph_num_processes = max(1, int(topograph_num_processes))
        self._topograph_use_c = bool(topograph_use_c)
        self._topograph_loss_module = None
        self._topograph_binary_skip_logged = False
        self._last_batch_loss_components: dict[str, float] = {}
        self._loss_weighter: HomoscedasticLossWeighter | None = None
        self._train_epoch_history: list[tuple[float, float]] = []
        self._val_epoch_history: list[tuple[float, float]] = []
        self._val_iou_history: list[tuple[float, float]] = []
        self._val_dice_history: list[tuple[float, float]] = []
        self._batch_points_by_epoch: dict[int, list[tuple[float, float]]] = {}
        self._training_log_lines: list[str] = []
        self._bus = _RecordingQueue(message_bus, self._training_log_lines)
        self._random_artifact_bank: _RandomArtifactBank | None = None
        self._non_finite_gradient_skip_count = 0
        self._global_batch = 0
        self._automatic_early_stopping_policy: EarlyStoppingPolicy | None = None
        self._automatic_early_stopping_config: EarlyStoppingRuntimeConfig | None = None
        self._latest_validation_result: dict[str, float] | None = None
        self._latest_validation_global_batch: int | None = None
        self._checkpoint_manager = CheckpointManager(
            self._checkpoint_path(),
            logger=self._bus.log,
        )

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.random_artifact_bank import _RandomArtifactBank
