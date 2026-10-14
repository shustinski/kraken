"""model_train_and_recognition: construction."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Callable
import multiprocessing as mp
import threading
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
from neuralimage.lib.message_bus import AbstractMessageBus


class ConstructionMixin:
    def __init__(self,train_dataloader:DataLoader, val_dataloader:DataLoader | None,
                 model:nn.Module, save_path:Path,epochs:int, message_bus:AbstractMessageBus,
                 callback:Callable[..., None]|None = None,
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
        self._stop_event = threading.Event()
        self._pause_event = mp.Event()
        self.message_queue = mp.Queue()
        self.succeeded = False
        self.error_message: str | None = None
        self._recommended_inference_threshold = 0.5
        self._loss_weighting_strategy = str(loss_weighting_strategy).strip().lower()
        self._mask_loss_weight_floor = float(mask_loss_weight_floor)
        self._topograph_enabled = bool(topograph_enabled)
        self._topograph_loss_weight = max(0.0, float(topograph_loss_weight))
        self._topograph_debug_viz = bool(topograph_debug_viz)
        self._topograph_num_processes = max(1, int(topograph_num_processes))
        self._topograph_use_c = bool(topograph_use_c)
