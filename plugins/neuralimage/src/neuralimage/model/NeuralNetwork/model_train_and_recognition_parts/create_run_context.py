"""model_train_and_recognition: create run context."""
from __future__ import annotations
import torch
import torch.nn as nn

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _RunContext


class CreateRunContextMixin:
    def _create_run_context(self, device: torch.device, is_main_process: bool, distributed: bool) -> _RunContext:
        self._prepare_training_device(device, is_main_process=is_main_process, distributed=distributed)

        bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self._prepare_loss_weighter(device)
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
        enable_cuda_event_timing = bool(device.type == 'cuda' and is_main_process and not distributed)
        if distributed and is_main_process and device.type == 'cuda':
            self._bus.put([
                'logging',
                'Detailed CUDA batch timing is disabled in DDP to avoid per-step device synchronization overhead.',
            ])

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
            enable_cuda_event_timing=enable_cuda_event_timing,
        )
