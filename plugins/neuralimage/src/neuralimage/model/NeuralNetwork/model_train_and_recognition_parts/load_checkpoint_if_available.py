"""model_train_and_recognition: load checkpoint if available."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
import torch
from neuralimage.lib.file_retry import retry_file_read
from neuralimage.preprocessing.config import build_preprocessing_config


class LoadCheckpointIfAvailableMixin:
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
            checkpoint = retry_file_read(
                lambda: torch.load(checkpoint_path, map_location='cpu', weights_only=False),
                path=checkpoint_path,
            )
            checkpoint_preprocessing = checkpoint.get('preprocessing')
            current_preprocessing = self._resolve_dataset_attribute(
                getattr(self, '_train_dataloader', None),
                '_preprocessing',
            )
            if (
                isinstance(checkpoint_preprocessing, dict)
                and isinstance(checkpoint_preprocessing.get('config'), dict)
                and current_preprocessing is not None
            ):
                saved_config = build_preprocessing_config(checkpoint_preprocessing['config'])
                saved_hash = str(checkpoint_preprocessing.get('hash') or saved_config.stable_hash())
                if saved_hash != current_preprocessing.stable_hash():
                    self._bus.put([
                        'logging',
                        'Checkpoint preprocessing does not match the current training dataset. '
                        'The checkpoint will not be resumed.',
                    ])
                    return 0, {}
            model_state = checkpoint.get('model_state_dict')
            optimizer_state = checkpoint.get('optimizer_state_dict')
            loss_weighter_state = checkpoint.get('loss_weighter_state_dict')
            scaler_state = checkpoint.get('scaler_state_dict')
            legacy_scheduler_state = checkpoint.get('scheduler_state_dict')
            warmup_scheduler_state = checkpoint.get('warmup_scheduler_state_dict', legacy_scheduler_state)
            lr_scheduler_state = checkpoint.get('lr_scheduler_state_dict')
            completed_epoch = int(checkpoint.get('completed_epoch', 0))
            if model_state:
                self._base_model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            if loss_weighter_state and self._loss_weighter is not None:
                self._loss_weighter.load_state_dict(loss_weighter_state)
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
            best_checkpoint_loss = checkpoint.get('best_checkpoint_loss')
            if best_checkpoint_loss is not None:
                self._checkpoint_manager.best_loss = float(best_checkpoint_loss)
            best_checkpoint_batch = checkpoint.get('best_checkpoint_global_batch')
            if best_checkpoint_batch is not None:
                self._checkpoint_manager.best_global_batch = int(best_checkpoint_batch)

            self._resume_batch = max(0, int(checkpoint.get('next_batch', 0) or 0))
            if self._resume_batch > 0:
                completed_epoch = max(0, int(checkpoint.get('current_epoch', completed_epoch)))
            python_state = checkpoint.get('python_random_state')
            numpy_state = checkpoint.get('numpy_random_state')
            torch_state = checkpoint.get('torch_rng_state')
            cuda_state = checkpoint.get('cuda_rng_state_all')
            sampler_state = checkpoint.get('sampler_state_dict')
            self._resume_epoch_torch_rng_state = checkpoint.get('epoch_start_torch_rng_state')
            if python_state is not None:
                random.setstate(python_state)
            if numpy_state is not None:
                np.random.set_state(numpy_state)
            if torch_state is not None:
                torch.set_rng_state(torch_state)
                self._resume_runtime_torch_rng_state = torch_state
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)
            sampler = getattr(self._train_dataloader, 'sampler', None)
            wrapped_sampler = getattr(getattr(self._train_dataloader, 'batch_sampler', None), 'sampler', None)
            if callable(getattr(wrapped_sampler, 'load_state_dict', None)):
                sampler = wrapped_sampler
            load_sampler_state = getattr(sampler, 'load_state_dict', None)
            if sampler_state is not None and callable(load_sampler_state):
                load_sampler_state(sampler_state)

            start_epoch = max(0, min(completed_epoch, self._epochs))
            self._bus.put([
                'logging',
                f'Параметры восстановлены из контрольной точки {checkpoint_path.name}. Последняя завершенная эпоха: {completed_epoch}.',
            ])
            return start_epoch, checkpoint
        except Exception as error:
            self._bus.put(['logging', f'Ошибка загрузки контрольной точки: {error}. Обучение начнется с первой эпохи.'])
            return 0, {}
