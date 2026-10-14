"""model_train_and_recognition: save checkpoint."""
from __future__ import annotations
import random
from dataclasses import asdict
import numpy as np
import torch
from neuralimage.model.NeuralNetwork.early_stopping import (
    CheckpointManager,
)


class SaveCheckpointMixin:
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
        current_epoch: int | None = None,
        next_batch: int = 0,
    ) -> None:
        checkpoint_manager = getattr(self, '_checkpoint_manager', None)
        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(self._checkpoint_path())
            self._checkpoint_manager = checkpoint_manager
        automatic_policy = getattr(self, '_automatic_early_stopping_policy', None)
        train_dataloader = getattr(self, '_train_dataloader', None)
        sampler = getattr(train_dataloader, 'sampler', None)
        wrapped_sampler = getattr(getattr(train_dataloader, 'batch_sampler', None), 'sampler', None)
        if callable(getattr(wrapped_sampler, 'state_dict', None)):
            sampler = wrapped_sampler
        sampler_state_fn = getattr(sampler, 'state_dict', None)
        sampler_state = sampler_state_fn() if callable(sampler_state_fn) else None
        preprocessing = self._resolve_dataset_attribute(train_dataloader, '_preprocessing')
        preprocessing_payload = (
            {
                'config': asdict(preprocessing),
                'hash': preprocessing.stable_hash(),
            }
            if preprocessing is not None
            else None
        )
        checkpoint = {
            'version': 2,
            'completed_epoch': completed_epoch,
            'epochs': self._epochs,
            'model_state_dict': self._base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_weighter_state_dict': (
                getattr(self, '_loss_weighter', None).state_dict()
                if getattr(self, '_loss_weighter', None) is not None
                else None
            ),
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
            'current_epoch': int(completed_epoch if current_epoch is None else current_epoch),
            'next_batch': max(0, int(next_batch)),
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'sampler_state_dict': sampler_state,
            'preprocessing': preprocessing_payload,
            'epoch_start_torch_rng_state': getattr(self, '_epoch_start_torch_rng_state', None),
            'global_batch': int(getattr(self, '_global_batch', 0)),
            'automatic_early_stopping_state': (
                automatic_policy.state_dict()
                if automatic_policy is not None
                else None
            ),
            'best_checkpoint_loss': checkpoint_manager.best_loss,
            'best_checkpoint_global_batch': checkpoint_manager.best_global_batch,
        }
        checkpoint_manager.save_regular(checkpoint)
