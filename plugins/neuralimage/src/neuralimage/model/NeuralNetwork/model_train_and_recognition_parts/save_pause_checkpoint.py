"""model_train_and_recognition: save pause checkpoint."""
from __future__ import annotations


class SavePauseCheckpointMixin:
    def _save_pause_checkpoint(
        self,
        *,
        epoch: int,
        next_batch: int,
        run_context: _RunContext,
        early_stopping_state: _EarlyStoppingState,
    ) -> None:
        self._save_model_artifact()
        self._save_checkpoint(
            epoch,
            run_context.optimizer,
            run_context.scaler,
            run_context.warmup_scheduler,
            run_context.scheduler,
            early_stopping_best_loss=early_stopping_state.best_loss,
            early_stopping_bad_epochs=early_stopping_state.bad_epochs,
            early_stopping_best_epoch=early_stopping_state.best_epoch,
            early_stopping_best_model_state=early_stopping_state.best_model_state,
            early_stopping_best_threshold=early_stopping_state.best_threshold,
            current_epoch=epoch,
            next_batch=next_batch,
        )
        checkpoint_path = self._checkpoint_path()
        self._bus.put(['metrics', {
            'type': 'training_paused',
            'checkpoint_path': str(checkpoint_path),
            'epoch': int(epoch),
            'next_batch': int(next_batch),
        }])
