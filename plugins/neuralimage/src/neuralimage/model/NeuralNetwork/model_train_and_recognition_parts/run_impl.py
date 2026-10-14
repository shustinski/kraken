"""model_train_and_recognition: run impl."""
from __future__ import annotations
import torch
import torch.distributed as dist

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import _TrainingPauseRequested


class RunImplMixin:
    def _run_impl(self, device: torch.device, rank: int, world_size: int, distributed: bool):
        del world_size
        runtime_state = self._initialize_training_runtime(
            device=device,
            rank=rank,
            distributed=distributed,
        )

        if runtime_state.start_epoch >= self._epochs:
            self._bus.put(['logging', 'All epochs from checkpoint are already completed. Additional training is not required.'])

        paused = False
        try:
            for epoch in range(runtime_state.start_epoch, self._epochs):
                try:
                    self._run_single_training_epoch(
                        epoch=epoch,
                        device=device,
                        distributed=distributed,
                        runtime_state=runtime_state,
                    )
                except _TrainingPauseRequested as pause:
                    paused = True
                    if distributed:
                        dist.barrier()
                    if runtime_state.is_main_process:
                        self._save_pause_checkpoint(
                            epoch=pause.epoch,
                            next_batch=pause.next_batch,
                            run_context=runtime_state.run_context,
                            early_stopping_state=runtime_state.early_stopping_state,
                        )
                        self._bus.put(['logging', 'Training paused after a completed batch; checkpoint saved.'])
                    break
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

        if not paused:
            self._finalize_training_success(is_main_process=runtime_state.is_main_process)
