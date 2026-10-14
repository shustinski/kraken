"""general_neural_handler: start training."""
from __future__ import annotations
from pathlib import Path
from neuralimage.lib.data_interfaces import (
    WorkMode,
    normalize_multi_gpu_mode,
)
from neuralimage.model.NeuralNetwork.model_train_and_recognition import ModelTrainer


class StartTrainingMixin:
    def _start_training(self, model, model_save_path: Path):
        self.message_bus.publish('metrics', {'type': 'workflow_phase', 'phase': 'training'})
        self._write_training_manifest(model_save_path)
        resume_from_checkpoint = bool(
            self.work_mode in (WorkMode.further_training, WorkMode.continue_training)
            or getattr(self.tranining_parameters, 'resume_from_checkpoint', False)
        )
        multi_gpu_mode = normalize_multi_gpu_mode(
            getattr(self.tranining_parameters, 'multi_gpu_mode', ''),
            use_multi_gpu_fallback=bool(getattr(self.tranining_parameters, 'use_multi_gpu', False)),
        )
        self.current_thread = ModelTrainer(
            self.train_loader,
            self.val_loader,
            model,
            model_save_path,
            self.tranining_parameters.epochs,
            message_bus=self.message_bus,
            callback=self._stop_training_callback,
            optimizer_params=self.tranining_parameters.optimizer,
            mixed_precision=self.tranining_parameters.mixed_precision,
            loss_function=self.tranining_parameters.loss_function,
            loss_term_weights=getattr(self.tranining_parameters, 'loss_term_weights', {}),
            dice_loss_weight=self.tranining_parameters.dice_loss_weight,
            iou_loss_weight=self.tranining_parameters.iou_loss_weight,
            hard_mining_params=self.tranining_parameters.hard_mining,
            cutout_params=getattr(self.tranining_parameters, 'cutout', None),
            random_artifacts_params=getattr(self.tranining_parameters, 'random_artifacts', None),
            mixup_params=getattr(self.tranining_parameters, 'mixup', None),
            early_stopping_params=self.tranining_parameters.early_stopping,
            warmup_params=self.tranining_parameters.warmup,
            scheduler_params=getattr(self.tranining_parameters, 'scheduler', None),
            skip_uniform_labels=self.tranining_parameters.skip_uniform_labels,
            resume_from_checkpoint=resume_from_checkpoint,
            extend_epochs_on_resume=self.work_mode in (WorkMode.further_training, WorkMode.continue_training),
            use_multi_gpu=multi_gpu_mode != 'off',
            multi_gpu_mode=multi_gpu_mode,
            show_batch_preview=self.tranining_parameters.show_batch_preview,
            log_update_frequency=self.tranining_parameters.log_update_frequency,
            save_validation_binary_images=bool(
                getattr(self.tranining_parameters, 'save_validation_binary_images', False)
            ),
            control_dataloader=getattr(self, 'control_loader', None),
            loss_weighting_strategy=str(
                getattr(self.tranining_parameters, 'loss_weighting_strategy', 'static')
            ),
            mask_loss_weight_floor=float(
                getattr(self.tranining_parameters, 'mask_loss_weight_floor', 0.25)
            ),
            topograph_enabled=bool(getattr(self.tranining_parameters, 'topograph_enabled', False)),
            topograph_loss_weight=float(getattr(self.tranining_parameters, 'topograph_loss_weight', 0.1)),
            topograph_debug_viz=bool(getattr(self.tranining_parameters, 'topograph_debug_viz', False)),
            topograph_num_processes=max(
                1,
                int(getattr(self.tranining_parameters, 'topograph_num_processes', 1)),
            ),
            topograph_use_c=bool(getattr(self.tranining_parameters, 'topograph_use_c', False)),
        )
        self.current_thread.daemon = False
        self.current_thread.start()
        self._wait_for_current_thread('training')
        if not getattr(self.current_thread, 'succeeded', False) and not getattr(self, '_need_stop', False):
            self._training_failed = True
            if getattr(self.current_thread, 'error_message', None) is None:
                self.message_bus.publish('error', 'Обучение завершилось с ошибкой.')
        elif not getattr(self, '_need_stop', False):
            self.message_bus.publish('metrics', {
                'type': 'training_completed',
                'model_path': str(model_save_path),
                'checkpoint_path': str(model_save_path.with_suffix('.ckpt')),
            })
        self.current_thread = None
        # The process/thread lifecycle is over; drop heavy references eagerly.
        model = None
        self._release_torch_memory()
