"""general_neural_handler: construction."""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Callable
import torch
from neuralimage.lib.data_interfaces import (
    WorkMode,
    RecognitionParameters,
    TrainingParameters,
    SampleCutMode,
    normalize_validation_source,
)
from neuralimage.lib.file_func import filter_files
from neuralimage.infrastructure.dataset_source import DatasetStreamSource
from neuralimage.lib.message_bus import AbstractMessageBus


class ConstructionMixin:
    def __init__(
        self,
        work_mode: WorkMode,
        question_module: Callable[..., bool],
        message_bus: AbstractMessageBus,
        recogniton_parameters: RecognitionParameters | None = None,
        tranining_parameters: TrainingParameters | None = None,
        callback: Callable[..., None] | None = None,
        dataset_source: DatasetStreamSource | None = None,
    ):
        self.work_mode = work_mode
        self.dataset_source = dataset_source
        self.callback = callback
        self.recognition_parameters = recogniton_parameters
        self.tranining_parameters = tranining_parameters
        self.question = question_module
        self.message_bus = message_bus
        self.message_bus.publish('logging', 'Инициализация основных функций')

        self.current_thread: threading.Thread | None = None
        self._need_stop = False
        self._need_pause = False
        self._training_failed = False
        self._hard_mining_active = False
        self.train_loader = None
        self.val_loader = None
        self.control_loader = None
        self._train_control_dataset = None

    @staticmethod
    def _release_torch_memory() -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            # Memory release is best-effort and must not break the workflow.
            pass

    def _drop_runtime_references(self) -> None:
        self.current_thread = None
        self.train_loader = None
        self.val_loader = None
        self._release_torch_memory()

    def start(self):
        self._training_failed = False
        if self.work_mode == WorkMode.recognition_only:
            self._start_recognition()
            return

        self._prepare_training_pipeline()
        if self._need_stop:
            return

        model, model_save_path = self._resolve_training_model()
        if self._need_stop:
            return
        self._start_training(model, model_save_path)

        if self._need_stop or self._training_failed or self.work_mode in (WorkMode.train_only, WorkMode.continue_training):
            return

        if isinstance(self.recognition_parameters.model, str):
            self.recognition_parameters.model = model_save_path
        self._start_recognition()

    def _prepare_training_pipeline(self):
        if self.tranining_parameters.cut_mode != SampleCutMode.online:
            raise ValueError(
                'Disk-cut datasets are no longer supported. Use online patch generation.'
            )
        dataset_image, dataset_label = self._prepare_dataset_folders()
        validation_image = None
        validation_label = None
        if self._uses_external_validation():
            validation_image, validation_label = self._prepare_external_validation_folders()
            if self._need_stop:
                return
        train_dataset, val_dataset = self._create_dataset(
            dataset_image,
            dataset_label,
            validation_image_folder=validation_image,
            validation_label_folder=validation_label,
        )
        self._create_dataloader(train_dataset, val_dataset)

    def _uses_external_validation(self) -> bool:
        if not bool(getattr(self.tranining_parameters, 'validation', False)):
            return False
        validation_source = normalize_validation_source(
            getattr(self.tranining_parameters, 'validation_source', 'split')
        )
        return validation_source == 'external'

    def _prepare_dataset_folders(self) -> tuple[Path, Path]:
        return self._prepare_dataset_pair(
            self.tranining_parameters.image_path,
            self.tranining_parameters.label_path,
            purpose='train',
        )

    def _prepare_external_validation_folders(self) -> tuple[Path, Path]:
        validation_image_path = getattr(self.tranining_parameters, 'validation_image_path', None)
        validation_label_path = getattr(self.tranining_parameters, 'validation_label_path', None)
        if validation_image_path is None or validation_label_path is None:
            self.message_bus.publish('error', 'External validation requires both image and label folders.')
            self._need_stop = True
            return Path(), Path()
        return self._prepare_dataset_pair(
            Path(validation_image_path),
            Path(validation_label_path),
            purpose='validation',
        )

    def _prepare_dataset_pair(self, dataset_image: Path, dataset_label: Path, *, purpose: str) -> tuple[Path, Path]:
        if self.dataset_source is not None:
            if dataset_image != self.dataset_source.image_root or dataset_label != self.dataset_source.label_root:
                raise ValueError('The selected dataset is not registered in the server input source')
            return dataset_image, dataset_label
        normalized_purpose = str(purpose or 'train').strip().lower()
        is_training_data = normalized_purpose == 'train'

        recursive = bool(getattr(self.tranining_parameters, 'recursive_file_search', False))
        if filter_files(dataset_label, ('.cif',), recursive=recursive):
            binary_dir_name = 'binary_cif' if is_training_data else f'binary_cif_{normalized_purpose}'
            binary_labels = dataset_label.parent / binary_dir_name
            self._start_cif_conversion(dataset_label, binary_labels)
            dataset_label = binary_labels

        return dataset_image, dataset_label
