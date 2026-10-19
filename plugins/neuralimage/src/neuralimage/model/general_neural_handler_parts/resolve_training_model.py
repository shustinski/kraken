"""general_neural_handler: resolve training model."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from neuralimage.lib.data_interfaces import (
    WorkMode,
)
from neuralimage.lib.func import get_input_channels
from neuralimage.model.NeuralNetwork import create_model
from neuralimage.model.NeuralNetwork.model_io import load_model_artifact
from neuralimage.model.image_workers import ConvertCifThread
from neuralimage.preprocessing.config import PreprocessingConfig, build_preprocessing_config


class ResolveTrainingModelMixin:
    def _resolve_training_model(self):
        from neuralimage.model.general_neural_handler import create_model, load_model_artifact

        artifact_dir = self._resolve_training_artifact_dir()
        if self.work_mode in (WorkMode.train_and_recognition, WorkMode.train_only):
            model_name = str(self.recognition_parameters.model)
            model_kwargs = self._resolve_model_creation_kwargs(model_name)
            model = create_model(model_name, self.tranining_parameters.colors, **model_kwargs)
            setattr(model, '_neuralimage_model_name', model_name)
            setattr(model, '_neuralimage_input_channels', int(self.tranining_parameters.colors))
            setattr(model, '_neuralimage_model_kwargs', dict(model_kwargs))
            model_save_path = artifact_dir / self._declare_model_name()
        else:
            model = load_model_artifact(self.recognition_parameters.model, map_location='cpu')
            self._validate_loaded_model_input_channels(model)
            self._apply_artifact_preprocessing_for_training(model)
            if self.work_mode in (WorkMode.further_training, WorkMode.continue_training) and hasattr(model, 'deep_supervision'):
                deep_supervision_enabled = bool(getattr(self.tranining_parameters, 'deep_supervision', True))
                setattr(model, 'deep_supervision', deep_supervision_enabled)
                model_kwargs = getattr(model, '_neuralimage_model_kwargs', {})
                if not isinstance(model_kwargs, dict):
                    model_kwargs = {}
                model_kwargs = dict(model_kwargs)
                model_kwargs['deep_supervision'] = deep_supervision_enabled
                setattr(model, '_neuralimage_model_kwargs', model_kwargs)
            model_save_path = artifact_dir / Path(self.recognition_parameters.model).name
        return model, model_save_path

    def _apply_artifact_preprocessing_for_training(self, model: Any) -> None:
        metadata = getattr(model, '_neuralimage_artifact_metadata', {})
        payload = metadata.get('preprocessing') if isinstance(metadata, dict) else None
        if not isinstance(payload, dict) or not isinstance(payload.get('config'), dict):
            return
        artifact_config = build_preprocessing_config(payload['config'])
        artifact_hash = str(payload.get('hash') or artifact_config.stable_hash())
        requested = getattr(self.tranining_parameters, 'preprocessing', None) or PreprocessingConfig()
        if requested.any_enabled() and requested.stable_hash() != artifact_hash:
            raise ValueError(
                'Training preprocessing is incompatible with the loaded model artifact. '
                'Use the preprocessing stored with the model or select a compatible checkpoint.'
            )
        self.tranining_parameters.preprocessing = artifact_config

    def _resolve_loaded_model_input_channels(self, model: Any) -> int:
        declared_channels = getattr(model, '_neuralimage_input_channels', None)
        if declared_channels is not None:
            try:
                return int(declared_channels)
            except (TypeError, ValueError):
                pass
        return int(get_input_channels(model))

    def _validate_loaded_model_input_channels(self, model: Any) -> bool:
        expected_channels = max(1, int(getattr(self.tranining_parameters, 'colors', 1)))
        actual_channels = self._resolve_loaded_model_input_channels(model)
        if actual_channels == expected_channels:
            return True

        selected_mode = 'RGB' if expected_channels == 3 else 'grayscale'
        self.message_bus.publish(
            'error',
            (
                'Training input channels mismatch: '
                f'selected {selected_mode} mode ({expected_channels} channels), '
                f'but the loaded model expects {actual_channels} channel(s). '
                'Choose a matching checkpoint or change the color mode.'
            ),
        )
        self._need_stop = True
        return False

    def _resolve_training_artifact_dir(self) -> Path:
        artifact_dir = getattr(self.tranining_parameters, 'artifact_dir', None)
        if artifact_dir is not None:
            resolved = Path(artifact_dir)
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved
        if self.work_mode in (WorkMode.further_training, WorkMode.continue_training) and str(getattr(self.recognition_parameters, 'model', '')).strip():
            return Path(self.recognition_parameters.model).parent
        return self.tranining_parameters.image_path.parent

    def _start_cif_conversion(self, source: Path, result: Path):
        if self._check_folder_existance(result):
            return
        self.current_thread = ConvertCifThread(
            source,
            result,
            message_bus=self.message_bus,
            recursive=bool(getattr(self.tranining_parameters, 'recursive_file_search', False)),
        )
        self.current_thread.start()
        self._wait_for_current_thread('cif conversion')
