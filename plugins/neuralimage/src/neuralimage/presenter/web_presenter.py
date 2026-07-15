from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.ports import StateStore
from neuralimage.application.services import build_workflow_parameters, can_start_processing
from neuralimage.application.services.training_artifacts import build_training_artifact_dir
from neuralimage.infrastructure.config.state_store import WORKFLOW_SNAPSHOT_FILENAME, save_workflow_snapshot
from neuralimage.lib.data_interfaces import WorkMode
from neuralimage.lib.message_bus import AbstractMessageBus


QuestionModule = Callable[..., bool]
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildHandlerResult:
    handler: object | None
    error: str | None = None


class WebPresenter:
    """
    Presenter for Web UI.
    Keeps web handlers/services thin and centralizes business mapping logic
    similarly to Qt presenter responsibilities.
    """

    def __init__(self, state_store: StateStore) -> None:
        self._state_store = state_store

    def load_initial_states(self) -> tuple[MainWindowState, SettingsState]:
        return self._state_store.load_main_window_state(), self._state_store.load_settings_state()

    def build_handler(
        self,
        main_state: MainWindowState,
        settings_state: SettingsState,
        message_bus: AbstractMessageBus,
        question_module: QuestionModule,
        callback: Callable[..., None] | None = None,
        runtime_context: Any | None = None,
    ) -> BuildHandlerResult:
        if not can_start_processing(main_state, settings_state):
            return BuildHandlerResult(None, 'Fill in required fields and verify that all paths exist.')

        work_mode, training_parameters, recognition_parameters = build_workflow_parameters(
            main_state,
            settings_state,
        )
        if work_mode is None:
            return BuildHandlerResult(None, 'Invalid work mode.')
        if (
            runtime_context is not None
            and getattr(runtime_context, 'training_completed', False)
            and work_mode in (WorkMode.train_and_recognition, WorkMode.further_training)
            and getattr(runtime_context, 'trained_model_path', None) is not None
        ):
            work_mode = WorkMode.recognition_only
            recognition_parameters.model = runtime_context.trained_model_path

        if work_mode in (
            WorkMode.train_only,
            WorkMode.continue_training,
            WorkMode.train_and_recognition,
            WorkMode.further_training,
        ):
            try:
                artifact_dir = getattr(runtime_context, 'artifact_dir', None)
                if artifact_dir is None:
                    artifact_dir = build_training_artifact_dir(main_state, settings_state, work_mode)
                    if runtime_context is not None:
                        runtime_context.artifact_dir = artifact_dir
                training_parameters.artifact_dir = artifact_dir
                training_parameters.resume_from_checkpoint = bool(
                    runtime_context is not None and getattr(runtime_context, 'training_checkpoint', None)
                )
                save_workflow_snapshot(
                    main_state,
                    settings_state,
                    destination=Path(artifact_dir) / WORKFLOW_SNAPSHOT_FILENAME,
                    workflow_snapshot=(work_mode, training_parameters, recognition_parameters),
                )
            except OSError as error:
                return BuildHandlerResult(None, f'Failed to prepare run artifacts: {error}')

        if recognition_parameters is not None:
            manifest_path = Path(recognition_parameters.result_folder) / '.neuralimage-recognition-progress.json'
            recognition_parameters.resume_manifest_path = manifest_path
            recognition_parameters.resume_from_manifest = bool(
                runtime_context is not None and getattr(runtime_context, 'recognition_manifest', None)
            )
            if runtime_context is not None and runtime_context.recognition_manifest is None:
                runtime_context.recognition_manifest = manifest_path

        self._state_store.save_main_window_state(main_state)
        self._state_store.save_settings_state(settings_state)

        try:
            from neuralimage.model.general_neural_handler import GeneralNeuralHandler

            handler = GeneralNeuralHandler(
                work_mode=work_mode,
                recogniton_parameters=recognition_parameters,
                tranining_parameters=training_parameters,
                question_module=question_module,
                message_bus=message_bus,
                callback=callback,
            )
        except Exception as error:
            _LOG.exception('Failed to initialize GeneralNeuralHandler')
            return BuildHandlerResult(None, f'Failed to initialize business logic: {error}')

        return BuildHandlerResult(handler=handler)

