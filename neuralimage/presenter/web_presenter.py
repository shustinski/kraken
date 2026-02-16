from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from lib.message_bus import AbstractMessageBus
from presenter.state_store import IniStateStore, StateStore
from presenter.validation import can_start_processing
from presenter.workflow_mapper import build_workflow_parameters
from view.window_dataclasses import MainWindowState, SettingsState


QuestionModule = Callable[[str, str], bool]
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

    def __init__(self, state_store: StateStore | None = None) -> None:
        self._state_store = state_store or IniStateStore()

    def load_initial_states(self) -> tuple[MainWindowState, SettingsState]:
        return self._state_store.load_main_window_state(), self._state_store.load_settings_state()

    def build_handler(
        self,
        main_state: MainWindowState,
        settings_state: SettingsState,
        message_bus: AbstractMessageBus,
        question_module: QuestionModule,
        callback: Callable[..., None] | None = None,
    ) -> BuildHandlerResult:
        if not can_start_processing(main_state):
            return BuildHandlerResult(None, 'Fill in required fields and verify that all paths exist.')

        work_mode, training_parameters, recognition_parameters = build_workflow_parameters(
            main_state,
            settings_state,
        )
        if work_mode is None:
            return BuildHandlerResult(None, 'Invalid work mode.')

        self._state_store.save_main_window_state(main_state)
        self._state_store.save_settings_state(settings_state)

        try:
            from model.general_neural_handler import GeneralNeuralHandler

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

