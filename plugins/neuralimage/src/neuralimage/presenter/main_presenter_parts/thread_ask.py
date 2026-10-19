"""main_presenter: thread ask."""
from __future__ import annotations
import threading
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox
from PyQt6 import QtCore
from neuralimage.application.services import (
    can_start_processing,
)
from neuralimage.lib.data_interfaces import (
    CutSettings,
)
from neuralimage.lib.images import SampleWorker
from neuralimage.lib.update_checker import (
    UpdateInfo,
    collect_release_history,
    is_newer_version,
    load_selected_update_channel,
    load_update_client_config,
    load_last_notified_version,
    load_update_manifest_url,
    save_last_notified_version,
    should_notify_version,
)
from neuralimage.lib.version import APP_VERSION
from neuralimage.lib.ui_texts import get_ui_section
import neuralimage.presenter.dialogs as presenter_dialogs
import neuralimage.presenter.sample_counting as sample_counting
import neuralimage.presenter.update_flow as update_flow
from neuralimage.presenter.background_workers import (
    AppUpdateCheckThread as PresenterAppUpdateCheckThread,
)


class ThreadAskMixin:
    def _stop_owned_qthread(self, attr_name: str, wait_ms: int, operation_name: str) -> None:
        thread = getattr(self, attr_name, None)
        if thread is None:
            return
        self._stop_worker_object(thread, operation_name)
        self._wait_for_qthread(thread, wait_ms, operation_name)
        if not self._is_qthread_running(thread):
            setattr(self, attr_name, None)

    def _stop_worker_object(self, worker: object, operation_name: str = 'worker') -> None:
        stop = getattr(worker, 'detach', None) or getattr(worker, 'stop', None)
        try:
            if callable(stop):
                stop()
                return
            request_interruption = getattr(worker, 'requestInterruption', None)
            if callable(request_interruption):
                request_interruption()
            quit_thread = getattr(worker, 'quit', None)
            if callable(quit_thread):
                quit_thread()
        except Exception as exc:
            self.message_bus.publish('error', f'Failed to request {operation_name} stop during shutdown: {exc}')

    def _wait_for_qthread(self, thread: object, wait_ms: int, operation_name: str) -> None:
        wait = getattr(thread, 'wait', None)
        if not callable(wait) or not self._is_qthread_running(thread):
            return
        try:
            stopped = bool(wait(max(0, int(wait_ms))))
        except Exception as exc:
            self.message_bus.publish('error', f'Failed while waiting for {operation_name} during shutdown: {exc}')
            return
        if stopped or not self._is_qthread_running(thread):
            return
        self.message_bus.publish(
            'error',
            f'Failed to stop {operation_name} within {max(0, int(wait_ms))} ms during shutdown.',
        )

    @staticmethod
    def _is_qthread_running(thread: object) -> bool:
        is_running = getattr(thread, 'isRunning', None)
        return bool(is_running()) if callable(is_running) else False

    def _on_close_requested(self):
        ui_texts = get_ui_section('main_window')
        reply = QMessageBox.question(
            self.view,
            str(ui_texts.get('exit_title', 'Выход')),
            str(ui_texts.get('exit_text', 'Закрыть программу?')),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._save_windows_to_qsettings()
            self.shutdown()
            self.view.allow_close()

    def _calculate_expected_samples(self):
        sample_counting.calculate_expected_samples(self)

    def get_cut_settings_from_window_state(self) -> CutSettings:
        return sample_counting.get_cut_settings_from_window_state(self)

    def _set_sample_number(self, calculator_settings: CutSettings) -> None:
        sample_counting.set_sample_number(self, calculator_settings)

    def _invalidate_sample_count_requests(self) -> None:
        sample_counting.invalidate_sample_count_requests(self)

    def _dispatch_sample_count_request(self) -> None:
        sample_counting.dispatch_sample_count_request(self)

    def _start_sample_count_request(self, request: tuple[int, str, CutSettings, object]) -> None:
        from neuralimage.presenter.main_presenter import SampleWorker

        sample_counting.start_sample_count_request(
            self,
            request,
            sample_worker_cls=SampleWorker,
            threading_module=threading,
        )

    def _run_sample_count_request(
        self,
        request_id: int,
        sample_folder: str,
        calculator_settings: CutSettings,
        synthetic_config: object,
        cached_path: str | None,
        cached_sizes: list[tuple[int, int]] | None,
    ) -> None:
        from neuralimage.presenter.main_presenter import SampleWorker

        sample_counting.run_sample_count_request(
            self,
            request_id,
            sample_folder,
            calculator_settings,
            synthetic_config,
            cached_path,
            cached_sizes,
            sample_worker_cls=SampleWorker,
        )

    @staticmethod
    def _normalize_sample_count_path(path: Path | str) -> str:
        return sample_counting.normalize_sample_count_path(path)

    @QtCore.pyqtSlot(int, str, object, int)
    def _on_sample_count_calculated(
        self,
        request_id: int,
        normalized_path: str,
        image_sizes: object,
        total_samples: int,
    ) -> None:
        sample_counting.on_sample_count_calculated(
            self,
            request_id,
            normalized_path,
            image_sizes,
            total_samples,
        )

    @QtCore.pyqtSlot(int, str)
    def _on_sample_count_failed(self, request_id: int, error_message: str) -> None:
        sample_counting.on_sample_count_failed(self, request_id, error_message)

    def _start_pending_sample_count_request_if_needed(self) -> None:
        sample_counting.start_pending_sample_count_request_if_needed(self)

    def _set_max_shift(self):
        sample_counting.set_max_shift(self)

    # ------------------------------------------------------------------ #
    #   Валидация и управление кнопкой "Начать"
    # ------------------------------------------------------------------ #
    def _validate_start_button(self):
        """Проверяем, что все обязательные поля заполнены."""
        self._update_main_window_state()
        self._update_settings_window_state()
        valid_state = can_start_processing(self.main_window_state, self.settings_state)
        self.view.enable_start.emit(valid_state and getattr(self, '_settings_configuration_valid', True))

    # ------------------------------------------------------------------ #
    #   Управление началом потоков
    # ------------------------------------------------------------------ #
    def _thread_ask(
        self,
        question: str,
        header: str = 'Обратите внимание',
        default_answer: bool = False,
        timeout_seconds: int = 0,
    ):
        answer = presenter_dialogs.ask_yes_no_with_timeout(
            parent=self.view,
            question=question,
            header=header,
            default_answer=default_answer,
            timeout_seconds=timeout_seconds,
        )
        handler = self.neuaral_handler
        if handler is not None:
            handler.answer.emit(answer)

    def _start_update_check(self, *, manual: bool = False) -> None:
        update_flow.start_update_check(
            self,
            manual=manual,
            load_update_manifest_url_fn=load_update_manifest_url,
            update_check_thread_cls=PresenterAppUpdateCheckThread,
            load_update_client_config_fn=load_update_client_config,
            load_selected_update_channel_fn=load_selected_update_channel,
        )

    def _clear_update_check_thread(self) -> None:
        update_flow.clear_update_check_thread(self)

    def _on_update_check_finished(self, update_info: object) -> None:
        update_flow.on_update_check_finished(
            self,
            update_info,
            update_info_cls=UpdateInfo,
            app_version=APP_VERSION,
            is_newer_version_fn=is_newer_version,
            load_last_notified_version_fn=load_last_notified_version,
            should_notify_version_fn=should_notify_version,
            save_last_notified_version_fn=save_last_notified_version,
        )

    @staticmethod
    def _markdown_to_message_html(markdown: str) -> str:
        return presenter_dialogs.markdown_to_message_html(markdown)

    def _show_update_notification(self, update_info: UpdateInfo, *, manual: bool = False) -> None:
        update_flow.show_update_notification(
            self,
            update_info,
            manual=manual,
            app_version=APP_VERSION,
            collect_release_history_fn=collect_release_history,
        )

    def _show_release_selector(self, update_info: UpdateInfo) -> None:
        update_flow.show_release_selector(self, update_info, app_version=APP_VERSION)
