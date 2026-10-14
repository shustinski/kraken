"""presenter: shutdown."""
from __future__ import annotations


class ShutdownMixin:
    def _persist_state(self) -> None:
        self._settings_service.save_folder_manager_payload(self._build_folder_manager_payload())
        self._settings_service.save_build_settings_payload(self._build_build_settings_payload())
        self._settings_service.sync()

    def shutdown(self) -> None:
        if self._worker is not None:
            request_cancel = getattr(self._worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        thread = self._worker_thread
        if thread is not None:
            try:
                thread.quit()
            except Exception:
                pass
            if thread.isRunning():
                thread.wait(30000)
        self._cleanup_worker()
        self._close_all_details_dialogs()
        self._persist_state()

    # Preferred analytics entrypoint.
    def _start_compute_metrics(self) -> None:
        self._start_compute_analytics()

    # Legacy lite compatibility alias.
    def _start_compute_mismatches(self) -> None:
        self._start_compute_metrics()

    def _set_base_folder(self) -> None:
        self._set_original_folder()

    def _clear_base_folder(self) -> None:
        self._clear_original_folder()
