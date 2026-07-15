from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from .updater import UpdateInfo, UpdateService, launch_installer


class _UpdateWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation: Callable[[], object], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._operation = operation

    def run(self) -> None:
        try:
            self.succeeded.emit(self._operation())
        except Exception as exc:  # UI boundary: report network/filesystem errors to the user.
            self.failed.emit(str(exc))


class QtUpdateController(QObject):
    """Ready-to-use Qt frontend for :class:`UpdateService`."""

    status_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget,
        service: UpdateService,
        *,
        application_name: str,
    ) -> None:
        super().__init__(parent)
        self.parent_widget = parent
        self.service = service
        self.application_name = application_name
        self._worker: _UpdateWorker | None = None

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def check(self, *, quiet: bool = False, include_dismissed: bool = False) -> None:
        if self.busy:
            return
        if not self.service.manifest_url:
            if not quiet:
                QMessageBox.information(self.parent_widget, "Updates", "Update source is not configured.")
            return
        self.status_changed.emit("Checking for updates…")
        self._run(
            lambda: self.service.check(include_dismissed=include_dismissed),
            lambda result: self._show_check_result(result, quiet=quiet),
            quiet=quiet,
        )

    def _show_check_result(self, result: object, *, quiet: bool) -> None:
        update = result if isinstance(result, UpdateInfo) else None
        if update is None:
            self.status_changed.emit("Up to date")
            if not quiet:
                QMessageBox.information(self.parent_widget, "Updates", "The latest version is already installed.")
            return
        self.status_changed.emit(f"Version {update.version} is available")
        message = QMessageBox(self.parent_widget)
        message.setIcon(QMessageBox.Icon.Information)
        message.setWindowTitle("Update available")
        message.setText(f"{self.application_name} {update.version} is available.")
        message.setInformativeText(update.release_notes or "Download and install it now?")
        install_button = message.addButton("Download and install", QMessageBox.ButtonRole.AcceptRole)
        later_button = message.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        message.exec()
        if message.clickedButton() is install_button:
            self._download(update)
        else:
            self.service.dismiss_for_session(update)
            self.status_changed.emit(f"Version {update.version} postponed until next launch")
        del later_button

    def _download(self, update: UpdateInfo) -> None:
        self.status_changed.emit(f"Downloading {update.version}…")
        self._run(lambda: self.service.download(update), lambda path: self._confirm_install(update, path), quiet=False)

    def _confirm_install(self, update: UpdateInfo, value: object) -> None:
        path = Path(value)
        answer = QMessageBox.question(
            self.parent_widget,
            "Install update",
            f"Version {update.version} was downloaded. Close the application and start the installer?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.status_changed.emit("Installation cancelled")
            return
        launch_installer(path)
        QApplication.instance().quit()

    def _run(self, operation: Callable[[], object], on_success: Callable[[object], None], *, quiet: bool) -> None:
        worker = _UpdateWorker(operation, self)
        self._worker = worker

        def finished(result: object) -> None:
            self._worker = None
            on_success(result)

        def failed(error: str) -> None:
            self._worker = None
            self.status_changed.emit("Update check failed")
            if not quiet:
                QMessageBox.warning(self.parent_widget, "Updates", f"Update operation failed.\n\n{error}")

        worker.succeeded.connect(finished)
        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
