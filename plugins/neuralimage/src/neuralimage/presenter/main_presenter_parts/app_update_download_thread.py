"""main_presenter: app update download thread."""
from __future__ import annotations
from PyQt6.QtCore import QThread
from PyQt6 import QtCore
from neuralimage.lib.update_checker import (
    ReleaseInfo,
    download_update_installer,
    fetch_update_info,
)

class AppUpdateCheckThread(QThread):
    checked = QtCore.pyqtSignal(object)

    def __init__(self, *, manifest_url: str, channel: str) -> None:
        super().__init__()
        self._manifest_url = str(manifest_url).strip()
        self._channel = str(channel or '').strip().lower()

    def run(self) -> None:
        self.checked.emit(fetch_update_info(self._manifest_url, expected_channel=self._channel))

class AppUpdateDownloadThread(QThread):
    finished_download = QtCore.pyqtSignal(str, str)
    failed_download = QtCore.pyqtSignal(str)

    def __init__(self, *, release: ReleaseInfo) -> None:
        super().__init__()
        self._release = release

    def run(self) -> None:
        try:
            installer_path = download_update_installer(self._release)
        except Exception as exc:
            self.failed_download.emit(str(exc))
            return
        self.finished_download.emit(str(installer_path), self._release.version)
