from __future__ import annotations

import webbrowser
from collections.abc import Callable
from pathlib import Path

from PyQt6 import QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QInputDialog, QMessageBox, QMenu, QTextBrowser, QVBoxLayout

from .client import (
    ReleaseInfo,
    UpdateInfo,
    collect_release_history,
    download_update_installer,
    fetch_update_info,
    is_newer_version,
    launch_update_installer,
    load_selected_update_channel,
    load_update_client_config,
    save_selected_update_channel,
)


class UpdateCheckThread(QtCore.QThread):
    checked = QtCore.pyqtSignal(object)

    def __init__(self, *, manifest_url: str, channel: str) -> None:
        super().__init__()
        self._manifest_url = str(manifest_url).strip()
        self._channel = str(channel or "").strip().lower()

    def run(self) -> None:
        self.checked.emit(fetch_update_info(self._manifest_url, expected_channel=self._channel))


class UpdateDownloadThread(QtCore.QThread):
    finished_download = QtCore.pyqtSignal(str, str)
    failed_download = QtCore.pyqtSignal(str)

    def __init__(self, *, release: ReleaseInfo, app_id: str) -> None:
        super().__init__()
        self._release = release
        self._app_id = app_id

    def run(self) -> None:
        try:
            installer_path = download_update_installer(self._release, app_id=self._app_id)
        except Exception as exc:
            self.failed_download.emit(str(exc))
            return
        self.finished_download.emit(str(installer_path), self._release.version)


class UpdateNotificationDialog(QDialog):
    def __init__(
        self,
        parent,
        *,
        title: str,
        body: str,
        download_available: bool,
        has_releases: bool,
    ) -> None:
        super().__init__(parent)
        self._selected_action = "later"
        self.setWindowTitle(title)
        self.resize(760, 520)
        layout = QVBoxLayout(self)
        self.content_view = QTextBrowser(self)
        self.content_view.setReadOnly(True)
        self.content_view.setOpenExternalLinks(True)
        self.content_view.setMarkdown(body)
        layout.addWidget(self.content_view)

        buttons = QDialogButtonBox(self)
        later_button = buttons.addButton("Позже", QDialogButtonBox.ButtonRole.RejectRole)
        later_button.clicked.connect(self.reject)
        if download_available:
            install_button = buttons.addButton("Скачать и установить", QDialogButtonBox.ButtonRole.AcceptRole)
            install_button.clicked.connect(lambda: self._finish("install"))
            open_button = buttons.addButton("Открыть ссылку", QDialogButtonBox.ButtonRole.ActionRole)
            open_button.clicked.connect(lambda: self._finish("open_download"))
        if has_releases:
            select_button = buttons.addButton("Выбрать версию", QDialogButtonBox.ButtonRole.ActionRole)
            select_button.clicked.connect(lambda: self._finish("select_version"))
        layout.addWidget(buttons)

    @property
    def selected_action(self) -> str:
        return self._selected_action

    def _finish(self, action: str) -> None:
        self._selected_action = action
        self.accept()


class QtUpdateController(QtCore.QObject):
    def __init__(
        self,
        parent,
        *,
        app_id: str,
        app_name: str,
        current_version: str,
        config_path: str | Path | None = None,
        env_prefix: str | None = None,
        settings_org: str | None = None,
        settings_app: str = "Updater",
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._app_id = app_id
        self._app_name = app_name
        self._current_version = current_version
        self._config_path = config_path
        self._env_prefix = env_prefix
        self._settings_org = settings_org or app_id
        self._settings_app = settings_app
        self._status_callback = status_callback
        self._check_thread: UpdateCheckThread | None = None
        self._download_thread: UpdateDownloadThread | None = None

    def add_menu_action(
        self,
        menu,
        text: str = "Check for updates",
        *,
        submenu_title: str = "Update",
        submenu_object_name: str = "",
        action_object_name: str = "",
    ) -> QAction:
        update_menu = QMenu(submenu_title, self._parent)
        if submenu_object_name:
            update_menu.setObjectName(submenu_object_name)
        menu.addMenu(update_menu)
        action = QAction(text, self._parent)
        if action_object_name:
            action.setObjectName(action_object_name)
        action.triggered.connect(lambda _checked=False: self.check_for_updates(manual=True))
        update_menu.addAction(action)
        return action

    def check_for_updates(self, *, manual: bool = True) -> None:
        config = load_update_client_config(
            app_id=self._app_id,
            config_path=self._config_path,
            env_prefix=self._env_prefix,
        )
        channel = load_selected_update_channel(
            config.default_channel,
            available_channels=config.available_channels,
            settings_org=self._settings_org,
            settings_app=self._settings_app,
        )
        manifest_url = config.get_manifest_url(channel)
        if not manifest_url:
            if manual:
                QMessageBox.information(self._parent, self._app_name, "Источник обновлений не настроен.")
            return
        if self._check_thread is not None:
            if manual:
                QMessageBox.information(self._parent, self._app_name, "Проверка обновлений уже выполняется.")
            return
        save_selected_update_channel(channel, settings_org=self._settings_org, settings_app=self._settings_app)
        self._set_status("Запущена проверка обновлений.")
        self._check_thread = UpdateCheckThread(manifest_url=manifest_url, channel=channel)
        self._check_thread.checked.connect(lambda update_info: self._on_check_finished(update_info, manual=manual, channel=channel))
        self._check_thread.finished.connect(self._clear_check_thread)
        self._check_thread.start()

    def _clear_check_thread(self) -> None:
        self._check_thread = None

    def _on_check_finished(self, update_info: object, *, manual: bool, channel: str) -> None:
        if not isinstance(update_info, UpdateInfo):
            if manual:
                QMessageBox.warning(self._parent, self._app_name, "Не удалось проверить наличие обновлений.")
            return
        if not manual and not is_newer_version(update_info.version, self._current_version):
            return
        self._show_update_notification(update_info, manual=manual, channel=channel)

    def _show_update_notification(self, update_info: UpdateInfo, *, manual: bool, channel: str) -> None:
        if manual:
            body = (
                f"Установлена версия: {self._current_version}\n\n"
                f"Выбранный канал: {channel}\n\n"
                f"Версия на сервере: {update_info.version}."
            )
        elif not is_newer_version(update_info.version, self._current_version):
            return
        else:
            body = (
                f"Установлена версия: {self._current_version}\n\n"
                f"Доступна версия: {update_info.version}."
            )
        history = collect_release_history(update_info)
        if history:
            body = f"{body}\n\n## История релизов\n\n{history}"
        dialog = UpdateNotificationDialog(
            self._parent,
            title="Доступна новая версия",
            body=body,
            download_available=bool(update_info.download_url or self._resolve_latest_release(update_info).download_url),
            has_releases=bool(update_info.releases),
        )
        dialog.exec()
        if dialog.selected_action == "install":
            self._start_download(self._resolve_latest_release(update_info))
        elif dialog.selected_action == "select_version":
            self._show_release_selector(update_info)
        elif dialog.selected_action == "open_download":
            webbrowser.open(update_info.download_url or self._resolve_latest_release(update_info).download_url)

    def _show_release_selector(self, update_info: UpdateInfo) -> None:
        releases = tuple(update_info.releases)
        if not releases:
            QMessageBox.warning(self._parent, self._app_name, "В манифесте нет релизов.")
            return
        labels = [f"{release.version} (текущая)" if release.version == self._current_version else release.version for release in releases]
        selected_label, accepted = QInputDialog.getItem(self._parent, "Выбор версии", "Выберите версию:", labels, 0, False)
        if not accepted:
            return
        release = releases[labels.index(selected_label)]
        if not release.download_url:
            QMessageBox.warning(self._parent, self._app_name, "Для выбранной версии не задан путь к установщику.")
            return
        if release.version == self._current_version:
            QMessageBox.information(self._parent, self._app_name, "Выбранная версия уже установлена.")
            return
        if QMessageBox.question(self._parent, "Установка обновления", f"Установить версию {release.version}?") == QMessageBox.StandardButton.Yes:
            self._start_download(release)

    def _start_download(self, release: ReleaseInfo) -> None:
        if self._download_thread is not None:
            return
        self._set_status(f"Загрузка версии {release.version} запущена.")
        self._download_thread = UpdateDownloadThread(release=release, app_id=self._app_id)
        self._download_thread.finished_download.connect(self._on_download_finished)
        self._download_thread.failed_download.connect(self._on_download_failed)
        self._download_thread.finished.connect(self._clear_download_thread)
        self._download_thread.start()

    def _clear_download_thread(self) -> None:
        self._download_thread = None

    def _on_download_finished(self, installer_path: str, version: str) -> None:
        answer = QMessageBox.question(
            self._parent,
            "Установка обновления",
            f"Обновление {version} загружено. Приложение будет закрыто, затем запустится установщик.\nПродолжить?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            launch_update_installer(installer_path)
        except OSError as exc:
            self._on_download_failed(str(exc))
            return
        self._set_status(f"Запущен установщик обновления: {installer_path}.")
        self._parent.close()

    def _on_download_failed(self, message: str) -> None:
        QMessageBox.warning(self._parent, self._app_name, f"Не удалось скачать или запустить обновление: {message}")

    def _set_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)

    @staticmethod
    def _resolve_latest_release(update_info: UpdateInfo) -> ReleaseInfo:
        for release in update_info.releases:
            if release.version == update_info.version:
                return release
        return ReleaseInfo(update_info.version, update_info.download_url, update_info.release_notes, update_info.channel)
