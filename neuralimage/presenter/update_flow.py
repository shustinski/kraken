from __future__ import annotations

import webbrowser

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QMessageBox,
    QTextBrowser,
    QVBoxLayout,
)

from lib.ui_texts import get_ui_section
from presenter.dialogs import markdown_to_message_html


class _UpdateNotificationDialog(QDialog):
    def __init__(
        self,
        parent,
        *,
        title: str,
        body_markdown: str,
        download_available: bool,
        has_releases: bool,
        texts: dict[str, object],
    ) -> None:
        super().__init__(parent)
        self._selected_action = 'later'
        self.setWindowTitle(str(title))
        self.resize(860, 640)

        layout = QVBoxLayout(self)

        self.content_view = QTextBrowser(self)
        self.content_view.setReadOnly(True)
        self.content_view.setOpenExternalLinks(True)
        self.content_view.setOpenLinks(True)
        self.content_view.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        self.content_view.setHtml(markdown_to_message_html(body_markdown))
        self.content_view.moveCursor(self.content_view.textCursor().MoveOperation.Start)
        layout.addWidget(self.content_view)

        buttons = QDialogButtonBox(self)
        later_button = buttons.addButton(
            str(texts.get('update_later', 'Позже')),
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        later_button.clicked.connect(self.reject)

        if download_available:
            install_button = buttons.addButton(
                str(texts.get('update_download_install', 'Скачать и установить')),
                QDialogButtonBox.ButtonRole.AcceptRole,
            )
            install_button.clicked.connect(lambda: self._finish('install'))

        if has_releases:
            select_version_button = buttons.addButton(
                str(texts.get('update_select_version', 'Выбрать версию')),
                QDialogButtonBox.ButtonRole.ActionRole,
            )
            select_version_button.clicked.connect(lambda: self._finish('select_version'))

        if download_available:
            open_button = buttons.addButton(
                str(texts.get('update_open_download', 'Скачать')),
                QDialogButtonBox.ButtonRole.ActionRole,
            )
            open_button.clicked.connect(lambda: self._finish('open_download'))

        layout.addWidget(buttons)

    @property
    def selected_action(self) -> str:
        return self._selected_action

    def _finish(self, action: str) -> None:
        self._selected_action = str(action)
        self.accept()


def refresh_update_client_config(
    presenter,
    *,
    load_update_client_config_fn,
    load_selected_update_channel_fn,
) -> None:
    presenter._update_client_config = load_update_client_config_fn()
    presenter._selected_update_channel = load_selected_update_channel_fn(
        presenter._update_client_config.default_channel,
        available_channels=presenter._update_client_config.available_channels,
    )
    presenter.view.configure_update_channels(
        presenter._update_client_config.available_channels,
        presenter._selected_update_channel,
    )


def on_update_channel_selected(
    presenter,
    channel: str,
    *,
    save_selected_update_channel_fn,
    load_update_client_config_fn,
    load_selected_update_channel_fn,
) -> None:
    refresh_update_client_config(
        presenter,
        load_update_client_config_fn=load_update_client_config_fn,
        load_selected_update_channel_fn=load_selected_update_channel_fn,
    )
    requested_channel = str(channel or '').strip().lower()
    available_channels = presenter._update_client_config.available_channels
    if requested_channel in available_channels:
        presenter._selected_update_channel = requested_channel
        save_selected_update_channel_fn(requested_channel)
        presenter.view.configure_update_channels(available_channels, requested_channel)
        presenter.message_bus.publish('logging', f'Канал обновлений переключен на {requested_channel}.')


def start_update_check(
    presenter,
    *,
    manual: bool,
    load_update_manifest_url_fn,
    update_check_thread_cls,
    load_update_client_config_fn,
    load_selected_update_channel_fn,
) -> None:
    texts = get_ui_section('main_window')
    refresh_update_client_config(
        presenter,
        load_update_client_config_fn=load_update_client_config_fn,
        load_selected_update_channel_fn=load_selected_update_channel_fn,
    )
    manifest_url = load_update_manifest_url_fn(presenter._selected_update_channel)
    if not manifest_url:
        if manual:
            presenter.view.show_warning.emit(
                str(texts.get('update_check_not_configured', 'Источник обновлений не настроен.'))
            )
        return
    if presenter._update_check_thread is not None:
        if manual:
            presenter.view.show_info.emit(
                str(texts.get('update_check_in_progress', 'Проверка обновлений уже выполняется.'))
            )
        return
    presenter._update_check_manual = manual
    presenter.message_bus.publish('logging', 'Запущена проверка обновлений.')
    presenter._update_check_thread = update_check_thread_cls(
        manifest_url=manifest_url,
        channel=presenter._selected_update_channel,
    )
    presenter._update_check_thread.checked.connect(presenter._on_update_check_finished)
    presenter._update_check_thread.finished.connect(presenter._clear_update_check_thread)
    presenter._update_check_thread.start()


def clear_update_check_thread(presenter) -> None:
    presenter._update_check_thread = None
    presenter._update_check_manual = False


def on_update_check_finished(
    presenter,
    update_info: object,
    *,
    update_info_cls,
    app_version: str,
    is_newer_version_fn,
    load_last_notified_version_fn,
    should_notify_version_fn,
    save_last_notified_version_fn,
) -> None:
    texts = get_ui_section('main_window')
    manual = presenter._update_check_manual
    if not isinstance(update_info, update_info_cls):
        if manual:
            presenter.view.show_warning.emit(
                str(texts.get('update_check_failed', 'Не удалось проверить наличие обновлений.'))
            )
        return
    if manual:
        presenter._show_update_notification(update_info, manual=True)
        return
    if not is_newer_version_fn(update_info.version, app_version):
        return
    last_notified = load_last_notified_version_fn(presenter._selected_update_channel)
    if not should_notify_version_fn(update_info.version, app_version, last_notified):
        return
    presenter._show_update_notification(update_info)
    save_last_notified_version_fn(update_info.version, presenter._selected_update_channel)


def show_update_notification(
    presenter,
    update_info,
    *,
    manual: bool,
    app_version: str,
    collect_release_history_fn,
) -> None:
    texts = get_ui_section('main_window')
    title = str(texts.get('update_available_title', 'Доступна новая версия'))
    template_key = 'update_manual_text' if manual else 'update_available_text'
    body_template = str(
        texts.get(
            template_key,
            'Установлена версия {current_version}. Доступна версия {new_version}.',
        )
    )
    body = body_template.format(
        current_version=app_version,
        new_version=update_info.version,
        channel=presenter._selected_update_channel,
    )
    release_history = collect_release_history_fn(update_info)
    body_markdown = body
    if release_history:
        history_title = str(texts.get('update_release_history_title', 'История релизов:'))
        body_markdown = f'{body_markdown}\n\n## {history_title}\n\n{release_history}'

    dialog = _UpdateNotificationDialog(
        presenter.view,
        title=title,
        body_markdown=body_markdown,
        download_available=bool(update_info.download_url),
        has_releases=bool(update_info.releases),
        texts=texts,
    )
    dialog.exec()
    if dialog.selected_action == 'install':
        presenter._start_update_download(presenter._resolve_latest_release(update_info))
        return
    if dialog.selected_action == 'select_version':
        presenter._show_release_selector(update_info)
        return
    if dialog.selected_action == 'open_download':
        try:
            webbrowser.open(update_info.download_url)
        except Exception:
            pass


def show_release_selector(presenter, update_info, *, app_version: str) -> None:
    texts = get_ui_section('main_window')
    releases = tuple(update_info.releases)
    if not releases:
        presenter.view.show_warning.emit(
            str(texts.get('update_check_failed', 'Не удалось проверить наличие обновлений.'))
        )
        return

    labels: list[str] = []
    initial_index = 0
    for index, release in enumerate(releases):
        label = str(release.version)
        if release.version == app_version:
            label = f'{label} ({str(texts.get("update_current_version_label", "текущая"))})'
            initial_index = index
        labels.append(label)

    selected_label, accepted = QInputDialog.getItem(
        presenter.view,
        str(texts.get('update_select_title', 'Выбор версии')),
        str(texts.get('update_select_label', 'Выберите версию для установки или отката:')),
        labels,
        initial_index,
        False,
    )
    if not accepted:
        return
    selected_release = releases[labels.index(selected_label)]
    presenter._confirm_release_install(selected_release)


def confirm_release_install(presenter, release, *, app_version: str) -> None:
    texts = get_ui_section('main_window')
    if not release.download_url:
        presenter.view.show_warning.emit(
            str(texts.get('update_missing_download', 'Для выбранной версии не задан путь к установщику.'))
        )
        return
    if release.version == app_version:
        presenter.view.show_info.emit(
            str(texts.get('update_selected_current', 'Выбранная версия уже установлена.'))
        )
        return

    question = str(
        texts.get(
            'update_confirm_selected',
            'Установить версию {selected_version} поверх текущей {current_version}?',
        )
    ).format(
        selected_version=release.version,
        current_version=app_version,
    )
    if release.notes:
        question = f'{question}\n\n{release.notes}'
    reply = QMessageBox.question(
        presenter.view,
        str(texts.get('update_install_title', 'Установка обновления')),
        question,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return
    presenter._start_update_download(release)


def start_update_download(
    presenter,
    release,
    *,
    update_download_thread_cls,
) -> None:
    if presenter._update_download_thread is not None:
        return
    is_running = False
    handler = presenter.neuaral_handler
    if handler is not None:
        is_running_method = getattr(handler, 'isRunning', None)
        if callable(is_running_method):
            is_running = bool(is_running_method())
    if is_running:
        texts = get_ui_section('main_window')
        presenter.view.show_warning.emit(
            str(
                texts.get(
                    'update_busy_message',
                    'Сначала остановите активную задачу, затем повторите обновление.',
                )
            )
        )
        return
    presenter.message_bus.publish('logging', f'Загрузка версии {release.version} запущена.')
    presenter._update_download_thread = update_download_thread_cls(release=release)
    presenter._update_download_thread.finished_download.connect(presenter._on_update_download_finished)
    presenter._update_download_thread.failed_download.connect(presenter._on_update_download_failed)
    presenter._update_download_thread.finished.connect(presenter._clear_update_download_thread)
    presenter._update_download_thread.start()


def clear_update_download_thread(presenter) -> None:
    presenter._update_download_thread = None


def on_update_download_finished(
    presenter,
    installer_path: str,
    version: str,
    *,
    launch_update_installer_fn,
) -> None:
    texts = get_ui_section('main_window')
    question = str(
        texts.get(
            'update_ready_to_install',
            'Обновление {new_version} загружено. Приложение будет закрыто, затем запустится установщик.\nПродолжить?',
        )
    ).format(new_version=version)
    reply = QMessageBox.question(
        presenter.view,
        str(texts.get('update_install_title', 'Установка обновления')),
        question,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return
    try:
        launch_update_installer_fn(installer_path)
    except OSError as exc:
        presenter._on_update_download_failed(str(exc))
        return
    presenter.message_bus.publish('logging', f'Запущен установщик обновления: {installer_path}.')
    presenter._save_windows_to_qsettings()
    presenter.view.allow_close()


def on_update_download_failed(presenter, error_message: str) -> None:
    texts = get_ui_section('main_window')
    message = str(
        texts.get(
            'update_download_failed',
            'Не удалось скачать или запустить обновление: {error}',
        )
    ).format(error=error_message)
    presenter.view.show_warning.emit(message)
    presenter.message_bus.publish('error', message)
