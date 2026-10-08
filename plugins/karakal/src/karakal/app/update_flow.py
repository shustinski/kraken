"""Karakal updates: the Update button, the offer at start and the Help menu actions.

The button has three states. Grey: no newer version known, a click checks now. Blue: a newer
version is in the update folder, a click installs it. Downloading: the installer is copied,
the button shows the percent. The update folder is checked at start and every 30 minutes.

At start (standalone app only) the program asks before the main window opens, every launch
until the user updates. The shared ``updater`` package is used only for its threads and
helpers; its dialogs stay with the other plugins.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QEventLoop, QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from updater.client import (
    ReleaseInfo,
    UpdateInfo,
    download_update_installer,
    get_update_staging_dir,
    is_newer_version,
    launch_update_installer,
    verify_installer_sha256,
)
from updater.qt import UpdateCheckThread

from ..core.features import display_version
from ..updater import (
    KARAKAL_UPDATE_APP_ID,
    KARAKAL_UPDATE_CHECK_TIMEOUT_SECONDS,
    karakal_channel_manifest_location,
    load_karakal_update_channel,
    save_karakal_last_update_check,
)
from ..version import __version__

UPDATE_CHECK_INTERVAL_MS = 30 * 60 * 1000
STARTUP_CHECK_TIMEOUT_MS = 2000
_COPY_CHUNK_BYTES = 4 * 1024 * 1024

STATE_IDLE = "idle"
STATE_AVAILABLE = "available"
STATE_DOWNLOADING = "downloading"

# Checks that outlived their window or the startup wait: kept until the thread ends.
_LIVE_THREADS: set[QThread] = set()


def _keep_until_finished(thread: QThread) -> None:
    if thread.isRunning():
        _LIVE_THREADS.add(thread)
        thread.finished.connect(lambda t=thread: _LIVE_THREADS.discard(t))


def current_version() -> str:
    return display_version() or __version__


def latest_release(info: UpdateInfo) -> ReleaseInfo:
    for release in info.releases:
        if release.version == info.version:
            return release
    return ReleaseInfo(info.version, info.download_url, info.release_notes, info.channel, sha256=info.sha256)


def notes_since(info: UpdateInfo, installed: str) -> str:
    """Release notes of every version newer than the installed one, newest first."""

    chunks = [
        f"### {release.version}\n\n{str(release.notes).strip()}".rstrip()
        for release in info.releases
        if is_newer_version(release.version, installed)
    ]
    if not chunks and str(info.release_notes).strip():
        chunks.append(f"### {info.version}\n\n{str(info.release_notes).strip()}")
    return "\n\n".join(chunks)


class InstallerCopyThread(QThread):
    """Copy the installer from the update folder to the temp folder, with progress."""

    progressed = pyqtSignal(int)
    copied = pyqtSignal(str, str)
    failed = pyqtSignal(str)

    def __init__(self, release: ReleaseInfo) -> None:
        super().__init__()
        self._release = release

    def run(self) -> None:
        try:
            source = Path(str(self._release.download_url or "").strip())
            if not source.is_file():
                # Not a file in the update folder (e.g. a link): no progress, the shared download.
                path = download_update_installer(self._release, app_id=KARAKAL_UPDATE_APP_ID)
                self.copied.emit(str(path), self._release.version)
                return
            target_dir = get_update_staging_dir(KARAKAL_UPDATE_APP_ID)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            total = max(1, source.stat().st_size)
            done = 0
            with source.open("rb") as reader, target.open("wb") as writer:
                while chunk := reader.read(_COPY_CHUNK_BYTES):
                    writer.write(chunk)
                    done += len(chunk)
                    self.progressed.emit(min(99, int(done * 100 / total)))
            verify_installer_sha256(target, getattr(self._release, "sha256", ""))
            self.progressed.emit(100)
            self.copied.emit(str(target), self._release.version)
        except Exception as error:  # noqa: BLE001 - shown to the user as is
            self.failed.emit(str(error))


def _launch_installer(installer_path: str, parent: QWidget | None) -> bool:
    try:
        launch_update_installer(installer_path)
    except OSError as error:
        QMessageBox.warning(parent, "Karakal", f"Не удалось запустить установщик: {error}")
        return False
    return True


class UpdateOfferDialog(QDialog):
    """«Version X is available»: what is new since the installed version, update now or later."""

    def __init__(self, parent: QWidget | None, info: UpdateInfo, t: Callable[..., str]) -> None:
        super().__init__(parent)
        installed = current_version()
        self.setWindowTitle(t("update.offer.title"))
        self.resize(720, 520)
        layout = QVBoxLayout(self)
        header = QLabel(t("update.offer.header", version=info.version, installed=installed), self)
        header.setWordWrap(True)
        header.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(header)
        notes = QTextBrowser(self)
        notes.setOpenExternalLinks(True)
        notes.setMarkdown(notes_since(info, installed) or t("update.offer.no_notes"))
        layout.addWidget(notes, stretch=1)
        hint = QLabel(t("update.offer.hint"), self)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(self)
        later = buttons.addButton(t("update.offer.later"), QDialogButtonBox.ButtonRole.RejectRole)
        now = buttons.addButton(t("update.offer.now"), QDialogButtonBox.ButtonRole.AcceptRole)
        now.setDefault(True)
        later.clicked.connect(self.reject)
        now.clicked.connect(self.accept)
        layout.addWidget(buttons)


def _copy_with_progress_dialog(release: ReleaseInfo, parent: QWidget | None, t: Callable[..., str]) -> str | None:
    """Modal copy of the installer (used at start, before the main window)."""

    dialog = QProgressDialog(t("update.downloading_dialog", version=release.version), "", 0, 100, parent)
    dialog.setWindowTitle("Karakal")
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setValue(0)
    thread = InstallerCopyThread(release)
    result: dict[str, str] = {}
    loop = QEventLoop()
    thread.progressed.connect(dialog.setValue)
    thread.copied.connect(lambda path, _version: result.update(path=path))
    thread.failed.connect(lambda message: result.update(error=message))
    thread.finished.connect(loop.quit)
    thread.start()
    loop.exec()
    dialog.close()
    if "error" in result:
        QMessageBox.warning(parent, "Karakal", t("update.failed", error=result["error"]))
        return None
    return result.get("path")


def check_update_now(timeout_ms: int = STARTUP_CHECK_TIMEOUT_MS) -> UpdateInfo | None:
    """Look into the update folder, waiting at most ``timeout_ms``; None when nothing newer."""

    channel = load_karakal_update_channel()
    manifest = karakal_channel_manifest_location(channel, manual=False)
    if not manifest:
        return None
    thread = UpdateCheckThread(
        manifest_url=manifest, channel=channel, timeout_seconds=KARAKAL_UPDATE_CHECK_TIMEOUT_SECONDS
    )
    result: list[object] = []
    loop = QEventLoop()
    thread.checked.connect(result.append)
    thread.finished.connect(loop.quit)
    QTimer.singleShot(int(timeout_ms), loop.quit)
    thread.start()
    loop.exec()
    _keep_until_finished(thread)
    info = result[0] if result else None
    if not isinstance(info, UpdateInfo):
        return None
    save_karakal_last_update_check()
    return info if is_newer_version(info.version, current_version()) else None


def startup_translator() -> Callable[..., str]:
    """Texts in the language the main window will use (settings, tester builds default to Russian)."""

    from ..core.features import tester_build
    from ..infra.services import KarakalSettingsService, default_settings
    from ..ui.i18n import Translator

    language = KarakalSettingsService(default_settings()).load_language()
    if tester_build() and not language:
        language = "ru"
    return Translator(language).tr


def offer_update_at_start(t: Callable[..., str] | None = None) -> bool:
    """Ask before the main window opens. True: the installer runs and the program must exit."""

    try:
        t = t or startup_translator()
        info = check_update_now()
    except Exception:  # noqa: BLE001 - a broken update folder must never stop the start
        return False
    if info is None:
        return False
    dialog = UpdateOfferDialog(None, info, t)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False
    installer = _copy_with_progress_dialog(latest_release(info), None, t)
    return bool(installer) and _launch_installer(installer, None)


class KarakalUpdateManager(QObject):
    """State of the Update button and the actions behind it and the Help menu."""

    stateChanged = pyqtSignal(str)

    def __init__(
        self,
        window: QWidget,
        t: Callable[..., str],
        *,
        is_busy: Callable[[], bool] = lambda: False,
        cancel_busy: Callable[[], None] = lambda: None,
        close_app: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(window)
        self._window = window
        self._t = t
        self._is_busy = is_busy
        self._cancel_busy = cancel_busy
        self._close_app = close_app or window.close
        self.state = STATE_IDLE
        self.info: UpdateInfo | None = None
        self.progress = 0
        self._check_thread: UpdateCheckThread | None = None
        self._copy_thread: InstallerCopyThread | None = None
        self._after_check: Callable[[UpdateInfo | None], None] | None = None
        self._busy_wait_timer: QTimer | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(UPDATE_CHECK_INTERVAL_MS)
        self._timer.timeout.connect(lambda: self.check(manual=False))

    # Checks ------------------------------------------------------------------

    def start(self, first_check_delay_ms: int = 1500) -> None:
        self._timer.start()
        QTimer.singleShot(int(first_check_delay_ms), lambda: self.check(manual=False))

    def stop(self) -> None:
        """The window closes: no more checks, running threads end on their own without callbacks."""

        self._timer.stop()
        if self._busy_wait_timer is not None:
            self._busy_wait_timer.stop()
        for thread, signals in (
            (self._check_thread, ("checked",)),
            (self._copy_thread, ("progressed", "copied", "failed")),
        ):
            if thread is None:
                continue
            for name in (*signals, "finished"):
                try:
                    getattr(thread, name).disconnect()
                except (TypeError, RuntimeError):
                    pass
            _keep_until_finished(thread)
        self._check_thread = None
        self._copy_thread = None

    def check(self, *, manual: bool, then: Callable[[UpdateInfo | None], None] | None = None) -> None:
        if self.state == STATE_DOWNLOADING:
            return
        if self._check_thread is not None:
            if manual:
                self._after_check = then or self._after_manual_check
            return
        channel = load_karakal_update_channel()
        manifest = karakal_channel_manifest_location(channel, manual=manual, window=self._window)
        if manifest is None:
            return
        if not manifest:
            if manual:
                QMessageBox.information(self._window, "Karakal", self._t("update.not_configured"))
            return
        self._after_check = then or (self._after_manual_check if manual else None)
        thread = UpdateCheckThread(
            manifest_url=manifest, channel=channel, timeout_seconds=KARAKAL_UPDATE_CHECK_TIMEOUT_SECONDS
        )
        thread.checked.connect(lambda info, m=manual: self._on_checked(info, manual=m))
        thread.finished.connect(self._clear_check_thread)
        self._check_thread = thread
        thread.start()

    def _clear_check_thread(self) -> None:
        self._check_thread = None

    def _on_checked(self, info: object, *, manual: bool) -> None:
        after, self._after_check = self._after_check, None
        if not isinstance(info, UpdateInfo):
            if manual:
                QMessageBox.warning(self._window, "Karakal", self._t("update.check_failed"))
            return
        save_karakal_last_update_check()
        self.info = info
        newer = is_newer_version(info.version, current_version())
        if self.state != STATE_DOWNLOADING:
            self._set_state(STATE_AVAILABLE if newer else STATE_IDLE)
        if after is not None:
            after(info if newer else None)

    def _after_manual_check(self, info: UpdateInfo | None) -> None:
        if info is None:
            QMessageBox.information(
                self._window, "Karakal", self._t("update.up_to_date", version=current_version())
            )
            return
        dialog = UpdateOfferDialog(self._window, info, self._t)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.install(latest_release(info))

    # Button ------------------------------------------------------------------

    def on_button_clicked(self) -> None:
        if self.state == STATE_AVAILABLE and self.info is not None:
            self.install(latest_release(self.info))
        elif self.state == STATE_IDLE:
            self.check(manual=True)

    def button_text(self) -> str:
        if self.state == STATE_DOWNLOADING:
            return self._t("update.button.downloading", percent=self.progress)
        if self.state == STATE_AVAILABLE and self.info is not None:
            return self._t("update.button.available", version=self.info.version)
        return self._t("update.button")

    def button_tooltip(self) -> str:
        if self.state == STATE_AVAILABLE and self.info is not None:
            return self._t("update.button.available_tooltip", version=self.info.version, installed=current_version())
        if self.state == STATE_DOWNLOADING:
            return ""
        return self._t("update.button.idle_tooltip", version=current_version())

    def _set_state(self, state: str) -> None:
        self.state = state
        self.stateChanged.emit(state)

    # Install -----------------------------------------------------------------

    def install(self, release: ReleaseInfo) -> None:
        if self.state == STATE_DOWNLOADING:
            return
        if not str(release.download_url or "").strip():
            QMessageBox.warning(self._window, "Karakal", self._t("update.no_installer"))
            return
        if self._is_busy():
            box = QMessageBox(self._window)
            box.setWindowTitle("Karakal")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(self._t("update.busy.text"))
            after = box.addButton(self._t("update.busy.after"), QMessageBox.ButtonRole.AcceptRole)
            cancel_run = box.addButton(self._t("update.busy.cancel_run"), QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(self._t("update.busy.cancel"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is after:
                self._wait_until_idle(release)
            elif box.clickedButton() is cancel_run:
                self._cancel_busy()
                self._wait_until_idle(release)
            return
        answer = QMessageBox.question(
            self._window,
            "Karakal",
            self._t("update.confirm", version=release.version),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._copy(release)

    def _wait_until_idle(self, release: ReleaseInfo) -> None:
        """Install as soon as the running analysis ends."""

        if self._busy_wait_timer is not None:
            self._busy_wait_timer.stop()
        timer = QTimer(self)
        timer.setInterval(1000)

        def poll() -> None:
            if self._is_busy():
                return
            timer.stop()
            self._busy_wait_timer = None
            self._copy(release)

        timer.timeout.connect(poll)
        self._busy_wait_timer = timer
        timer.start()

    def _copy(self, release: ReleaseInfo) -> None:
        self.progress = 0
        self._set_state(STATE_DOWNLOADING)
        thread = InstallerCopyThread(release)
        thread.progressed.connect(self._on_progress)
        thread.copied.connect(self._on_copied)
        thread.failed.connect(self._on_copy_failed)
        thread.finished.connect(self._clear_copy_thread)
        self._copy_thread = thread
        thread.start()

    def _clear_copy_thread(self) -> None:
        self._copy_thread = None

    def _on_progress(self, percent: int) -> None:
        self.progress = int(percent)
        self.stateChanged.emit(self.state)

    def _on_copied(self, installer_path: str, _version: str) -> None:
        if not _launch_installer(installer_path, self._window):
            self._set_state(STATE_AVAILABLE if self.info is not None else STATE_IDLE)
            return
        self.stop()
        # The installer waits for the program to close and starts it again when done.
        self._close_app()

    def _on_copy_failed(self, message: str) -> None:
        newer = self.info is not None and is_newer_version(self.info.version, current_version())
        self._set_state(STATE_AVAILABLE if newer else STATE_IDLE)
        QMessageBox.warning(self._window, "Karakal", self._t("update.failed", error=message))

    # Help menu ---------------------------------------------------------------

    def choose_version(self) -> None:
        """Install any version of the channel, older ones too (rollback)."""

        self.check(manual=False, then=lambda _newer: self._show_versions())

    def _show_versions(self) -> None:
        info = self.info
        if info is None or not info.releases:
            QMessageBox.information(self._window, "Karakal", self._t("update.versions.none"))
            return
        installed = current_version()
        releases = tuple(info.releases)
        labels = [
            self._t("update.versions.installed", version=release.version)
            if release.version == installed
            else release.version
            for release in releases
        ]
        label, accepted = QInputDialog.getItem(
            self._window, self._t("update.versions.title"), self._t("update.versions.label"), labels, 0, False
        )
        if not accepted:
            return
        release = releases[labels.index(label)]
        if release.version == installed:
            QMessageBox.information(self._window, "Karakal", self._t("update.versions.same"))
            return
        self.install(release)

