"""Reusable pytest-qt harness that drives the real Kraken Hub UI (offscreen).

Foundation for the inventory-driven GUI campaign (plan 5.2) and the
state-aware explorer (plan 6.1).

Composition
-----------
``launch_hub`` builds the same desktop composition as
``kraken_hub.composition.app.run_manager_gui`` — ``build_workspace_service``
+ ``ProjectManagerShell`` + ``DesktopController`` + the standard side panels —
but hermetically:

- ``QSettings`` is isolated into ``<data_dir>/settings`` via the product's
  own ``kraken_core.settings.configure_settings_directory`` (restored on exit);
- remote-server environment variables (``KRAKEN_SERVER_URL`` etc.) are
  removed for the duration of the launch, so the service is the local
  composition;
- Windows Credential Manager writes (``windows_credentials.save_credentials``)
  are neutralised so a test login never persists into the user's vault;
- the detached ``kraken-agent`` subprocess is replaced by a no-op runtime,
  the same isolation the existing local-project GUI tests use
  (``tests/test_hub_project_tabs.py`` patches ``manager_app.LocalAgentRuntime``);
  without it, opening a local project spawns a real agent process that
  survives the test and locks files in the temporary data dir.

The caller supplies the data dir; use
``tests.test_hub_embedded_service_parts.kraken_tempdir.kraken_tempdir`` so
cleanup is reserved-DOS-name safe (defect D10).

Synchronisation
---------------
All waiting goes through :func:`wait_until` (Qt event processing until a
predicate holds — no ``time.sleep``, no arbitrary delays).

UI texts
--------
The Hub UI is Russian; the menu/button texts used here are the real ones
from ``kraken_manager.presentation.qt.shell`` / ``pages``
("Управление", "Слои…", "Создать проект", ...).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QWidget,
)

import kraken_core.settings as kraken_settings
from kraken_core.qt import configure_application_identity
from kraken_core.theme import THEME_SYSTEM, apply_app_theme
from kraken_hub.composition.catalog import build_workspace_service
from kraken_hub.infrastructure.credentials import windows as windows_credentials
from kraken_hub.presentation.qt import manager_app
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
    _configure_administration_page,
    _login,
    _my_work_panel,
    _performer_panel,
    _plugin_panel,
    _statistics_panel,
)
from kraken_manager.presentation.qt import ProjectManagerShell

__all__ = [
    "HubHandle",
    "active_hub",
    "catalog_project_names",
    "capture_failure",
    "click_button",
    "click_menu",
    "create_project",
    "dialog_buttons",
    "dialog_text",
    "find_button",
    "harness_capture",
    "journal_events",
    "launch_hub",
    "modal_action",
    "modal_dialog",
    "open_project",
    "wait_until",
]

# ---------------------------------------------------------------------------
# Log ring buffer (tail captured by capture_failure)
# ---------------------------------------------------------------------------

_LOG_RING: deque[str] = deque(maxlen=1000)


class _LogRingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _LOG_RING.append(self.format(record))
        except Exception:
            pass


_ring_handler = _LogRingHandler()
_ring_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
# Root logger: the product packages (kraken_hub, kraken_manager, kraken_core)
# are siblings, not children of one another, so only the root sees all of them.
logging.getLogger().addHandler(_ring_handler)

# ---------------------------------------------------------------------------
# Active-hub registry (used by the failure-capture fixture)
# ---------------------------------------------------------------------------

_ACTIVE_HUBS: list["HubHandle"] = []


def active_hub() -> "HubHandle | None":
    """Return the most recently launched hub that has not been closed yet."""
    return _ACTIVE_HUBS[-1] if _ACTIVE_HUBS else None


# ---------------------------------------------------------------------------
# Observable waiting
# ---------------------------------------------------------------------------


def wait_until(predicate: Callable[[], object], timeout: float = 15.0, interval: float = 0.05, what: str = ""):
    """Process Qt events until ``predicate`` is truthy; return its value.

    No ``time.sleep``: ``QTest.qWait`` pumps the event loop, so background
    workers, timers and modal dialogs keep making progress while we wait.
    Predicate exceptions are treated as "not yet" (state may be transient).
    The predicate's truthy result is returned (``None`` when a boolean-style
    predicate was used), so callers can wait for a concrete object such as a
    menu action instead of re-locating it afterwards.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            value = predicate()
        except Exception:
            value = None
        if value:
            return value
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"wait_until timed out after {timeout:.1f}s: {what or str(predicate)}")
        QTest.qWait(max(1, min(int(interval * 1000), int(remaining * 1000) or 1)))


def _active_modal() -> QWidget | None:
    dialog = QApplication.activeModalWidget()
    if isinstance(dialog, (QDialog, QMessageBox)):
        return dialog
    return None


# ---------------------------------------------------------------------------
# Menu and button driving (real Russian texts from the shell)
# ---------------------------------------------------------------------------


def _normalize_text(text: object) -> str:
    value = str(text or "").strip()
    for suffix in ("\u2026", "...", "\u2025"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].rstrip()
    return re.sub(r"\s+", " ", value)


def _menu_action(container, name: str):
    wanted = _normalize_text(name)
    for action in container.actions():
        if action.isSeparator():
            continue
        if _normalize_text(action.text()) == wanted:
            return action
    return None


def _enabled_menu_action(container, name: str):
    action = _menu_action(container, name)
    if action is not None and action.isEnabled():
        return action
    return None


def click_menu(window, path: str, *, timeout: float = 10.0):
    """Navigate a menu path like ``"Действия → Загрузить проверенные файлы…"``.

    Supports submenus (each segment except the last must be a submenu title)
    and waits for the final action to be enabled before triggering it.
    Returns the triggered ``QAction``.
    """
    parts = [part.strip() for part in str(path).split("\u2192") if part.strip()]
    if not parts:
        raise ValueError("empty menu path")
    container = window.menuBar()
    for part in parts[:-1]:
        action = wait_until(
            lambda: _menu_action(container, part),
            timeout=timeout,
            what=f"menu {part!r} to appear",
        )
        submenu = action.menu() if action is not None else None
        if submenu is None:
            raise LookupError(f"menu entry {part!r} is not a submenu")
        container = submenu
    target = wait_until(
        lambda: _enabled_menu_action(container, parts[-1]),
        timeout=timeout,
        what=f"menu action {parts[-1]!r} to be enabled",
    )
    target.trigger()
    return target


def find_button(parent, text: str):
    """Return the first ``QPushButton`` with the given text under ``parent`` (or None)."""
    wanted = _normalize_text(text)
    for button in parent.findChildren(QPushButton):
        if _normalize_text(button.text()) == wanted:
            return button
    return None


def click_button(parent, text: str, *, timeout: float = 10.0) -> QPushButton:
    """Wait for the button to exist and be enabled, then send a real mouse click."""
    found = {}

    def ready():
        button = find_button(parent, text)
        if button is None or not button.isEnabled():
            return False
        found["button"] = button
        return True

    wait_until(ready, timeout=timeout, what=f"button {text!r} to be enabled")
    QTest.mouseClick(found["button"], Qt.MouseButton.LeftButton)
    return found["button"]


# ---------------------------------------------------------------------------
# Modal dialogs
# ---------------------------------------------------------------------------


def modal_action(trigger: Callable[[], None], fill: Callable[[QDialog], None] | None = None, *, timeout: float = 60.0) -> list[str]:
    """Run ``trigger`` (which blocks in a ``dialog.exec()`` loop) and drive the modal.

    A 20 ms timer observes ``QApplication.activeModalWidget()`` inside the
    nested event loop; the first dialog instance is passed to ``fill`` (the
    live-acceptance ``modal()`` pattern from ``tests/test_live_ui_workflow.py``).
    ``QMessageBox`` popups are ignored here — ``launch_hub`` runs a global
    recorder that auto-dismisses and reports them.
    Returns the visited dialog titles; raises ``TimeoutError``/the fill error
    on failure.
    """
    visited: list[str] = []
    filled: set[int] = set()
    errors: list[BaseException] = []
    timer = QTimer()
    timer.setInterval(20)
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.setInterval(int(timeout * 1000))

    def timed_out() -> None:
        errors.append(TimeoutError(f"modal_action timed out after {timeout:.0f}s"))
        dialog = _active_modal()
        if dialog is not None and not isinstance(dialog, QMessageBox):
            dialog.reject()

    def poll() -> None:
        dialog = _active_modal()
        if dialog is None or isinstance(dialog, QMessageBox) or not isinstance(dialog, QDialog):
            return
        if not visited or visited[-1] != dialog.windowTitle():
            visited.append(dialog.windowTitle())
        if fill is not None and id(dialog) not in filled:
            filled.add(id(dialog))
            try:
                fill(dialog)
            except Exception as exc:
                errors.append(exc)
                dialog.reject()

    timer.timeout.connect(poll)
    deadline.timeout.connect(timed_out)
    timer.start()
    deadline.start()
    try:
        trigger()
    finally:
        timer.stop()
        deadline.stop()
    if not visited:
        raise AssertionError("The requested UI action did not open its dialog")
    if errors:
        raise errors[0]
    return visited


def modal_dialog(window, timeout: float = 15.0, what: str = "a modal dialog to appear") -> QWidget:
    """Wait for and return the currently active modal (QDialog or QMessageBox).

    Use this for modals that appear without blocking the caller (non-modal
    flows, or from inside a ``modal_action`` fill callback).
    """
    wait_until(lambda: _active_modal() is not None, timeout=timeout, what=what)
    dialog = _active_modal()
    assert dialog is not None
    return dialog


def dialog_text(dialog) -> str:
    """Main text of a modal: QMessageBox text, otherwise the window title."""
    if isinstance(dialog, QMessageBox):
        return dialog.text()
    return dialog.windowTitle()


def dialog_buttons(dialog) -> list[QPushButton]:
    """Buttons of a modal: QMessageBox buttons, button-box buttons, else all."""
    if isinstance(dialog, QMessageBox):
        return list(dialog.buttons())
    box = dialog.findChild(QDialogButtonBox)
    if box is not None:
        return list(box.buttons())
    return dialog.findChildren(QPushButton)


# ---------------------------------------------------------------------------
# Catalog / project helpers
# ---------------------------------------------------------------------------


@dataclass
class HubHandle:
    """One launched Hub: the real main window plus the objects behind it."""

    window: ProjectManagerShell
    controller: DesktopController
    service: object  # DualCatalogService (local composition)
    session: object  # DesktopSession (UserSession)
    data_dir: Path
    account: tuple[str, str]
    messages: list[tuple[str, str]] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    evidence: list[tuple[Path, Path, Path]] = field(default_factory=list)


def catalog_project_names(handle: HubHandle) -> list[str]:
    """Names of the projects currently shown on the catalog page."""
    return [item.name for item in handle.controller.catalog_page.project_model.items()]


def _catalog_project(handle: HubHandle, name: str):
    return next((item for item in handle.service.list_projects() if item.name == name), None)


def create_project(handle: HubHandle, name: str, *, width: int = 2, height: int = 1, timeout: float = 30.0):
    """Create a project through the real «Создать проект» dialog (default 2×1, local storage).

    The dialog's Ok button is the real trigger; the storage roots come from
    the isolated QSettings pre-seeded by ``launch_hub``, so no folder picker
    appears. Returns the created ``Project`` (existing projects are returned
    as-is).
    """
    existing = _catalog_project(handle, name)
    if existing is not None:
        return existing

    def fill(dialog: QDialog) -> None:
        name_edit = dialog.findChild(QLineEdit, "projectName")
        if name_edit is None:
            raise AssertionError("create-project dialog: name field not found")
        name_edit.setText(name)
        profile_box = next(
            (combo for combo in dialog.findChildren(QComboBox) if combo.findText("Локальный") >= 0),
            None,
        )
        if profile_box is None:
            raise AssertionError("create-project dialog: local storage profile not found")
        profile_box.setCurrentIndex(profile_box.findText("Локальный"))
        dimensions = dialog.findChild(QWidget, "gridDimensionsWidget")
        if dimensions is None or not hasattr(dimensions, "set_dimensions"):
            raise AssertionError("create-project dialog: dimensions widget not found")
        dimensions.set_dimensions(width, height)
        click_button(dialog, "Создать проект")

    modal_action(
        lambda: click_button(handle.controller.catalog_page, "Создать проект"),
        fill,
        timeout=timeout,
    )
    wait_until(
        lambda: name in catalog_project_names(handle),
        timeout=timeout,
        what=f"project {name!r} to appear in the catalog",
    )
    created = _catalog_project(handle, name)
    assert created is not None
    return created


def open_project(handle: HubHandle, name: str, *, timeout: float = 30.0):
    """Open a catalog project with a real double-click and wait for its workspace."""
    catalog = handle.controller.catalog_page
    wait_until(
        lambda: name in catalog_project_names(handle),
        timeout=timeout,
        what=f"project {name!r} to appear in the catalog",
    )
    model, view = catalog.project_model, catalog.project_list
    row = next(i for i in range(model.rowCount()) if (model.item_at(i) is not None and model.item_at(i).name == name))
    index = model.index(row, 0)
    view.setCurrentIndex(index)
    view.scrollTo(index)
    QApplication.processEvents()
    position = view.visualRect(index).center()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    project = _catalog_project(handle, name)
    wait_until(
        lambda: handle.controller._workspace is not None and str(handle.controller._project_id) == str(project.id),
        timeout=timeout,
        what=f"project {name!r} to open its workspace",
    )
    return project


# ---------------------------------------------------------------------------
# Journal events (observable persistence)
# ---------------------------------------------------------------------------


def journal_events(target, project_id=None, *, since: int = 0) -> list:
    """New project-journal events for assertions on observable persistence.

    ``target`` may be a ``HubHandle``, a main window (shell) or a service.
    For handle/window the current project is used unless ``project_id`` is
    given; a bare service requires ``project_id``.
    """
    service = target
    if hasattr(target, "service") and hasattr(target, "controller"):  # HubHandle
        service = target.service
        project_id = project_id or target.controller._project_id
    elif hasattr(target, "_desktop_controller"):  # ProjectManagerShell
        controller = target._desktop_controller
        service = controller.service
        project_id = project_id or controller._project_id
    if project_id is None:
        raise ValueError("journal_events needs a project_id when called with a bare service")
    return list(service.history(project_id))[since:]


# ---------------------------------------------------------------------------
# Failure capture
# ---------------------------------------------------------------------------

_MAX_TREE_DEPTH = 15
_MAX_TREE_NODES = 5000


def _sanitize_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label)).strip("-.")
    return safe or "capture"


def _widget_tree_node(widget: QWidget, depth: int, counter: list[int], truncated: list[bool]) -> dict:
    node: dict = {
        "class": widget.metaObject().className() if widget.metaObject() is not None else type(widget).__name__,
        "objectName": widget.objectName(),
    }
    text = getattr(widget, "text", None)
    if callable(text):
        try:
            value = str(widget.text())
        except Exception:
            value = ""
        if value:
            node["text"] = value[:200]
    parent = widget.parentWidget()
    node["visible"] = bool(widget.isVisibleTo(parent if parent is not None else widget))
    node["enabled"] = bool(widget.isEnabled())
    counter[0] += 1
    if counter[0] > _MAX_TREE_NODES or depth >= _MAX_TREE_DEPTH:
        truncated[0] = True
        return node
    children = []
    for child in widget.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
        children.append(_widget_tree_node(child, depth + 1, counter, truncated))
        if counter[0] > _MAX_TREE_NODES:
            truncated[0] = True
            break
    if children:
        node["children"] = children
    return node


def capture_failure(window, evidence_dir: Path | str, label: str = "failure") -> tuple[Path, Path, Path]:
    """Write screenshot + widget-tree JSON + log tail for a failed hub state.

    Returns ``(png, widgets_json, log_txt)``.
    """
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_label(label)
    png = evidence / f"{safe}.png"
    tree_path = evidence / f"{safe}-widgets.json"
    log_path = evidence / f"{safe}-log.txt"
    window.grab().save(str(png))
    counter = [0]
    truncated = [False]
    payload = {
        "label": safe,
        "window_title": str(window.windowTitle()),
        "truncated": False,
        "tree": _widget_tree_node(window, 0, counter, truncated),
    }
    payload["truncated"] = truncated[0]
    tree_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tail = "\n".join(_LOG_RING)
    log_path.write_text(tail + ("\n" if tail else ""), encoding="utf-8")
    return (png, tree_path, log_path)


class _FailureCapturePlugin:
    """Captures the active hub when the owning test's call phase fails."""

    def __init__(self, item, capture: Callable[..., tuple[Path, Path, Path]]) -> None:
        self._item = item
        self._capture = capture

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        if item is not self._item or call.when != "call":
            return
        report = outcome.get_result()
        if not report.failed:
            return
        try:
            self._capture(label=self._item.name)
        except Exception:
            logging.getLogger("kraken_hub.tests").exception("qt-harness failure capture failed")


@pytest.fixture
def harness_capture(tmp_path_factory, request):
    """Failure-capture hook for campaign tests.

    Yields ``{"evidence_dir": Path, "paths": None, "capture": callable}``.
    When the test's call phase fails, the currently active hub (the last
    ``launch_hub`` not yet closed) is captured automatically into
    ``evidence_dir`` and the paths are stored in ``state["paths"]``.
    ``state["capture"](label)`` performs the same capture on demand
    (used by the harness self-test for a controlled failure path).
    """
    evidence_dir = tmp_path_factory.mktemp("qt-harness")
    state: dict = {"evidence_dir": evidence_dir, "paths": None}

    def capture(label: str = "failure") -> tuple[Path, Path, Path]:
        handle = active_hub()
        if handle is None:
            raise RuntimeError("harness_capture: no hub launched by launch_hub")
        state["paths"] = capture_failure(handle.window, evidence_dir, label=label)
        return state["paths"]

    state["capture"] = capture
    plugin = _FailureCapturePlugin(request.node, capture)
    plugin_name = f"qt-harness-capture-{uuid4().hex}"
    request.config.pluginmanager.register(plugin, plugin_name)
    try:
        yield state
    finally:
        request.config.pluginmanager.unregister(plugin)


# ---------------------------------------------------------------------------
# Login (real bootstrap/login dialogs) and message recorder
# ---------------------------------------------------------------------------


def _sign_in(service, account: tuple[str, str]):
    """Drive the real HUB-BOOTSTRAP-ACCOUNT / HUB-LOGIN dialogs to a session."""
    username, password = account
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = _active_modal()
        if dialog is None or not isinstance(dialog, QDialog) or isinstance(dialog, QMessageBox):
            return
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if tabs is not None:
            tabs.setCurrentIndex(2)  # локальная вкладка — путь существующих тестов
        fields = {
            "initialAccountUsername": username,
            "initialAccountDisplayName": username,
            "initialAccountPassword": password,
            "initialAccountPasswordConfirmation": password,
            "loginUsername": username,
            "loginPassword": password,
        }
        for object_name, value in fields.items():
            field = dialog.findChild(QLineEdit, object_name)
            if field is not None and field.text() != value:
                field.setText(value)
        box = dialog.findChild(QDialogButtonBox)
        if box is None:
            return
        ok = box.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None and ok.isEnabled():
            QTest.mouseClick(ok, Qt.MouseButton.LeftButton)

    timer.timeout.connect(poll)
    timer.start()
    try:
        session = _login(None, service)
    finally:
        timer.stop()
    if session is None:
        raise RuntimeError(f"Hub login was not completed for account {username!r}")
    return session


def _message_recorder(messages: list, failures: list) -> QTimer:
    """Auto-dismiss QMessageBox popups, recording them (warnings = failures)."""
    timer = QTimer()
    timer.setInterval(25)

    def close_messages() -> None:
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QMessageBox):
            return
        messages.append((dialog.windowTitle(), dialog.text()))
        if dialog.icon() in (QMessageBox.Icon.Warning, QMessageBox.Icon.Critical):
            failures.append((dialog.windowTitle(), dialog.text()))
        button = (
            dialog.button(QMessageBox.StandardButton.Yes)
            or dialog.button(QMessageBox.StandardButton.Ok)
            or next((b for b in dialog.buttons() if b.text() in ("Позже",)), None)
        )
        if button is not None:
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    timer.timeout.connect(close_messages)
    return timer


# ---------------------------------------------------------------------------
# No-op agent runtime (same isolation as tests/test_hub_project_tabs.py)
# ---------------------------------------------------------------------------


class _HarnessAgentRuntime:
    """Stand-in for LocalAgentRuntime: no detached agent subprocess in tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.process = None
        self.base_url = ""
        self.token = ""

    def ensure_started(self) -> frozenset[str]:
        return frozenset()

    def detach(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

_ISOLATED_ENV = ("KRAKEN_SERVER_URL", "KRAKEN_GITLAB_TOKEN", "KRAKEN_SERVER_TOKEN", "KRAKEN_UPDATE_URL")


@contextmanager
def launch_hub(
    data_dir: Path | str,
    *,
    open_project: str | None = None,
    account: tuple[str, str] | None = None,
    evidence_dir: Path | str | None = None,
) -> Iterator[HubHandle]:
    """Launch the real Hub (local composition) offscreen and yield a ``HubHandle``.

    - ``data_dir``: fresh directory for the local store (use ``kraken_tempdir``
      for the caller-managed lifetime); workspace roots live inside it and are
      pre-registered in the isolated QSettings so local project creation needs
      no folder pickers.
    - ``account``: ``(username, password)``; the real bootstrap dialog creates
      the first account on a fresh store, the real login dialog signs in on an
      existing one. Defaults to ``("harness", "")``.
    - ``open_project``: catalog project NAME; created via the real UI with
      default 2×1 dimensions when it does not exist yet, then opened.
    - ``evidence_dir``: when set, a hub failure in the test body captures
      screenshot + widget tree + log tail there before re-raising.

    On exit the window is closed through the real close guard (agent detach),
    isolated state (env, QSettings dir, credential writes) is restored, and
    any Warning/Critical message boxes shown during the run are reported.
    """
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("launch_hub requires a running QApplication (use the qapp fixture)")
    data_dir = Path(data_dir).resolve()
    username, password = account or ("harness", "")

    env_saved = {name: os.environ.pop(name, None) for name in _ISOLATED_ENV}
    settings_saved = kraken_settings._directory
    kraken_settings.configure_settings_directory(data_dir / "settings")
    credentials_saved = windows_credentials.save_credentials
    windows_credentials.save_credentials = lambda *_args, **_kwargs: True  # noqa: ARG005 - neutralise vault writes
    runtime_saved = manager_app.LocalAgentRuntime
    manager_app.LocalAgentRuntime = _HarnessAgentRuntime

    messages: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    message_timer = _message_recorder(messages, failures)
    handle: HubHandle | None = None
    body_error: tuple | None = None
    cleanup_errors: list[BaseException] = []

    try:
        service = build_workspace_service(data_dir=data_dir)
        if service.remote is not None:
            raise RuntimeError("launch_hub must produce the local composition (no remote server)")
        settings = kraken_settings.QSettings("Kraken", "KrakenHub")
        settings.setValue("workspace/source-root", str(data_dir / "workspace-source"))
        settings.setValue("workspace/derived-root", str(data_dir / "workspace-derived"))

        message_timer.start()
        session = _sign_in(service, (username, password))

        configure_application_identity(app, app_id="Kraken.ProjectManager", icon_name="kraken")
        apply_app_theme(str(kraken_settings.QSettings("Kraken", "KrakenHub").value("appearance/theme", THEME_SYSTEM)), app)

        shell = ProjectManagerShell()
        shell.set_session_summary(f"{session.principal.display_name}\nЛокальная сессия")
        controller = DesktopController(
            shell,
            service,
            session,
            thumbnail_store_uri="memory://",
            plugin_items=[],
        )
        shell._desktop_controller = controller  # keep Qt slots alive (run_manager_gui)
        controller._agent_timer.stop()

        # Standard side panels, exactly as run_manager_gui composes them.
        shell.page("plugins").set_content(_plugin_panel([]))
        shell.page("performers").set_content(_performer_panel(service))
        shell.page("my_work").set_content(_my_work_panel(service, session, controller))
        shell.page("statistics").set_content(
            _statistics_panel(service, session.principal, background_workers=controller._background_workers, async_reads=True)
        )
        _configure_administration_page(shell, service, session)

        shell.show()
        handle = HubHandle(
            window=shell,
            controller=controller,
            service=service,
            session=session,
            data_dir=data_dir,
            account=(username, password),
            messages=messages,
            failures=failures,
        )
        _ACTIVE_HUBS.append(handle)
        if open_project is not None:
            create_project(handle, open_project)
            open_project(handle, open_project)
        yield handle
    except BaseException:
        body_error = sys.exc_info()
        if evidence_dir is not None and handle is not None:
            try:
                handle.evidence = list(capture_failure(handle.window, Path(evidence_dir), label="launch-failure"))
            except Exception:
                logging.getLogger("kraken_hub.tests").exception("qt-harness launch failure capture failed")
    finally:
        if handle is not None and handle in _ACTIVE_HUBS:
            _ACTIVE_HUBS.remove(handle)
        message_timer.stop()
        if handle is not None:
            try:
                wait_until(
                    lambda: not handle.controller._background_workers,
                    timeout=30.0,
                    what="background workers to finish",
                )
            except TimeoutError as exc:
                cleanup_errors.append(exc)
            try:
                handle.controller._sync_timer.stop()
                handle.controller._agent_timer.stop()
                layer_dialog = handle.controller._layer_dialog
                if layer_dialog is not None and layer_dialog.isVisible():
                    layer_dialog.close()
                handle.window.close()  # close guard: _allow_close -> agent detach
            except Exception as exc:
                cleanup_errors.append(exc)
        for name, value in env_saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        kraken_settings._directory = settings_saved
        windows_credentials.save_credentials = credentials_saved
        manager_app.LocalAgentRuntime = runtime_saved

    if body_error is not None:
        raise body_error[1].with_traceback(body_error[2])
    if cleanup_errors:
        raise RuntimeError("qt-harness teardown problems: " + "; ".join(str(exc) for exc in cleanup_errors))
    if handle is not None and handle.failures:
        raise RuntimeError(f"Hub showed error dialogs during the test: {handle.failures}")
