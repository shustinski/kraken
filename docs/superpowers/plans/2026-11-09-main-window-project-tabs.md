# Kraken Hub Main Window: Project Tabs + Icon Rail — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open projects as persistent top tabs (each tab keeps its live workspace state), and move the six sections from the wide left text list to a compact icon rail.

**Architecture:** `ProjectManagerShell` gains a top `QTabBar#projectTabs` that holds one persistent `ProjectWorkspacePage` per open project (page key `project:<id>`); switching tabs is a pure `QStackedWidget` swap. `DesktopController` moves from single-active-project fields (`_project_id`/`_workspace` as "the only project") to a per-project registry `_project_workspaces` plus active pointers driven by shell tab signals. The sidebar `QListWidget` is replaced by a 48 px icon rail; session/sync labels move to the status bar.

**Tech Stack:** Python 3.14, PyQt6, pytest (+ pytest-qt for `qtbot`), uv. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-09-main-window-project-tabs-design.md`

## Global Constraints

- PyQt6 only; no new packages.
- `ProjectManagerShell` (kraken_manager) must stay free of composition logic: no imports from `kraken_hub` or from `ProjectWorkspaceService`; it emits signals and owns widgets.
- QSS changes must land in **both** `src/kraken_core/resources/styles/dark_modern.qss` (dark) and `src/kraken_core/resources/styles/style.qss` (light) — theme files are selected by `kraken_core/theme.py`.
- Icons: no icon resources exist except the Kraken logo (`kraken.ico`/`kraken.png`). All new glyphs are hand-drawn `QPixmap`s, following the existing painted-plus recipe in `src/kraken_manager/presentation/qt/pages.py` (`addLayerButton`). Draw rail icons and the tab close glyph in mid-tone `#647d9c` (readable in both themes).
- Section page keys (`projects`, `my_work`, `statistics`, `performers`, `plugins`, `administration`) and the `set_content` API must stay unchanged — `run_manager_gui.py` depends on them.
- Do not change: Administration section behavior, catalog page content/actions, layer tabs, matrix widget internals, menu bar contents.
- UI strings in Russian, matching the spec.
- Controller-level integration tests set `QT_QPA_PLATFORM=offscreen` at module import (pattern: `tests/test_desktop_runtime_smoke.py`) and stub `manager_app.LocalAgentRuntime` (pattern: same file).
- One commit per task, at the end of the task.

## Review Focus

1. **Remote project with flapping/offline server while tabs are open** — closing/switching tabs must not raise when the remote client is mid-reconnect; `unsubscribe_project` is only called when the service exposes it (`callable` guard), exactly like today's `open_project`. → pinned in Task 2: `test_closing_project_tab_tolerates_service_without_unsubscribe`.
2. **Closing the last project tab** — must never crash on the empty `QTabBar` (currentIndex -1); when the closed tab was the active page, activation falls back to the "Проекты" catalog; when a section page is showing, the section stays showing (spec: fallback applies to active-tab close). → pinned in Task 1: `test_shell_close_project_tab_falls_back_to_neighbor_then_catalog` and `test_shell_closing_all_tabs_while_section_is_shown_keeps_section`.
3. **Project deleted by another session while its tab is open** — activating the stale tab must close it gracefully (no exception, neighbor tab becomes active). → pinned in Task 2: `test_activated_tab_for_deleted_project_closes_gracefully`.
4. **Many open projects with long names** — the tab bar must scroll (QTabBar scrollers), not wrap or hide tabs; all tabs stay reachable. → pinned in Task 1: `test_shell_many_project_tabs_switch_stably`.
5. **App close with several open projects and a running background worker** — the close guard must still block until workers finish (worker set is global, but must be verified with 2+ open projects). → pinned in Task 2: `test_close_guard_blocks_with_multiple_open_projects`.

---

### Task 1: Shell — project tab bar, per-project page registry, chrome sync

**Files:**
- Modify: `src/kraken_manager/presentation/qt/shell.py`
- Test: `tests/test_manager_qt_shell.py` (add new tests; all existing tests in the file must keep passing)

**Interfaces:**
- Consumes: existing `register_page(key, label, page, navigation=...)`, `show_page(key)`, `current_page_key()`, menu actions (`layers_action`, `actions_menu`, `frame_properties_action`, `self._visual_menus`).
- Produces (used by Tasks 2–3):
  - `open_project_tab(project_id: str, name: str, page: QWidget) -> QWidget` — registers a new project page under key `project:<id>` or returns the existing page; activates its tab either way.
  - `close_project_tab(project_id: str) -> None` — removes page + tab; if it was active, activates the right neighbor (else left); if no tabs remain, `show_page("projects")`.
  - `rename_project_tab(project_id: str, name: str) -> None`
  - `current_project_id() -> str | None` — id of the active project tab, `None` when a section page is shown.
  - `project_tab_activated = pyqtSignal(object)` — project id (str) on project-tab activation, `None` when a section page is shown (emitted by both `show_page` and tab changes).
  - `project_tab_close_requested = pyqtSignal(str)` — emitted by the tab close button; the page/tab stay until the controller calls `close_project_tab`.
  - `addProjectRequested = pyqtSignal()`
  - attributes: `projectTabs: QTabBar` (objectName `projectTabs`), `add_project_button: QPushButton` (objectName `addProjectButton`).
  - Selection tracking: `selectionCountChanged(int)` on a project page is tracked per project; the status bar message `f"Выбрано кадров: {count:n}"` and `frame_properties_action` (enabled iff count == 1) reflect the **active** project only; the last known count (default 0) is re-pushed on tab activation.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_manager_qt_shell.py` (needs `pyqtSignal` import from `PyQt6.QtCore`):

```python
class _SelectionPage(QWidget):
    selectionCountChanged = pyqtSignal(int)


def test_shell_project_tabs_open_and_switch_project_pages(qapp):
    shell = ProjectManagerShell()
    page_a = QWidget()
    page_b = QWidget()
    shell.open_project_tab("p-a", "Alpha", page_a)
    shell.open_project_tab("p-b", "Beta", page_b)
    assert shell.projectTabs.count() == 2
    assert shell.current_project_id() == "p-b"
    assert shell.page_stack.currentWidget() is page_b
    shell.projectTabs.setCurrentIndex(0)
    assert shell.current_project_id() == "p-a"
    assert shell.page_stack.currentWidget() is page_a
    assert shell.page("project:p-b") is page_b


def test_shell_reopening_project_returns_existing_page_without_new_tab(qapp):
    shell = ProjectManagerShell()
    page_a = QWidget()
    first = shell.open_project_tab("p-a", "Alpha", page_a)
    again = shell.open_project_tab("p-a", "Alpha (renamed)", QWidget())
    assert first is page_a
    assert again is page_a
    assert shell.projectTabs.count() == 1
    assert shell.projectTabs.tabText(0) == "Alpha"
    assert shell.current_project_id() == "p-a"


def test_shell_close_project_tab_falls_back_to_neighbor_then_catalog(qapp):
    shell = ProjectManagerShell()
    for pid, name in (("p-a", "Alpha"), ("p-b", "Beta"), ("p-c", "Gamma")):
        shell.open_project_tab(pid, name, QWidget())
    shell.close_project_tab("p-c")
    assert shell.current_project_id() == "p-b"  # right neighbor
    shell.close_project_tab("p-b")
    assert shell.current_project_id() == "p-a"  # left neighbor (last one left)
    shell.close_project_tab("p-a")
    assert shell.current_project_id() is None
    assert shell.current_page_key() == "projects"  # no tabs left -> catalog
    assert shell.page("project:p-a") is None


def test_shell_closing_all_tabs_while_section_is_shown_keeps_section(qapp):
    shell = ProjectManagerShell()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    shell.show_page("statistics")
    shell.close_project_tab("p-a")  # closed tab was not the active page
    assert shell.current_project_id() is None
    assert shell.current_page_key() == "statistics"  # section stays, no crash on empty tab bar


def test_shell_rename_project_tab_updates_label(qapp):
    shell = ProjectManagerShell()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    shell.rename_project_tab("p-a", "Alfa")
    assert shell.projectTabs.tabText(0) == "Alfa"


def test_shell_project_tab_close_button_requests_close(qapp):
    shell = ProjectManagerShell()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    requested = []
    shell.project_tab_close_requested.connect(requested.append)
    shell.projectTabs.tabCloseRequested.emit(shell.projectTabs.currentIndex())
    assert requested == ["p-a"]
    assert shell.current_project_id() == "p-a"  # page stays until the controller closes it


def test_shell_menu_enablement_follows_active_project_tab(qapp):
    shell = ProjectManagerShell()
    assert not shell.actions_menu.isEnabled()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    assert shell.actions_menu.isEnabled()
    assert shell.layers_action.isEnabled()
    shell.show_page("statistics")
    assert not shell.actions_menu.isEnabled()
    assert not shell.layers_action.isEnabled()
    shell.projectTabs.setCurrentIndex(0)
    assert shell.actions_menu.isEnabled()


def test_shell_selection_count_tracks_active_project_tab_only(qapp):
    shell = ProjectManagerShell()
    page_a = _SelectionPage()
    page_b = _SelectionPage()
    shell.open_project_tab("p-a", "Alpha", page_a)
    shell.open_project_tab("p-b", "Beta", page_b)
    page_b.selectionCountChanged.emit(3)
    assert "3" in shell.statusBar().currentMessage()
    shell.projectTabs.setCurrentIndex(0)
    assert "0" in shell.statusBar().currentMessage()  # A has no selection yet -> re-push 0
    page_b.selectionCountChanged.emit(5)
    assert "5" not in shell.statusBar().currentMessage()  # inactive project must not drive the bar
    page_a.selectionCountChanged.emit(1)
    assert "1" in shell.statusBar().currentMessage()
    assert shell.frame_properties_action.isEnabled()
    shell.projectTabs.setCurrentIndex(1)
    assert "3" in shell.statusBar().currentMessage()  # B's last count re-pushed


def test_shell_add_project_button_shows_catalog_and_emits_request(qapp):
    shell = ProjectManagerShell()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    requests = []
    shell.addProjectRequested.connect(lambda: requests.append(True))
    shell.add_project_button.click()
    assert requests == [True]
    assert shell.current_page_key() == "projects"
    assert shell.current_project_id() is None


def test_shell_many_project_tabs_switch_stably(qapp):
    shell = ProjectManagerShell()
    pages = [
        shell.open_project_tab(f"p-{i}", "Проект с достаточно длинным названием " + "x" * 24, QWidget())
        for i in range(8)
    ]
    for index in range(8):
        shell.projectTabs.setCurrentIndex(index)
        assert shell.page_stack.currentWidget() is pages[index]
        assert shell.current_project_id() == f"p-{index}"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_manager_qt_shell.py -v`
Expected: the 9 new tests FAIL with `AttributeError: ... has no attribute 'open_project_tab'` (or `projectTabs`); all pre-existing tests in the file still PASS.

- [ ] **Step 3: Implement the tab bar and registry in `shell.py`**

Layout: in `__init__`, replace `root.addWidget(self.page_stack, 1)` with a content column — a `QWidget` holding a `QVBoxLayout` with (1) an `HBox` row containing `self.projectTabs` (stretch) + `self.add_project_button`, and (2) `self.page_stack` (stretch). Build `projectTabs` as: `QTabBar`, objectName `projectTabs`, `setShape(QTabBar.Shape.RoundedNorth)`, `setExpanding(False)`, `setDrawBase(True)`, `setTabsClosable(True)`, not movable, and set the close glyph via `setTabCloseButtonIcon(QIcon(close_pixmap))` where `close_pixmap` is a 12×12 transparent pixmap with two diagonal lines (2,2)-(10,10) and (10,2)-(2,10), `QPen(QColor("#647d9c"), 1.8)`, round caps. `add_project_button` copies the `addLayerButton` recipe from `pages.py` (20×20 painted plus icon, `setFixedSize(38, 38)`, objectName `addProjectButton`, tooltip "Открыть проект", inline stylesheet `QPushButton#addProjectButton { padding: 0; }`); on click it does `self.addProjectRequested.emit()` then `self.show_page("projects")`. Add the missing `PyQt6.QtGui`/`QtCore` imports (`QColor`, `QIcon`, `QPainter`, `QPen`, `QPixmap`, `QPointF`, `QSize` as needed).

New methods (exact semantics per the Interfaces block):
- `open_project_tab`: if `f"project:{project_id}"` is already in `self._pages`, `setCurrentIndex` to that tab and return the existing page. Otherwise `register_page(key, name, page, navigation=False)`, `addTab(name)` with `setTabData(index, project_id)` and `setTabToolTip(index, name)`; if `hasattr(page, "selectionCountChanged")` connect `lambda count, pid=project_id: self._on_project_selection(pid, count)`. The tab bar auto-activates the new tab, driving activation (no extra `setCurrentWidget` needed).
- `_on_project_tab_changed(index)` (connected to `projectTabs.currentChanged`): for `index < 0` emit `project_tab_activated(None)`; otherwise read `project_id` from tab data, `page_stack.setCurrentWidget(page)`, `_sync_chrome_for_current_page()`, `_push_selection_count()`, emit `project_tab_activated(project_id)`.
- `close_project_tab`: find tab index by data; if missing, return. Pop the page from `self._pages`, `page_stack.removeWidget`, `page.close()`, `page.setParent(None)`; drop its `_selection_counts` entry; `blockSignals(True)` around `removeTab`; if the closed page was the stack's current widget: if tabs remain `setCurrentIndex(min(index, remaining - 1))` (right neighbor, else left), else `show_page("projects")`.
- `rename_project_tab`: set `tabText` of the matching tab.
- `current_project_id`: from `current_page_key()` — strip the `project:` prefix, else `None`.
- `_sync_chrome_for_current_page()`: extract the existing enablement block from `show_page` (layers_action, visual menus, actions_menu, frame_properties_action) with condition `key == "workspace" or key.startswith("project:")` (Task 2 drops the `"workspace"` clause).
- `_on_project_selection(project_id, count)` / `_push_selection_count()` / `self._selection_counts: dict[str, int]` per the selection rules in Interfaces (re-push uses stored count, default 0).
- `show_page`: after the existing page switch and `pageChanged` emit, call `_sync_chrome_for_current_page()` (replacing the inline enablement block) and, for non-project keys, emit `project_tab_activated(None)`.

The old `open_project_workspace` and the `"workspace"` page must keep working unchanged in this task (Task 2 removes them).

- [ ] **Step 4: Run the shell tests to verify they pass**

Run: `uv run pytest tests/test_manager_qt_shell.py -v`
Expected: all tests PASS (new + pre-existing).

- [ ] **Step 5: Commit**

```bash
git add src/kraken_manager/presentation/qt/shell.py tests/test_manager_qt_shell.py
git commit -m "Add project tab bar with per-project pages to manager shell"
```

---

### Task 2: Controller — multi-project state, activation sync, wake retention

**Files:**
- Modify: `src/kraken_hub/presentation/qt/parts/open_project.py`
- Modify: `src/kraken_hub/presentation/qt/parts/construction.py`
- Modify: `src/kraken_hub/presentation/qt/parts/poll_agent_jobs.py`
- Modify: `src/kraken_hub/presentation/qt/parts/rename_project.py` (rename + archive)
- Modify: `src/kraken_hub/presentation/qt/parts/delete_project.py`
- Modify: `src/kraken_hub/presentation/qt/manager_app.py` (import the new mixin)
- Create: `src/kraken_hub/presentation/qt/parts/close_project_tab.py`
- Modify: `src/kraken_manager/presentation/qt/shell.py` (remove `open_project_workspace`; drop the `"workspace"` clause from `_sync_chrome_for_current_page`)
- Test: `tests/test_hub_project_tabs.py` (new)
- Test: `tests/test_manager_qt_shell.py` (rewrite the 3 tests that use `open_project_workspace`)
- Test: `tests/test_layer_management_features.py` (1 call site)
- Test: `tests/test_server_workspace_inputs_parts/prepared_server_images.py` (1 call site)

**Interfaces:**
- Consumes: Task 1 shell API (`open_project_tab`, `close_project_tab`, `rename_project_tab`, `current_project_id`, `project_tab_activated`, `project_tab_close_requested`, `addProjectRequested`); existing helpers `self._load_layers(workspace, project)`, `self._select_layer(workspace, project_id, item)`, `self._setup_project_workspace(project)`, `self._configure_analysis(workspace, project_id)`, `self.service.project_permissions`, `self.service.get_project`.
- Produces:
  - `controller._project_workspaces: dict[str, ProjectWorkspacePage]` — the registry of open projects (source of truth for "which projects are open").
  - `controller._on_project_tab_activated(project_id: str | None)` — `None` clears `_project_id`/`_workspace`; otherwise `_activate_project`.
  - `controller._activate_project(project_id: str) -> None` — closes `_layer_dialog` if open; `get_project`; if the project is gone → `_on_project_tab_close_requested(project_id)`; otherwise set active pointers, `set_project_title`, `matrix_view.set_visual_modes(*shell.visual_modes())`, `_sync_review_return_visibility(project_id)`, `_load_layers`, `_apply_retained_wakes(project_id)`.
  - `controller._on_project_tab_close_requested(project_id: str) -> None` (in `parts/close_project_tab.py`) — pop registry; call `service.unsubscribe_project` only when the method exists and the project is remote (same `callable`/`is_remote_project` guards as today); if the closed project was active, clear `_project_id`/`_workspace` and close `_layer_dialog`; call `workspace.matrix_view.close()` (do **not** clear the thumbnail cache — the project is still openable); then `self.shell.close_project_tab(project_id)`.
  - `controller._apply_wake_to_active(project_id: str) -> None` and `controller._apply_retained_wakes(project_id: str) -> None` (in `parts/poll_agent_jobs.py`) — `_apply_wake_to_active` is the extracted body of today's drain (title + matrix `set_session` with `generation + 1` + `_load_layers`); `_apply_retained_wakes` removes that project's queued wakes and, if any, runs `_apply_wake_to_active` once.

- [ ] **Step 1: Write the failing tests** — create `tests/test_hub_project_tabs.py`:

```python
"""Multi-project tab state for the desktop controller."""
from __future__ import annotations
import os
import threading
from pathlib import Path
from uuid import uuid4
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication, QMessageBox

from kraken_hub.application.notifications import ProjectEventWake
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.presentation.qt import manager_app
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.domain.project import GridOrientation
from kraken_manager.presentation.qt import FrameSelection, ProjectListItem, ProjectManagerShell


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def environment(qapp, tmp_path, monkeypatch):
    class FakeRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            self.process = None
            self.base_url = ""
            self.token = ""
        def ensure_started(self):
            return frozenset()
        def shutdown(self) -> None:
            return None
        def detach(self) -> None:
            return None

    monkeypatch.setattr(manager_app, "LocalAgentRuntime", FakeRuntime)
    service = EmbeddedProjectService(tmp_path)
    session = service.create_initial_account("admin", "Administrator", "")
    shell = ProjectManagerShell()
    controller = DesktopController(shell, service, session, thumbnail_store_uri="memory://", plugin_items=[])
    controller._agent_timer.stop()
    yield controller, shell, service, session
    controller._sync_timer.stop()
    for workspace in list(getattr(controller, "_project_workspaces", {}).values()):
        timer = getattr(workspace, "_analysis_timer", None)
        if timer is not None:
            timer.stop()
    controller.shell.close_guard = None
    shell.close()


def _create_project(service, session, name, root: Path):
    sources = root / f"src-{name}"
    derived = root / f"der-{name}"
    sources.mkdir()
    derived.mkdir()
    return service.create_project(
        principal=session.principal,
        name=name,
        width=4,
        height=3,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
        layer_template=True,
        source_root=sources,
        derived_root=derived,
    )


def _item(project) -> ProjectListItem:
    return ProjectListItem(str(project.id), project.name, project.width, project.height, "local")
```

Tests to add in the same file (assertions pinned, bodies per spec):

```python
def test_opening_two_projects_keeps_both_tabs_alive(environment, tmp_path):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    ws1 = controller._workspace
    ws1.matrix_view.set_selection(FrameSelection.single(1, 1))
    controller.open_project(_item(p2))
    assert shell.projectTabs.count() == 2
    assert shell.current_project_id() == str(p2.id)
    assert controller._project_workspaces[str(p1.id)] is ws1
    shell.projectTabs.setCurrentIndex(0)
    assert controller._project_id == str(p1.id)
    assert controller._workspace is ws1
    assert len(list(ws1.matrix_view.selection().coordinates())) == 1  # selection survives the switch


def test_reopening_open_project_reuses_workspace(environment, tmp_path):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    controller.open_project(_item(p1))
    ws1 = controller._workspace
    controller.open_project(_item(p1))
    assert controller._workspace is ws1
    assert shell.projectTabs.count() == 1


def test_layer_manager_dialog_closes_on_tab_switch(environment, tmp_path):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    controller.open_layer_manager()
    assert controller._layer_dialog is not None
    controller.open_project(_item(p2))
    assert controller._layer_dialog is None


def test_wakes_for_open_inactive_project_are_retained_and_applied_on_activation(environment, tmp_path):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    controller.open_project(_item(p2))  # p2 active
    ws1 = controller._project_workspaces[str(p1.id)]
    generation_before = ws1.matrix_view.session.generation
    wake = ProjectEventWake(project_id=str(p1.id), event_type="Layer.updated", event_id="e-1")
    controller._on_remote_wake(wake)
    controller._sync_timer.stop()
    controller._drain_live_wakes()
    assert any(w.project_id == str(p1.id) for w in controller._pending_wakes.values())
    assert ws1.matrix_view.session.generation == generation_before
    stale = ProjectEventWake(project_id="ghost", event_type="Layer.updated", event_id="e-2")
    controller._on_remote_wake(stale)
    controller._drain_live_wakes()
    assert not any(w.project_id == "ghost" for w in controller._pending_wakes.values())
    shell.projectTabs.setCurrentIndex(0)  # activate p1
    assert ws1.matrix_view.session.generation == generation_before + 1
    assert not any(w.project_id == str(p1.id) for w in controller._pending_wakes.values())


def test_activated_tab_for_deleted_project_closes_gracefully(environment, tmp_path, monkeypatch):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    controller.open_project(_item(p2))
    real_get = service.get_project
    monkeypatch.setattr(
        service, "get_project",
        lambda pid: None if str(pid) == str(p1.id) else real_get(pid),
    )
    shell.projectTabs.setCurrentIndex(0)  # p1 is gone -> its tab must close, p2 becomes active
    assert str(p1.id) not in controller._project_workspaces
    assert shell.projectTabs.count() == 1
    assert controller._project_id == str(p2.id)


def test_closing_remote_project_tab_unsubscribes(monkeypatch):
    from types import SimpleNamespace

    controller = object.__new__(DesktopController)
    closed, closed_tabs = [], []

    class ServiceStub:
        is_remote_project = staticmethod(lambda _id: True)

        def unsubscribe_project(self, project_id):
            closed.append(project_id)

    class ShellStub:
        def close_project_tab(self, project_id):
            closed_tabs.append(project_id)

    workspace = SimpleNamespace(
        matrix_view=SimpleNamespace(close=lambda: None),
    )
    controller.service = ServiceStub()
    controller.shell = ShellStub()
    controller._project_workspaces = {"p-r": workspace}
    controller._project_id = "p-r"
    controller._workspace = workspace
    controller._layer_dialog = None

    controller._on_project_tab_close_requested("p-r")

    assert closed == ["p-r"]
    assert closed_tabs == ["p-r"]
    assert controller._project_id is None
    assert controller._workspace is None


def test_closing_project_tab_tolerates_service_without_unsubscribe():
    from types import SimpleNamespace

    controller = object.__new__(DesktopController)

    class ServiceStub:  # local service: no unsubscribe_project at all
        is_remote_project = staticmethod(lambda _id: False)

    class ShellStub:
        def close_project_tab(self, project_id):
            pass

    controller.service = ServiceStub()
    controller.shell = ShellStub()
    controller._project_workspaces = {"p-l": SimpleNamespace(matrix_view=SimpleNamespace(close=lambda: None))}
    controller._project_id = None
    controller._workspace = None
    controller._layer_dialog = None

    controller._on_project_tab_close_requested("p-l")  # must not raise

    assert controller._project_workspaces == {}


def test_close_guard_blocks_with_multiple_open_projects(environment, tmp_path, monkeypatch):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    controller.open_project(_item(p2))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

    class Working(QThread):
        def __init__(self, release):
            super().__init__()
            self._release = release

        def run(self):
            while not self._release.isSet():
                self.msleep(20)

    release = threading.Event()
    worker = Working(release)
    worker.start()
    controller._background_workers.add(worker)
    assert controller._allow_close() is False
    release.set()
    worker.wait(5000)
    controller._background_workers.discard(worker)
    assert controller._allow_close() is True
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_hub_project_tabs.py -v`
Expected: FAIL — `AttributeError`/`AssertionError` on `_project_workspaces`, reuse of workspace, dialog close, wake retention (all absent today).

- [ ] **Step 3: Implement the controller rework**

`parts/construction.py` (`ConstructionMixin.__init__`):
- add `ProjectWorkspacePage` to the existing `from kraken_manager.presentation.qt import (...)` import in this file, and after `self._layer_dialog = None` add `self._project_workspaces: dict[str, ProjectWorkspacePage] = {}`.
- alongside the other `self.shell.*` connections add:
  `self.shell.project_tab_activated.connect(self._on_project_tab_activated)`,
  `self.shell.project_tab_close_requested.connect(self._on_project_tab_close_requested)`,
  `self.shell.addProjectRequested.connect(self.refresh_projects)`.

`parts/open_project.py`:
- `open_project(item)`: keep the top checks (project missing → error; `is_remote`). After `project_id = str(project.id)`: if `project_id in self._project_workspaces` → `self.shell.open_project_tab(project_id, project.name, self._project_workspaces[project_id])` and return (activation sync runs via the shell signal). Otherwise keep the first-open body (storage binding via `_setup_project_workspace` when needed, "Хранилище недоступно" warning, thumbnail store, data/asset sources, `FrameMatrixWidget`), then:
  ```python
  page = ProjectWorkspacePage(matrix_view=matrix_view)
  self._project_workspaces[project_id] = page
  workspace = self.shell.open_project_tab(project_id, project.name, page)
  ```
  then the first-open-only wiring: `workspace._matrix_data_source`/`_matrix_asset_source`, `matrix_view.set_session(...)`, the `addLayerRequested`/`layerActivated`/representation/context-menu signal connections, `self._configure_analysis(workspace, project.id)`, remote `subscribe_project`, and the first-layer selection block. Remove from this body: `self._workspace = ...` / `self._project_id = ...` assignments (the activation signal sets them — `open_project_tab` activates the new tab synchronously), the old-project `unsubscribe` block (subscriptions are per-tab now), the `open_project_workspace` call, the `_layer_dialog` hide block (done by `_activate_project`), the duplicate `set_project_title` / `set_visual_modes` / `review_return` block (done by `_activate_project`).
- Add `_on_project_tab_activated(project_id)` (per Interfaces) and `_activate_project(project_id)` (per Interfaces), plus `_sync_review_return_visibility(project_id)` — extract the existing `review_return_action.setVisible` permission block from `open_project` (including the remote-principal fallback) verbatim.

`parts/close_project_tab.py` (new file, `CloseProjectTabMixin`): `_on_project_tab_close_requested` per the Interfaces block. Register the mixin in `manager_app.py` (import + base-class tuple, matching the existing `parts.*` pattern).

`parts/poll_agent_jobs.py`:
- extract `_apply_wake_to_active(project_id)` from the current `_drain_live_wakes` body (title + `set_session` with `generation + 1` + `_load_layers`, guarded by `get_project` returning a project and the workspace existing in the registry).
- add `_apply_retained_wakes(project_id)` per Interfaces.
- rewrite `_drain_live_wakes`: clear the queue; compute `catalog_changed` as today (and `refresh_projects` when set); `active = self._project_id`; if `(catalog_changed or active has wakes)` and `active` is open → `_apply_wake_to_active(active)` once; then re-queue every wake whose `project_id` is in `self._project_workspaces` but not `active`, using the same key recipe `(project_id, entity_kind or stream_id, entity_id or stream_id)`; all other wakes are dropped.

`parts/rename_project.py`:
- `rename_project`: after `self.refresh_projects()` add `self.shell.rename_project_tab(str(renamed.id), renamed.name)` (keep the existing active-title re-set).
- `archive_project`: after `self.refresh_projects()` add: if `str(project.id) in self._project_workspaces` → `self._on_project_tab_close_requested(str(project.id))`.

`parts/delete_project.py`: replace the `if self._project_id == str(project.id) and self._workspace is not None:` block (thumbnail-cache clear, matrix close, pointer/dialog reset) with `if str(project.id) in self._project_workspaces: self._on_project_tab_close_requested(str(project.id))`.

`shell.py`: delete `open_project_workspace` and the `"workspace"` page concept; in `_sync_chrome_for_current_page` the condition becomes `self.current_page_key().startswith("project:")`.

- [ ] **Step 4: Update the tests that used the old API**

- `tests/test_manager_qt_shell.py`:
  - `test_shell_opens_lightweight_project_workspace` → `workspace = shell.open_project_tab("p-1", "P1", ProjectWorkspacePage())`; assert `shell.current_page_key() == "project:p-1"` (rest unchanged).
  - `test_workspace_uses_bottom_layer_tabs_and_status_bar_selection` → same call-site replacement (the `shell.statusBar().currentMessage() == "Выбрано кадров: 1"` assertion keeps working through the shell's selection tracking).
  - `test_selected_projects_navigation_can_return_from_workspace` → `shell.open_project_tab("p-1", "Project", QWidget())` instead of `register_page("workspace", ...)`; keep the nav click; assert `current_page_key() == "projects"` and `current_project_id() is None`.
- `tests/test_layer_management_features.py` (`test_workspace_menus_and_modeless_layer_manager`): `workspace = shell.open_project_tab("p-1", "P1", ProjectWorkspacePage())` (import `ProjectWorkspacePage` from `kraken_manager.presentation.qt` if not already).
- `tests/test_server_workspace_inputs_parts/prepared_server_images.py`: same one-line replacement.

- [ ] **Step 5: Run the affected test suites**

Run: `uv run pytest tests/test_hub_project_tabs.py tests/test_manager_qt_shell.py tests/test_layer_management_features.py tests/test_server_workspace_inputs_parts tests/test_desktop_runtime_smoke.py tests/test_hub_manager_app.py -v`
Expected: PASS. (Live-server/live-agent smoke tests skip themselves without env vars.)

- [ ] **Step 6: Commit**

```bash
git add src/kraken_hub/presentation/qt src/kraken_manager/presentation/qt/shell.py tests/test_hub_project_tabs.py tests/test_manager_qt_shell.py tests/test_layer_management_features.py tests/test_server_workspace_inputs_parts/prepared_server_images.py
git commit -m "Keep open projects alive in tabs and track per-project controller state"
```

---

### Task 3: Icon rail, status-bar session/sync, theme QSS

**Files:**
- Modify: `src/kraken_manager/presentation/qt/shell.py`
- Modify: `src/kraken_core/resources/styles/dark_modern.qss`
- Modify: `src/kraken_core/resources/styles/style.qss`
- Test: `tests/test_manager_qt_shell.py`
- Test: `tests/test_live_server_worker_faults.py`
- Test: `tests/test_live_ui_workflow.py`
- Test: `tests/test_hub_manager_app_parts/contour_vectorize_receives.py`

**Interfaces:**
- Consumes: `DEFAULT_NAVIGATION` (unchanged: 6 keys/labels in order), `set_page_visible`, `set_session_summary`, `set_sync_status`.
- Produces:
  - `shell.rail_actions: dict[str, QToolButton]` — insertion-ordered per `DEFAULT_NAVIGATION`; each button checkable, exclusive (`QButtonGroup`), tooltip = section label, 22×22 hand-drawn icon (objectNames `railButton_<key>`).
  - `shell` no longer has `navigation_list`, `_navigation_items`, `session_label`, `sync_status_label`, or the "KRAKEN" brand label.
  - `sessionSummary` / `syncStatus` become persistent `QLabel`s in the status bar (left side, via `statusBar().addWidget`), same object names and same label texts as today (`set_sync_status` keeps the `● …` mapping and the `status` property + polish/unpolish).
  - `set_page_visible(key, visible)` maps to the rail button's visibility; hiding the current section falls back to `show_page("projects")` (existing semantics).

- [ ] **Step 1: Write the failing tests** — in `tests/test_manager_qt_shell.py` (add `QLabel` to its `PyQt6.QtWidgets` import), replace `test_project_manager_shell_has_stable_navigation_and_replaceable_pages`' `shell.navigation_list.count() == 6` line with `assert len(shell.rail_actions) == 6`, replace `test_project_manager_shell_can_hide_administration_navigation` with the rail version below, and add:

```python
def test_shell_rail_exposes_section_buttons_in_order(qapp):
    shell = ProjectManagerShell()
    assert list(shell.rail_actions) == [
        "projects", "my_work", "statistics", "performers", "plugins", "administration",
    ]
    for button in shell.rail_actions.values():
        assert not button.icon().isNull()
        assert button.toolTip()
    assert not hasattr(shell, "navigation_list")


def test_shell_rail_button_switches_section_and_highlights(qapp):
    shell = ProjectManagerShell()
    shell.rail_actions["statistics"].click()
    assert shell.current_page_key() == "statistics"
    assert shell.rail_actions["statistics"].isChecked()
    assert not shell.rail_actions["projects"].isChecked()
    shell.open_project_tab("p-1", "P1", QWidget())
    assert not any(button.isChecked() for button in shell.rail_actions.values())


def test_shell_set_page_visible_hides_rail_button_and_falls_back(qapp):
    shell = ProjectManagerShell()
    shell.show_page("administration")
    shell.set_page_visible("administration", False)
    assert not shell.rail_actions["administration"].isVisible()
    assert shell.current_page_key() == "projects"
    shell.set_page_visible("administration", True)
    assert shell.rail_actions["administration"].isVisible()


def test_shell_session_and_sync_labels_live_in_status_bar(qapp):
    shell = ProjectManagerShell()
    shell.set_session_summary("mark\nЛокальная сессия + сервер http://127.0.0.1:8080")
    shell.set_sync_status("synchronized")
    session_label = shell.findChild(QLabel, "sessionSummary")
    sync_label = shell.findChild(QLabel, "syncStatus")
    assert session_label is not None and session_label.isVisible()
    assert "mark" in session_label.text()
    assert sync_label is not None and sync_label.isVisible()
    assert sync_label.text() == "● Синхронизировано"
    shell.set_sync_status("")
    assert not sync_label.isVisible()
```

Also update `test_selected_projects_navigation_can_return_from_workspace` (introduced in Task 2) to click the rail:
`qtbot.mouseClick(shell.rail_actions["projects"], Qt.MouseButton.LeftButton)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_manager_qt_shell.py -v`
Expected: the new/updated tests FAIL (`rail_actions` missing / `navigation_list` still present); others PASS.

- [ ] **Step 3: Implement the rail and status-bar labels in `shell.py`**

- `_build_rail()` replaces `_build_sidebar()`: `QFrame#managerRail`, `setFixedWidth(48)`, `QVBoxLayout` (margins 6/8/6/8, spacing 6), a `QButtonGroup` (exclusive), and one checkable `QToolButton` per `DEFAULT_NAVIGATION` entry (autoRaise, 22×22 icon via `setIconSize`, tooltip = label, objectName `railButton_<key>`), each connected to a `_on_rail_clicked(key)` handler that does `self.navigationRequested.emit(key)` then `self.show_page(key)`; trailing `addStretch(1)`. Root layout becomes `[rail][content column]` (the content column from Task 1). Remove `_build_sidebar`, the brand label, `navigation_list`, `_navigation_items`, `session_label`, `sync_status_label`, `_on_navigation_row_changed`, `_on_navigation_item_clicked`.
- `_rail_icon(key) -> QPixmap`: 24×24 transparent pixmap, `QPen(QColor("#647d9c"), 2.0)` round caps, one simple line glyph per key: projects = folder outline; my_work = single person (head circle + shoulder arc); statistics = three vertical bars of different heights; performers = two overlapping head circles + shoulders; plugins = plug (rounded rect body + two prongs); administration = gear (circle + 8 short spokes).
- `show_page`: after switching, set each rail button `setChecked(key == normalized)`; `_on_project_tab_changed` unchecks all rail buttons (project tab active).
- `set_page_visible`: operate on `self.rail_actions[key]` (raise `KeyError` as before when unknown); keep the current-section fallback to `show_page("projects")`.
- `set_session_summary` / `set_sync_status`: create in `__init__` persistent labels `QLabel(objectName="sessionSummary")` (initial "Нет активной сессии", word wrap) and `QLabel(objectName="syncStatus")` (word wrap, hidden) and `self.statusBar().addWidget(...)` both; keep today's exact text mapping and the `status` property + `unpolish/polish` dance in `set_sync_status`.
- QSS — `dark_modern.qss` append:

```css
#managerRail {
    background-color: #0d131a;
    border-right: 1px solid #223244;
}

#managerRail QToolButton {
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 6px;
}

#managerRail QToolButton:hover {
    background-color: #1d2a39;
}

#managerRail QToolButton:checked {
    background-color: #243447;
}

#projectTabs QTabBar::close-button {
    subcontrol-origin: padding;
    subcontrol-position: right;
    right: 4px;
}
```

  `style.qss` (light): same selectors with light-theme tones taken from that file's existing palette (rail background one step darker than the theme's base `QWidget` background, border from the theme's border tone, hover/checked from its selection tints) plus the same `#projectTabs QTabBar::close-button` positioning rule. The `#managerRail QToolButton` override is required because the global `QToolButton` rule paints a blue background. The tab close glyph itself stays the painted `#647d9c` pixmap from Task 1 (QSS cannot recolor a painted icon), so the QSS only positions the close button.

- [ ] **Step 4: Update the remaining tests that use the old navigation API**

- `tests/test_live_server_worker_faults.py` (`_work_ui`): replace the two `navigation_list`/`_navigation_items` lines + `QTest.mouseClick` with `shell.rail_actions["my_work"].click()`.
- `tests/test_live_ui_workflow.py` (`select_catalog_project`): replace `navigation = ...; item = ...; QTest.mouseClick(...)` with `controller.shell.rail_actions["projects"].click()`.
- `tests/test_hub_manager_app_parts/contour_vectorize_receives.py`: `administration_item = shell._navigation_items["administration"]; assert not administration_item.isHidden()` → `administration_button = shell.rail_actions["administration"]; assert administration_button.isVisible()`.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest tests -x -q`
Expected: PASS (live-server/live-agent tests skip without env vars; no other regressions).

- [ ] **Step 6: Manual theme check**

Start the real app via the project's normal entry point on a machine with a display: open the window, verify the rail icons and the tab bar (with close glyphs and "+") render in the active theme, open two projects, switch to a section and back, then switch the theme (Вид → Настройки / theme menu) and verify rail + tabs stay readable in both dark and light. Note the outcome in the commit message trailer. If no display is available, say so explicitly instead of claiming the check passed.

- [ ] **Step 7: Commit**

```bash
git add src/kraken_manager/presentation/qt/shell.py src/kraken_core/resources/styles/dark_modern.qss src/kraken_core/resources/styles/style.qss tests/test_manager_qt_shell.py tests/test_live_server_worker_faults.py tests/test_live_ui_workflow.py tests/test_hub_manager_app_parts/contour_vectorize_receives.py
git commit -m "Replace manager sidebar with a compact icon rail and move session state to the status bar"
```
