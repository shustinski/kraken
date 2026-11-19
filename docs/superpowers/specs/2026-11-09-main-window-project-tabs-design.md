# Kraken Hub main window: project tabs + compact navigation rail

Date: 2026-11-09
Status: draft — pending owner approval
Scope: `ProjectManagerShell` (kraken_manager) + `DesktopController` and its mixins (kraken_hub)

## Intent and why

The current main window treats the project workspace as a disposable guest page:

1. **The project "closes" on navigation.** The workspace is a page registered without a
   sidebar entry (`navigation=False` in `ProjectManagerShell.open_project_workspace`).
   Clicking "Моя работа" or "Статистика" hides it, and there is no direct way back —
   the only path is "Проекты" → click the project again. That re-open destroys the old
   workspace page (`replace_page` → `previous.close(); previous.setParent(None)`) and
   builds a fresh `ProjectWorkspacePage` with a fresh `FrameMatrixWidget`: matrix
   selection, active layer, scroll position, and loaded state are all lost.
2. **The permanent left list is heavy.** The sidebar is a 190–280 px wide text list
   (Проекты, Моя работа, Статистика, Исполнители, Плагины, Администрирование) that is
   always present and dominates the window while the user is actually working in a
   project.

Goal: the project becomes a first-class, persistent working context. Several projects
can be open at once; leaving a project never destroys it; the window chrome is compact.

Owner decisions (conversation of 2026-11-09):

- Projects open as **tabs at the top** of the content area, visually following the
  existing layer-tab pattern (`ProjectWorkspacePage.layer_tabs`, `RoundedSouth`),
  mirrored to `RoundedNorth` at the top.
- Sections move to a **thin left icon rail (~48 px)** — "variant A" of the two layout
  options discussed. The owner explicitly kept variant B (pinned section tabs, no rail)
  in mind as a possible later change, so section access must stay independent of the
  project tab bar.
- The Administration section is out of scope for redesign: it keeps its current
  behavior and only changes location (rail icon).

## Goals

1. Multiple projects open simultaneously; each has its own tab; per-project state
   (matrix selection, active layer, scroll, loaded data, analysis panel) survives tab
   switches and switches to section pages.
2. Re-opening an already-open project focuses its existing tab — no rebuild.
3. Closing a tab closes exactly that project (unsubscribes, closes its dialogs).
4. Sections (Проекты catalog, Моя работа, Статистика, Исполнители, Плагины,
   Администрирование) are always reachable from the ~48 px icon rail.
5. The project tab bar is always visible, so the active project is one click away from
   any section page.
6. Window title, menu bar enablement (Управление / Действия / Вид), and status-bar
   behavior are preserved in spirit.

## Non-goals (v1)

- No redesign of the Administration section itself.
- No live background updates for inactive open projects beyond keeping their queued
  wakes (see "Live wakes"); no tab badges.
- No persistence of the open tab set across app restarts.
- No drag-reorder, tab context menus, or tab search (tabs are not movable, mirroring
  layer tabs).
- No memory unloading of inactive projects.
- No changes to the catalog page content/actions, layer tabs, or matrix widget internals.

## Anti-goals

- Never destroy workspace state on navigation (the bug being fixed).
- Do not mix section pages and project pages in one tab bar (that was variant B).
- No new dependencies, no new UI framework; PyQt6 + existing architecture only.

## Constraints

- **Layering.** `ProjectManagerShell` (`src/kraken_manager/presentation/qt/shell.py`)
  is "intentionally free of composition logic": it must not know about
  `ProjectWorkspaceService` or sessions. The shell owns the tab bar, the rail, the page
  registry, and emits signals; `DesktopController`
  (`src/kraken_hub/presentation/qt/manager_app.py` + `parts/*` mixins) implements
  project open/close semantics and wires signals.
- **Themes.** The app stylesheet is `dark_modern.qss` (dark) or `style.qss` (light)
  from `src/kraken_core/resources/styles/` (see `kraken_core/theme.py`). There are
  currently **no** QSS selectors for manager chrome (`managerSidebar` etc. have no
  styles; the sidebar renders with default `QWidget` styling). New chrome needs
  selectors added to both theme files; do not refactor the global QSS.
- **Icon resources.** `src/kraken_core/resources/icons/` contains only the Kraken logo
  (`kraken.ico`, `kraken.png`). Rail icons and the tab close glyph must be
  hand-drawn `QPixmap` line icons — an established pattern in this codebase (the layer
  "+" button in `pages.py` paints its plus icon).
- **Existing tests pin the old contract.** `tests/test_manager_qt_shell.py` asserts
  `navigation_list.count() == 6`, `replace_page`, `set_page_visible`,
  `open_project_workspace()` with no arguments, and `current_page_key() == "workspace"`.
  `tests/test_live_server_worker_faults.py` clicks `shell.navigation_list`. These encode
  the intentionally-changed behavior and must be rewritten against the new contract.
  Other tests that reference `ProjectManagerShell` (`test_hub_manager_app.py`,
  `test_desktop_runtime_smoke.py`, `test_layer_management_features.py`,
  `test_legacy_project_workspace.py`, `test_live_server_worker_acceptance.py`,
  `test_live_ui_workflow.py`, `test_hub_manager_app_parts/*`,
  `test_server_workspace_inputs_parts/prepared_server_images.py`) must keep passing
  after the rework; adjust them where they depend on the old single-workspace API.
- **Composition root.** `run_manager_gui.py` injects section content via
  `shell.page(key)` + `page.set_content(widget)` for plugins/performers/my_work/
  statistics/administration. Section page keys and the `set_content` API must be kept.

## Current behavior (reference)

- `ProjectManagerShell` (QMainWindow): sidebar `QFrame#managerSidebar` (190–280 px)
  with brand label, `QListWidget#managerNavigation` (6 items), `session_label`,
  `sync_status_label`; `QStackedWidget#managerPageStack` with pages
  `projects / my_work / statistics / performers / plugins / administration` plus a
  hidden single `workspace` page; menu bar with Управление (Слои, План восстановления,
  Физические слои и контакты, Электрическая связность, Контроль качества),
  Действия (Загрузить проверенные файлы, Статистика выбранного кадра), Вид (Настройки,
  Рамка/Заполнение ячейки, Источник обновлений, Проверить обновления).
- `open_project_workspace()` registers or **replaces** the single `workspace` page;
  replacement destroys the previous page.
- `DesktopController.open_project` (`parts/open_project.py`) builds a new
  `ProjectWorkspacePage(matrix_view=FrameMatrixWidget(...))` on every open and stores
  it in single-active-project fields: `self._project_id`, `self._workspace`,
  `self._layer_dialog` — used by ~30 mixin methods across `parts/*`.
- `show_page()` enables the workspace menus only when the current page key is
  `"workspace"`.
- Live sync: `_wire_live_sync`/`_on_remote_wake`/`_drain_live_wakes`
  (`parts/poll_agent_jobs.py`) queue `ProjectEventWake`s in `self._pending_wakes`
  (keyed by project/entity) but `_drain_live_wakes` clears the whole queue and applies
  only wakes matching the single active `self._project_id` — wakes for other projects
  are dropped.
- Remote projects: `subscribe_project` / `unsubscribe_project` are switched on the
  single active project in `open_project`.
- Layer tabs (the visual reference): `QTabBar#layerTabs`, `RoundedSouth`,
  `setExpanding(False)`, `setDrawBase(True)`, plus a 38×38 `QPushButton#addLayerButton`
  with a painted plus icon, laid out in an `HBox` row.

## Approaches considered

1. **A — top project tab bar + thin icon rail (chosen).** Project tabs at the top of
   the content area (mirroring layer tabs); sections become ~48 px icon buttons in a
   left rail. Trade-offs: sections remain visible as icons; the tab bar stays clean
   (projects only); smallest structural distance from the existing shell (sidebar →
   rail). Keeps option B open later because section access is decoupled from the tab
   bar.
2. **B — pinned section tabs + project tabs in one row, no rail (parked by owner).**
   Maximum content space, but one row mixes two kinds of tabs; the bar crowds as
   projects accumulate and sections/projects become hard to tell apart.
3. **C — keep the wide sidebar, make the single workspace persistent and reachable via
   a "Проект" nav item (rejected).** Fixes state loss with a minimal change, but does
   not address the always-present wide list and allows only one project at a time.

## Design (variant A)

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│ Управление  Действия  Вид                              ─ ▢ ✕ │
├─────┬────────────────────────────────────────────────────────┤
│ ⌂   │ [heggs ✕] [второй проект ✕] [+]                       │
│ 👤  ├────────────────────────────────────────────────────────┤
│ 📊  │                                                        │
│ ⬡   │        Рабочее пространство проекта (матрица, слои)    │
│ ⚙   │                                                        │
├─────┴────────────────────────────────────────────────────────┤
│ mark · локальная сессия + сервер        ● Синхронизировано   │
└──────────────────────────────────────────────────────────────┘
```

- **Menu bar:** unchanged.
- **Left rail:** fixed width 48 px, `QFrame#managerRail`, vertical `VBox` of checkable
  icon `QToolButton`s — one per section, in the existing order: Проекты, Моя работа,
  Статистика, Исполнители, Плагины, Администрирование. Each button: 24×24 hand-drawn
  line icon, tooltip = section label, `checkable` with exclusive selection (an
  `QButtonGroup`); the checked button highlights the section whose page is shown.
  Clicking a button shows that section page (existing `show_page` path).
- **Tab bar row:** right of the rail, above the content stack, always visible.
  `QTabBar#projectTabs` (`RoundedNorth`, `setExpanding(False)`, `setDrawBase(True)`,
  `setTabsClosable(True)`, not movable) + trailing `QPushButton#addProjectButton`
  (38×38 painted plus icon, same recipe as `addLayerButton`) in an `HBox` row.
  The "+" button activates the "Проекты" catalog page.
- **Content:** the existing `QStackedWidget#managerPageStack`. Project workspaces are
  now one page **per project** (page key `project:<project_id>`); section pages are
  unchanged.
- **Status bar:** session summary becomes a persistent left status-bar widget
  (replacing the sidebar footer label); sync status next to it; the transient
  "Выбрано кадров: N" message behavior is unchanged.
- **Brand:** the "KRAKEN" text label moves out (the rail is icon-only); the window
  icon/title carry the identity.

### Tab semantics

- **Open project** (catalog double-click or any existing open path):
  - if the project already has a tab → activate that tab and run the **activation
    sync** below; no rebuild;
  - otherwise create `ProjectWorkspacePage(matrix_view=FrameMatrixWidget(...))`
    exactly as today, run the **first-open setup** below, register it under
    `project:<id>`, add a closable tab labeled with the project name, activate it.
- **Switch tabs:** `setCurrentWidget` only — pages stay alive. The shell emits the
  activated project id (or `None` when a section page takes over); the controller
  updates its active-project pointers from that signal and runs the **activation
    sync** for the incoming project.
- **First-open setup (exactly once per workspace page):** everything
  `open_project` does today that is non-idempotent or a state-of-world notice —
  building the page/matrix, storage binding (`_setup_project_workspace` + its modal)
  and the "Хранилище недоступно" warning, `_configure_analysis` (connects signals
  and starts timers — not idempotent), the per-workspace signal wiring in
  `open_project` (layer/representation/context-menu/selection signals), and
  `subscribe_project` for remote projects. Builder call: the storage warning and
  modal are **first-open only** — re-opening an already-open project must not re-pop
  a dialog; later storage problems surface through `matrix_view.errorOccurred`.
- **Activation sync (on every tab activation and re-open; idempotent):** refresh
  title (`set_project_title`) + layers (`_load_layers`) + matrix session
  regeneration (as in `_drain_live_wakes`); re-apply
  `matrix_view.set_visual_modes(*shell.visual_modes())`; re-evaluate
  `review_return_action` visibility from permissions; re-push the page's current
  selection count (see below); sync menu enablement (see below).
- **Layer manager dialog:** single global instance (`_layer_dialog`). It is **closed
  on tab switch** (builder call, simplest correct behavior: prevents a dialog showing
  project A's pipeline while project B is active; the dialog is a tool the user opens
  when needed). All existing `set_pipeline` call sites keep working unchanged.
- **Close tab:** `unsubscribe_project` (if remote), close that project's layer
  manager dialog if open, remove the page (`close()` + `setParent(None)`) and the tab.
  If the closed tab was active: activate the right neighbor, else the left one; if no
  tabs remain, activate the "Проекты" catalog page.
- **Rename project** (existing rename flow): update the tab label in addition to the
  window title (existing code already re-titles).
- **Delete/archive project that has a tab** (existing delete/archive/restore flows):
  close its tab with the same semantics as a tab close before proceeding; restore
  re-enables re-opening (no tab is auto-recreated).
- **Window title:** shows the active project's name when a project tab is
  active. When a section page is showing, the title is NOT reset — it retains
  the last active project's name (existing behavior, kept as-is by decision
  2026-11-10).
- **Menus:** Управление / Действия / Вид (cell-visual submenus) are enabled exactly
  when the current page is a project workspace — the condition becomes
  "current page key starts with `project:`" (replacing `== "workspace"`). The
  enablement must run through **one "sync chrome for current page" helper invoked
  from both paths** — section navigation (which routes through `show_page`) and
  project-tab activation (which does not) — otherwise switching to a project tab
  would leave the menus disabled (or the logic would be duplicated).
- **Selection status bar message and `frame_properties_action`:** with several
  live workspaces, only the **active** page's selection may drive the global status
  bar and action. Each workspace page keeps emitting `selectionCountChanged`, but
  the shell/controller filters it to the active page and **re-pushes the active
  page's last count on tab activation** (the signal does not fire on a mere tab
  switch, so without the re-push the bar/action could show a different project's
  selection — a visible cross-project leak).

### Controller state rework (main internal change)

- `self._project_id` / `self._workspace` keep their names and their meaning to all
  ~30 mixin call sites: **the active project** (i.e. the current tab's project), not
  "the only project". The controller keeps a registry
  `self._project_workspaces: dict[str, ProjectWorkspacePage]` and updates the active
  pointers from the shell's tab-activation signal.
- `self._layer_dialog` remains a single instance (see above).
- **Per-tab subscriptions:** `subscribe_project` / `unsubscribe_project` are managed
  per open tab (subscribe on open for remote projects, unsubscribe on close), instead
  of being switched on the active project in `open_project`.
- **Live wakes:** the retain/apply/drop predicate in `_drain_live_wakes` is keyed off
  the **open-projects registry** (`_project_workspaces`), not `self._project_id`
  (which is `None` whenever a section page is shown — the common case in the new UI):
  - wake for the active project → apply now (as today);
  - wake for an open-but-inactive project → **retain** in `_pending_wakes` until its
    tab is activated (then applied to that project's page);
  - wake for a project that is not open → drop, as today.
  Catalog-change handling (`refresh_projects`) is unchanged.
- **Agent polling** (`_poll_agent_jobs`) and other active-project logic keep acting on
  the active project only (no behavior change); activating a tab refreshes that
  project (title + layers) via the existing helpers.

### Shell API (new contract)

- Removed: the single hidden `workspace` page and
  `open_project_workspace(workspace=None)`.
- Added:
  - `open_project_tab(project_id: str, name: str, page: QWidget) -> QWidget` —
    register a new project page or return the existing one and activate its tab;
  - `close_project_tab(project_id: str) -> None` — remove page + tab, pick fallback
    activation;
  - `rename_project_tab(project_id: str, name: str) -> None` — update the tab label
    (used by the rename flow);
  - `current_project_id() -> str | None` — id of the active project tab, or `None`
    when a section page is shown;
  - `project_tab_activated = pyqtSignal(object)` (project id or `None`);
  - `project_tab_close_requested = pyqtSignal(str)` (project id);
  - `addProjectRequested = pyqtSignal()` (the "+" button).
- Kept for section pages: `register_page`, `replace_page`, `page(key)`,
  `show_page(key)`, `current_page_key()`, `set_page_visible(key, visible)` (now
  mapping to the rail button's visibility; the administration-hide test must keep
  working semantically), `set_session_summary` / `set_sync_status` (now targeting
  status-bar widgets), `pageChanged`, `navigationRequested`, and all menu signals.
- Replaced: `navigation_list` / `_navigation_items` (QListWidget) → rail button
  registry `rail_actions: dict[str, QToolButton]`.

### Styling

Add selectors to both `dark_modern.qss` and `style.qss`:

- `#managerRail` — slightly distinct background from the content area (dark theme:
  `#0d131a`-class tone; light theme: matching light tone); fixed 48 px.
- Rail buttons: transparent background, 32×32 icon area, hover `#1d2a39`-class,
  checked/active accent `#1f6feb`-class (consistent with existing selection colors);
  disabled state for hidden/absent sections.
- `#projectTabs QTabBar::tab` — the global `QTabBar::tab` style is already
  top-rounded and suitable; add `::close-button` styling (transparent, hover
  highlight) and set the close glyph programmatically as a painted ✕ pixmap
  (hand-drawn, matching the plus-icon recipe) since themes ship no close icon.
- `#addProjectButton` — reuse the `#addLayerButton` styling approach (padded 38×38).

## Validation

- **Unit (shell, rewritten):** `tests/test_manager_qt_shell.py` — rail navigation and
  `set_page_visible` mapping; open/switch/close tab semantics (state survival: a
  widget placed in a workspace page keeps its state across tab + section switches);
  re-open returns the same page instance; close picks the correct fallback tab; menu
  enablement follows the active project tab; status-bar session/sync widgets.
- **Integration (controller):** existing `DesktopController`-level tests keep
  passing; add: two projects open → switch → active-project pointers and
  selection/layer state preserved; re-open open project → same `ProjectWorkspacePage`
  instance (no rebuild); close tab → `unsubscribe_project` called for remote projects;
  wakes for an open inactive project are retained and applied on activation.
- **Manual:** dark and light themes with the new chrome; long tab-label overflow
  (QTabBar scrollers) with several open projects.

## Left to the builder

- Exact rail icon glyphs (hand-drawn line icons; keep the 24×24 / stroke style
  consistent with the existing painted plus icon).
- Refresh details on tab activation (which existing helper(s) to reuse:
  `set_project_title` + `_load_layers` + matrix `set_session` regeneration as in
  `_drain_live_wakes`).
- Tab close glyph pixmap details.
- Test fixture wiring for the new shell API in the test modules listed under
  Constraints.
