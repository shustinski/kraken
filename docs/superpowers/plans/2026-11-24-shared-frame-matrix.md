# Shared Frame Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Contour's frame matrix becomes a subclass of `kraken_core.frame_matrix.qt.FrameMatrixWidget`; Kraken and Contour matrices differ only in declared frame actions.

**Architecture:** The base view gains per-axis cell geometry with overlap and a declarative action menu. Kraken hub gets `LayerFrameMatrixWidget` with its four actions. Contour's `FrameMatrixGraphicsView` is rebuilt on `FrameMatrixWidget` with a data source over its frame list and an asset source over files / `PyramidFrameStore`, keeping the `QListWidget` subset its callers use.

**Tech Stack:** Python 3.14, PyQt6, pytest/pytest-qt.

**Spec:** `docs/superpowers/specs/2026-11-24-shared-frame-matrix-design.md`

## Global Constraints

- Kraken defaults stay 48×48 cells, no gap, no overlap; existing Kraken matrix tests must pass unchanged in meaning.
- Action ids: Kraken `export_files`, `assign_work`, `vector_properties`, `frame_properties`; Contour `delete_frame`, `clear_frame`.
- Russian Kraken titles unchanged; Contour titles come from `self._tr("delete_frame_action")` / `self._tr("clear_frame_action")`.
- Run `graphify update .` after code changes.

## Review Focus

- Overlap larger than a cell: clamp to `cell - 1` per axis, as Contour does today.
- Hidden items (asset filter): no cell is produced; navigation and clicks skip them.
- Empty frame list: the matrix shows a 1×1 empty grid and emits nothing on click.
- Right-click on a frame outside the selection selects it first (base behaviour) before building the menu.
- Action predicate raising an exception: the action is hidden, the menu still opens.

---

### Task 1: Per-axis cell geometry in the base view

**Files:**
- Modify: `src/kraken_core/frame_matrix/qt.py`, `qt_parts/construction.py`, `qt_parts/paint_tile_cells.py`, `qt_parts/update_visible_tiles.py`, `qt_parts/viewport_tile_item.py`, `qt_parts/frame_matrix_minimap.py`, `qt_parts/construction_2.py`
- Test: `tests/test_frame_matrix_geometry.py`

**Interfaces:**
- Produces: `FrameMatrixView.set_cell_geometry(width: float, height: float, overlap_x: float = 0.0, overlap_y: float = 0.0) -> None`; `cell_size() -> tuple[float, float]`; `cell_pitch() -> tuple[float, float]`; `_cell_rect(column: int, row: int) -> QRectF` (0-based).

- [ ] Test: 64×48 cells with overlap 8/6 → `scene_rect_for_frame(2, 1) == QRectF(56, 0, 64, 48)`, scene rect width `3*56+8` for width 3, `frame_at_scene_pos(QPointF(60, 10)) == (2, 1)`; overlap 100 clamps to 63/47.
- [ ] Replace every `CELL_PITCH`/`CELL_SIZE` read with the per-axis values; tile bounds extend by `cell - pitch`.
- [ ] Run Kraken matrix tests + new test: PASS.

### Task 2: Declarative frame actions

**Files:**
- Create: `src/kraken_core/frame_matrix/actions.py`
- Modify: `qt_parts/context_menu_event.py`, `qt.py` (signal `frameActionTriggered = pyqtSignal(str, object)`, export `FrameMatrixAction`)
- Test: `tests/test_frame_matrix_actions.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) FrameMatrixAction(id: str, title: str | Callable[[FrameContext], str], is_visible: Callable[[FrameContext], bool] | None = None)`; `FrameMatrixView.frame_actions(context) -> Sequence[FrameMatrixAction]` (default `()`); `FrameMatrixView.visible_frame_actions(context) -> tuple[tuple[FrameMatrixAction, str], ...]`; `_exec_frame_menu(menu, global_pos)` seam for tests.

- [ ] Test: subclass with two actions, one hidden → `visible_frame_actions` returns one; context menu with patched `_exec_frame_menu` choosing it emits `frameActionTriggered("a", context)`; no actions → `contextMenuRequested` still emitted; raising predicate hides action.
- [ ] Implement; run tests: PASS.

### Task 3: Kraken layer matrix

**Files:**
- Create: `src/kraken_hub/presentation/qt/layer_matrix.py`
- Modify: `parts/open_project.py`, `parts/show_matrix_context_menu.py`
- Test: `tests/test_layer_matrix_actions.py`

**Interfaces:**
- Produces: `LayerFrameMatrixWidget(FrameMatrixWidget)` with `set_action_host(host)`; host protocol `can_assign_work() -> bool`, `vector_file_for(x, y) -> object | None`. `ShowMatrixContextMenuMixin._on_matrix_frame_action(action_id: str, context) -> None`.

- [ ] Test: selection of 1 → export title "Выгрузить файлы кадра…", frame_properties visible; selection of 3 → "Выгрузить файлы выбранных кадров (3)…", frame_properties hidden; assign_work only when host allows; vector_properties only when host returns a file.
- [ ] Implement, wire `frameActionTriggered`; run hub tests: PASS.

### Task 4: Contour matrix on the shared engine

**Files:**
- Create: `plugins/contour/src/contour/ui/frame_matrix_sources.py`
- Rewrite: `plugins/contour/src/contour/ui/frame_matrix_view.py` (remove `frame_matrix_view_parts/*` except `trailing_frame_number.py`, `module_state.py`)
- Modify: `ui/builders_parts/build_files_tab.py`, `widget_parts/navigation_mixin_parts/activate_frame_path_matrix.py`, legacy icon pipeline gates
- Test: `plugins/contour/tests/unit/test_frame_lod_pyramid.py` (rewrite matrix part), `test_contour_frame_matrix.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `FrameMatrixGraphicsView(FrameMatrixWidget)` with QListWidget subset (`count, item, addItem, clear, currentRow, currentItem, setCurrentRow, clearSelection, visualItemRect, setIconSize, setGridSize, setSpacing, setFrameOverlapPixels, setColumns, setPyramidFrameStore, navigatorLods, setCurrentFrameId, updateThumbnailPixmaps, refreshItems, refreshVisibleRegion, doItemsLayout, thumbnailSourceSize, shutdownPyramidLoading`, no-op setters), `owns_thumbnail_loading = True`, actions `delete_frame`/`clear_frame`; `ContourMatrixDataSource(snapshot)`; `ContourFrameAssetSource(paths, pyramid_store)`.

- [ ] Test: 5 items, 2 columns → cells at (1,1),(2,1),(1,2),(2,2),(1,3); hidden item yields no cell; click on cell (2,1) emits `frameNavigationRequested(1)` and `itemClicked(item 1)`; `visible_frame_actions` ids == `("delete_frame", "clear_frame")`; asset source returns PNG bytes of the requested size from a pyramid and from a file.
- [ ] Implement; wire contour handlers; run contour unit tests: PASS.

### Task 5: Verify and update graph

- [ ] Kraken matrix tests, hub tests, contour unit tests: PASS. `graphify update .`
