# Matrix Tiles in the Project Folder, Delivered in Blocks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tiles and previews are prepared ahead of time in `<source_project_dir>/tile`, never rebuilt for an unchanged layer, and served to the matrix one block per screen.

**Architecture:** `PreparedTilePyramid` gets a shared per-project preview folder and a set folder keyed by layer and image representation. The build runs a preview pass then a compose pass, serving urgent requests between frames. A server tile scheduler, fed by the outbox publisher, starts builds 30 s after the last image change. A new `viewport-tiles` endpoint returns ready tiles or previews in one binary block; the matrix uses it instead of per-tile requests.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy/PostgreSQL, Pillow, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-matrix-tile-blocks-design.md`

## Global Constraints

- Tile root: `<ProjectWorkspaceBinding.source_project_dir>/tile`; without a binding: the existing persistent root from `tile_root_for(owner)` with the same layout below it.
- Layout: `tile/previews-256/<sha[:2]>/<sha>.png`, `tile/sets/<layer_id>/<image_representation_id>/{current, complete-<TILE_FORMAT>, <token>/<orientation>/<x1>_<y1>_<x2>_<y2>.png}`.
- Tile set key: `(project_id, layer_id, image_representation_id)` — the vector representation never takes part.
- Debounce: 30 s after the last image event of a project.
- Block: at most 512 items; media type `application/x-kraken-tile-block`.
- Keep `matrix_tile_timing` events working; add `preview_pass` and `compose_pass` events.
- Run `graphify update .` after code changes.

## Review Focus

- Two layers whose frames share image digests: one preview file, both sets use it.
- A frame whose source raises any exception: skipped, logged, the build finishes and writes the marker.
- Layer archived during the 30 s wait: no build starts for it.
- Server restart with a complete set on disk: the first read serves tiles and starts no build.
- Block request against an old server (404): the client falls back to per-tile requests and still draws.

---

### Task 1: Project tile layout and shared previews

**Files:**
- Modify: `src/kraken_core/frame_matrix/prepared_tiles.py`
- Modify: `src/kraken_server/persistent_services_parts/matrix_viewport.py` (`_prepared_pyramid`, `_schedule_matrix_tiles`, `matrix_mosaic`)
- Modify: `src/kraken_hub/composition/parts/matrix_viewport.py` (local `_prepared_pyramid`)
- Test: `tests/test_matrix_tile_layout.py`

**Interfaces:**
- Produces: `TileLayout(root: Path)` with `.previews_dir -> Path` and `.set_dir(layer_id: str, image_representation_id: str) -> Path`; `tile_layout_for(owner, project_id: str) -> TileLayout` (binding's `source_project_dir / "tile"`, else `tile_root_for(owner)`); `PreparedTilePyramid(directory, load_index, read_image, on_ready=None, *, previews_dir: Path | None = None)` (default `directory / "previews-256"` keeps today's behaviour for callers that pass none); `schedule_key(project_id, layer_id, image_representation_id) -> tuple[str, str, str]`; `image_representation_for(owner, project_id, layer_id, representation_ids) -> str` (the IMAGE kind among ids, else the layer's active image representation).

- [ ] Test `test_layout_lives_in_project_source_folder`: a server stub whose `workspace_files.registry.get_project(pid)` returns a binding with `source_project_dir=tmp/src/P` → `tile_layout_for(stub, pid).root == tmp/src/P/tile`, `set_dir("L","I") == root/"sets"/"L"/"I"`, `previews_dir == root/"previews-256"`.
- [ ] Test `test_layout_without_binding_uses_persistent_root`: binding `None` → root is `tile_root_for(stub)`.
- [ ] Test `test_two_sets_share_one_preview`: two pyramids with different set dirs, same `previews_dir`, one cell with digest `"ab12"` each; `read_image` counts calls → after both `build_now()` it was called once and `previews_dir/"ab"/"ab12.png"` exists.
- [ ] Test `test_vector_choice_does_not_change_the_set`: `_prepared_pyramid(pid, lid, [image_id, vector_a])` is `_prepared_pyramid(pid, lid, [image_id, vector_b])`.
- [ ] Implement; move preview path to `previews_dir / safe[:2] / f"{safe}.png"`; when the new preview is missing and an old `matrix-tiles/*/previews-256/<safe[:2]>/<safe>` exists, copy it instead of re-reading the source.
- [ ] Run `pytest tests/test_matrix_tile_layout.py tests/test_matrix_mosaic.py tests/test_matrix_tile_timing.py -q`: PASS. Commit.

### Task 2: Preview pass, compose pass, urgent first, robust reads

**Files:**
- Modify: `src/kraken_core/frame_matrix/prepared_tiles.py` (`_build`, `_drain_urgent`, `_write_tile`, `_preview_with_source`, `read`, `ensure_started`)
- Test: `tests/test_matrix_tile_build.py`

**Interfaces:**
- Consumes: Task 1 `previews_dir`.
- Produces: `PreparedTilePyramid._preview_pass(generation: int) -> bool` (False when superseded); timing events `preview_pass` (`frames`, `read`, `cached`, `skipped`, `ms`) and `compose_pass` (`tiles`, `ms`).

- [ ] Test `test_urgent_tile_is_served_between_preview_frames`: 200 cells, `read_image` sleeps 5 ms and, on its 10th call, triggers `pyramid.read(...)` of a 1-frame-wide 4×4 rect in another thread; after `build_now()` the urgent tile's `queue_wait_ms` (from the timing log) is `< 200`.
- [ ] Test `test_unreadable_frame_is_skipped`: `read_image` raises `AttributeError` for cell (2,1) → `build_now()` returns, the marker exists, the log has one warning naming (2,1).
- [ ] Test `test_dead_builder_restarts_on_read`: simulate `_thread` dead with `_urgent` non-empty → `read(...)` raises `MatrixTileNotReady` and a new thread is started (`pyramid._thread.is_alive()` or finished with the tile on disk).
- [ ] Test `test_complete_set_read_starts_no_build`: after `build_now()`, a fresh `PreparedTilePyramid` on the same dirs `read()` of a written tile returns bytes and `_thread is None`.
- [ ] Implement: `_build` = index → `_preview_pass` (drain urgent after every frame) → compose tiles from previews (drain after every tile) → marker. `read_image` errors of any `Exception` are caught per cell in `_preview_with_source`.
- [ ] Run tests above + `tests/test_matrix_mosaic.py`: PASS. Commit.

### Task 3: Build 30 s after the last image change

**Files:**
- Create: `src/kraken_server/tile_scheduler.py`
- Modify: `src/kraken_server/outbox.py` (`OutboxPublisher(..., listeners: Sequence[Callable[[str, str, str], None]] = ())`, called per row with `(project_id, event_type, stream_id)` after publishing; listener errors are logged, never stop publishing)
- Modify: `src/kraken_server/composition.py` (create `TileScheduler(services)` and pass `listeners=(scheduler.on_event,)`)
- Modify: `src/kraken_server/persistent_services_parts/matrix_viewport.py` (`matrix_viewport` no longer calls `_schedule_matrix_tiles`; `matrix_mosaic` read path starts a build only when the set is incomplete)
- Test: `tests/test_tile_scheduler.py`

**Interfaces:**
- Produces: `TileScheduler(services, *, delay_seconds: float = 30.0, clock=time.monotonic, timer_factory=threading.Timer)`; `.on_event(project_id: str, event_type: str, stream_id: str) -> None`; `.flush_due() -> None` (for tests); `IMAGE_EVENTS = frozenset({"LayerCreated", "LayerFilesImported", "ArtifactVersionCreated", "ArtifactVersionImported", "ArtifactVersionActivated", "RepresentationActivated", "RepresentationCreated"})`.

- [ ] Test `test_build_waits_for_quiet_period`: fake timer; events at t=0 and t=20 → nothing at t=45, build requested at t=50 for each active layer × image representation.
- [ ] Test `test_archived_layer_is_not_built`: `LayerArchived` for layer L before the timer fires → no build for L, others still built.
- [ ] Test `test_unchanged_stamp_builds_nothing`: second quiet period with the same content stamp → `note_content_stamp` does not restart (no new thread).
- [ ] Test `test_listener_error_does_not_stop_outbox`: a listener raising → `publish_once()` still marks rows published.
- [ ] Implement; run tests + `tests/test_matrix_mosaic.py`: PASS. Commit.

### Task 4: Tile block encoding and `viewport-tiles` endpoint

**Files:**
- Create: `src/kraken_core/frame_matrix/tile_block.py`
- Modify: `src/kraken_server/persistent_services_parts/matrix_viewport.py` (`matrix_tile_block`)
- Modify: `src/kraken_server/interface/http/routes/project_reads.py` (`POST {API_PREFIX}/projects/{project_id}/viewport-tiles`)
- Modify: `src/kraken_hub/application/workspace.py` (declare `matrix_tile_block` and `matrix_mosaic` on `ProjectWorkspaceService`)
- Test: `tests/test_tile_block.py`

**Interfaces:**
- Produces: `TileBlockItem(x1, y1, x2, y2, payload: bytes)`; `encode_tile_block(revision: str, items: Sequence[TileBlockItem], missing: Sequence[tuple[int, int, int, int]]) -> bytes`; `decode_tile_block(data: bytes) -> tuple[str, tuple[TileBlockItem, ...], tuple[tuple[int, int, int, int], ...]]`; `BLOCK_MEDIA_TYPE = "application/x-kraken-tile-block"`; `MAX_BLOCK_ITEMS = 512`; service `matrix_tile_block(project_id, *, layer_id, image_representation_id, orientation, kind: Literal["tiles", "previews"], cells: int, x1, y1, x2, y2) -> bytes`.

- [ ] Test `test_block_round_trip`: two items + one missing → decode returns them equal.
- [ ] Test `test_tiles_block_lists_ready_and_missing`: built pyramid 32×32 (16-cell level ready), request `kind="tiles", cells=16` over 1..32 → 4 items for y_down, 0 missing; with one tile file deleted → 3 items, that rect in `missing`, and it is queued as urgent.
- [ ] Test `test_previews_block_returns_one_item_per_frame`: rect 1..3×1 with previews for (1,1),(3,1) → 2 items, (2,1) missing.
- [ ] Test `test_block_rejects_too_many_items`: a range above 512 items → HTTP 422.
- [ ] Test `test_hub_architecture` passes (operations now declared).
- [ ] Implement; run tests + `tests/test_hub_architecture.py`: PASS. Commit.

### Task 5: Client uses blocks, previews at close zoom, sane backoff

**Files:**
- Modify: `src/kraken_hub/infrastructure/remote/parts/remove_layer_pipeline_action.py` (`matrix_tile_block` HTTP call; 404 → `TileBlockUnsupported`)
- Modify: `src/kraken_hub/composition/parts/matrix_viewport.py` (local `matrix_tile_block`)
- Modify: `src/kraken_hub/presentation/qt/matrix_source.py` (`KrakenMatrixDataSource.load_tile_block(...) -> tuple[str, tuple[TileBlockItem, ...], tuple[...]]`)
- Modify: `src/kraken_core/frame_matrix/qt_parts/apply_viewport_result.py` (`_request_mosaics`: one block future per level and screen; close zoom requests `previews` and stores them as cell thumbnails)
- Modify: `src/kraken_core/frame_matrix/qt_parts/poll_background_results.py` (apply block items to the cache under today's `ThumbnailKey`s; backoff doubles once per retry round)
- Test: `tests/test_matrix_tile_blocks_client.py`

**Interfaces:**
- Consumes: Task 4 `decode_tile_block`, `matrix_tile_block`.
- Produces: `MatrixDataSource` may expose `load_tile_block(*, kind, cells, x1, y1, x2, y2, orientation)`; the widget prefers it over `load_mosaic`.

- [ ] Test `test_one_request_per_screen`: fake data source counting calls; a viewport needing 31 tiles → exactly one `load_tile_block` call, 31 mosaics in `_mosaic_images` after polling.
- [ ] Test `test_close_zoom_uses_previews`: zoom to `3.4` → `load_tile_block(kind="previews", ...)` and visible cells get `thumbnail` set.
- [ ] Test `test_old_server_falls_back`: `load_tile_block` raises `TileBlockUnsupported` → `load_mosaic` used for each tile, later screens do not try blocks again.
- [ ] Test `test_backoff_doubles_once_per_round`: a round with 6 missing tiles → next delay 1500 ms, not 48000.
- [ ] Implement; run tests + `tests/test_manager_qt_matrix.py tests/test_matrix_mosaic.py`: PASS. Commit.

### Task 6: Verify end to end

- [ ] Full test runs: `pytest tests -n 6 -q` and `pytest plugins/contour/tests/unit -m "" -n 6 -q`; compare failures with the known pre-existing list (sqlite `index_metadata`, multilayer dialog timeouts, contour `is_complete_cif` fixtures).
- [ ] Repeat the user's zoom scenario with `matrix_tile_timing` on; expected: `tile_call` lines replaced by one block per screen, `missing` empty once the set is built, no urgent `queue_wait_ms` above one tile's compose time.
- [ ] `graphify update .`
