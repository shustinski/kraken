# Shared frame matrix for Kraken layers and Contour

## Intent

Contour's frame matrix (`contour.ui.frame_matrix_view.FrameMatrixGraphicsView`) is a
separate QGraphicsView implementation. It must become a subclass of the same class as
the Kraken layer matrix (`kraken_core.frame_matrix.qt.FrameMatrixWidget`), so both run
on one engine. The two matrices differ only in which frame actions they offer.

## Goals

- One rendering, LOD, zoom, selection and thumbnail engine (`FrameMatrixWidget`).
- Frame actions are declared by the matrix class itself. The base class builds the
  context menu from that declaration and reports the chosen action.
- Kraken layer matrix: "Выгрузить файлы…", "Выдать работу…", "Свойства вектора…",
  "Свойства кадра…" (same visibility rules as today).
- Contour matrix: "Удалить кадр", "Очистить кадр".
- Contour keeps rectangular cells, frame overlap and pyramid-LOD thumbnails.

## Non-goals

- No change to Kraken data sources, server, or thumbnail storage formats.
- Contour callers outside the matrix are not rewritten in this change: they keep the
  `QListWidget`-like API (`count`, `item`, `addItem`, `setCurrentRow`, …), which the
  new class implements on top of the engine.
- Deleting Contour's now-unused icon-loading pipeline is a follow-up.

## Approaches considered

1. **Full migration (chosen).** Contour matrix subclasses `FrameMatrixWidget`; frames
   reach it through a `MatrixDataSource`/`MatrixAssetSource` pair over Contour's list of
   frames and its `PyramidFrameStore`.
2. Thin shared base with signals and actions, separate renderers. Rejected: the two
   matrices would still not be the same matrix.

## Design

### Base class (`kraken_core.frame_matrix`)

- **Cell geometry.** `FrameMatrixView` gets per-axis geometry: cell width/height and
  overlap (`set_cell_geometry(width, height, overlap_x=0, overlap_y=0)`). Every place
  that used the square `CELL_SIZE`/`CELL_PITCH` reads the per-axis values. Defaults
  keep Kraken's 48×48 cells with no overlap, so Kraken behaviour is unchanged.
- **Actions.** New `FrameMatrixAction(id, title, is_visible(context) -> bool)`.
  `FrameMatrixView.frame_actions(context)` returns the actions for a context (default:
  none). `contextMenuEvent` builds a `QMenu` from them and emits
  `frameActionTriggered(action_id, context)`. With no actions it emits the existing
  `contextMenuRequested` signal, so other users keep working.

### Kraken layer matrix (`kraken_hub`)

`LayerFrameMatrixWidget(FrameMatrixWidget)` declares the four actions. Visibility
predicates that depend on the app (permission to assign work, the vector file of the
frame) are supplied by the host through a small callback object. The hub connects
`frameActionTriggered` to the existing handlers.

### Contour matrix (`contour.ui.frame_matrix_view`)

`FrameMatrixGraphicsView(FrameMatrixWidget)` (name kept for callers) declares
"delete_frame" and "clear_frame" and keeps the `QListWidget` subset Contour uses.
List index `i` maps to cell `(i % columns + 1, i // columns + 1)`.

- `ContourMatrixDataSource`: an immutable snapshot of the frame list (status, label,
  hidden state, asset reference) rebuilt when items change; a new revision is pushed
  through `set_session`.
- `ContourFrameAssetSource`: returns PNG bytes for a frame at the requested size, from
  the `PyramidFrameStore` level closest to that size when one is set, otherwise from
  the image file.
- In-memory thumbnail store.
- Signals Contour relies on stay: `itemClicked`, `frameNavigationRequested`,
  `thumbnailLodChanged`. `frameContextMenuRequested` is replaced by
  `frameActionTriggered`.
- Contour's own icon pipeline no longer feeds the matrix.

## Left to the builder

Exact visual details (status marker, label placement) follow the Kraken renderer;
matching Contour's old pixel look is not required.
