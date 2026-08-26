# Cartograph

Cartograph is a Kraken Qt plugin for navigating large SEM frame arrays and
stitching a local 3×3 neighborhood without building a global mosaic.

v1 is a vertical slice: load a tile grid, place tiles nominally, register the
12-edge 4-neighborhood of a 3×3 window, optimize translations, persist
diagnostics, and render a local mosaic.

## Run

```powershell
cd plugins\cartograph
uv sync --extra dev
uv run python -m cartograph
```

Headless 3×3 slice:

```powershell
uv run python -m cartograph --grid path\to\tiles --center-row 1 --center-col 1 --output mosaic.png --diagnostics block.json
```

Filenames follow Kraken import mapping ``x_y`` (1-based column_row), or a
``grid.json`` manifest. Optional stage coordinates belong in the manifest or a
``.stage.json`` sidecar.

## Tests

```powershell
uv run pytest
```
