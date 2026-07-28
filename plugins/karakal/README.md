# Karakal

Karakal is a Kraken Qt plugin and standalone development app for inspecting frame matrices, comparing model outputs, selecting validation samples, and checking regular cell-grid defects.

The plugin is focused on image/model-result review workflows:

- build a frame matrix from model result folders;
- compare masks, confidence maps, and configured model pairs;
- inspect individual frames in a detailed viewer;
- mark and export validation or defect-check results;
- run a cell-defect inspection mode with percentile views and exportable check masks;
- support manager workflows for primary-labeling sample selection.

## Status

Karakal is an internal Kraken plugin that can also be launched directly for development and packaging. The public API is intentionally small; most user-facing behavior is exposed through the Qt interface.

## Requirements

- Python 3.13 or newer.
- Windows or Linux.
- Python dependencies from `pyproject.toml`:
  - `PyQt6`
  - `numpy`
  - `opencv-python`
  - `scipy`
- Optional build tools:
  - `pyinstaller` for executable builds;
  - Inno Setup on Windows for the installer.

## Installation

From the repository root, install the workspace dependencies with `uv`:

```bash
uv sync --all-extras
```

For direct plugin development from the plugin folder:

```bash
cd plugins/karakal
python -m pip install -e ".[dev,build]"
```

## Running

As a standalone development app:

```bash
cd plugins/karakal
python -m karakal
```

Through Kraken, the plugin metadata is stored in `resources/plugin.json`; Kraken launches the plugin with:

```bash
python -m karakal
```

## Typical Workflow

1. Add one or more model output folders.
2. Optionally select source frames and ground-truth markup folders.
3. Build the frame matrix.
4. Compute analytics.
5. Inspect frames in validation mode, manager mode, or cell-defect mode.
6. Use percentile views, good/bad frame groups, or selected matrix regions to focus review/export.
7. Export attached frame assets, result layers, or cell-defect check images.

## Input Data

Karakal expects image folders with matching frame names or frame indices. Supported image formats are handled through Qt/OpenCV and include common raster formats such as PNG, JPG, BMP, TIFF, and WebP where the runtime supports them.

Common inputs:

- model mask folders;
- optional model confidence folders;
- optional original source frame folder;
- optional ground-truth markup folder.

Nested frame folders are ignored by default when building a matrix from a selected folder.

## Output Data

Depending on the selected command, Karakal can export:

- attached files for selected frame records;
- result-layer JPG exports;
- cell-defect check images in BMP, PNG, or JPG;
- export manifests in JSON;
- benchmark JSON/CSV output under `build/`.

User exports and build artifacts should not be committed.

## Configuration

Karakal stores UI settings through Qt `QSettings`.

Environment variables supported by the codebase:

- `KARAKAL_CACHE_DIR`: override the default cache location.
- `KARAKAL_EXPORT_WORKERS`: limit export worker count.
- `KARAKAL_GRID_DEBUG_DIR`: write grid-inspection debug payloads.
- `KARAKAL_GRID_INSPECTION_EXECUTION`: `process`, `thread`, or `sequential`.
- `KARAKAL_GRID_INSPECTION_WORKERS`: grid-inspection worker count.
- `KARAKAL_GRID_INSPECTION_CHUNK_SIZE`: process-pool chunk size.
- `KARAKAL_GRID_INSPECTION_OPENCV_THREADS`: OpenCV threads per worker.
- `KARAKAL_GRID_INSPECTION_FRAME_TIMEOUT`: per-frame timeout for process execution.

See `.env.example` for safe local examples.

## Testing

From the repository root:

```bash
uv run pytest plugins/karakal/tests -q
uv run --extra dev ruff check plugins/karakal/src/karakal plugins/karakal/tests
```

For a fast syntax smoke check:

```bash
python -m py_compile plugins/karakal/src/karakal/app/main_window.py plugins/karakal/src/karakal/app/presenter.py
```

Qt tests should run with an offscreen platform in CI:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest plugins/karakal/tests -q
```

On Windows PowerShell:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest plugins/karakal/tests -q
```

## Benchmarks

Grid-inspection benchmark:

```bash
cd plugins/karakal
python benchmarks/benchmark_grid_inspection.py --limit 100 --implementation sequential
```

Comparison benchmark:

```bash
cd plugins/karakal
python -m karakal --benchmark-comparison
```

Benchmark outputs are written under `build/` and are ignored by Git.

## Building

Python package build:

```bash
cd plugins/karakal
python -m build
```

Windows executable build:

```powershell
cd plugins/karakal
pyinstaller karakal.spec
```

Windows installer with Inno Setup:

```powershell
cd plugins/karakal/packaging
iscc Karakal.iss
```

## Project Layout

```text
plugins/karakal/
  src/karakal/app/          Qt main window, presenter, UI state
  src/karakal/core/         frame collection, analytics, workers, grid inspection
  src/karakal/comparison/   pairwise and ensemble comparison engine
  src/karakal/ui/           reusable Qt widgets, details dialog, i18n
  src/karakal/plugin/       Kraken plugin adapter
  tests/unit/               unit and regression tests
  benchmarks/               reproducible performance scripts
  packaging/                installer configuration
  scripts/                  package build helper scripts
```

## Adding Modules

- Keep UI orchestration in `app/` and reusable widgets in `ui/`.
- Keep computational code in `core/` or `comparison/`.
- Do not run long computations in the Qt UI thread.
- Add focused regression tests for bug fixes and format-sensitive exports.
- Avoid new heavy dependencies unless the existing NumPy/OpenCV/PyQt stack cannot solve the problem.

## Known Limitations

- Some long exports are still coordinated from the presenter and may require further worker extraction for very large datasets.
- PyQt dynamic widget access is noisy for strict static type checkers.
- License is not declared in this repository yet: `[УКАЗАТЬ ЛИЦЕНЗИЮ]`.

## Screenshots

Screenshots are not committed with this plugin. Add release screenshots under project documentation when publication assets are ready.
