# ViaLaNet Polygon Widget

[![CI](https://github.com/shustinski/ViaLaNet/actions/workflows/ci.yml/badge.svg)](https://github.com/shustinski/ViaLaNet/actions/workflows/ci.yml)
[![Release](https://github.com/shustinski/ViaLaNet/actions/workflows/release.yml/badge.svg)](https://github.com/shustinski/ViaLaNet/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

Standalone PyQt6 application and embeddable widget for polygon extraction,
editing, and export from microscope imagery and similar grayscale inputs.

## Features

- Pluggable image processing pipeline (binary / edge / gradient / via
  detectors) with live preview.
- Contour extraction with profile-specific tuning (general contours, via
  boxes, structural elements) and an auto-tuner.
- Interactive editor: move / add / remove polygon vertices, split, merge,
  ruler with 45° snapping.
- Dataset export (masks, labels, CIF overlays).
- Bilingual UI (Russian / English) driven by the `--language` flag or the
  system locale.
- Distributed as a signed-capable Windows installer built by PyInstaller +
  Inno Setup.

## Requirements

- Windows 10 or newer (for the installer build). The widget itself is
  cross-platform but only Windows is CI-tested.
- Python 3.14 for development.
- Optional: Inno Setup 6+ (`iscc` on `PATH`, or installed under
  `Program Files\Inno Setup 7`).

## Installation

### End user — installer

Download the latest `Contour-setup-<version>.exe` from the
[Releases](https://github.com/shustinski/ViaLaNet/releases) page and run it.
The installer registers a Start Menu entry and optional desktop shortcut.

Logs are written to `%LOCALAPPDATA%\ViaLaNet\Contour\logs\app.log`.

### Developer — editable install

From the repo root with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --project plugins/contour --extra dev --extra build
uv run --project plugins/contour python -m contour
```

Or with a classic venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,build]"
pre-commit install
```

## Running

```powershell
contour                                  # installed entry point
python main.py                           # from a source checkout
python -m contour                 # as a module
```

### CLI options

| Flag | Description |
|------|-------------|
| `paths` | Positional: image files or a single directory to load at startup. |
| `--input-dir PATH` | Input image directory. |
| `--output-dir PATH` | Output directory for exported results. |
| `--cif-dir PATH` | Directory with CIF overlays. |
| `--pipeline-json PATH` | Path to pipeline JSON config. |
| `--language {ru,en}` | UI language override. |
| `--width INT` / `--height INT` | Initial window size. |
| `--no-qss` | Disable the bundled QSS theme. |
| `--verbose`, `-v` | Enable DEBUG logging. |
| `--log-file PATH` | Override the log file location. |
| `--version` | Print version and exit. |

### Kraken Agent protocol v1

Agent supplies `KRAKEN_JOB_MANIFEST`, `KRAKEN_RESULT_MANIFEST`, and
`KRAKEN_STAGING_ROOT`; equivalent CLI flags use the corresponding lower-case
`--kraken-*` names. In this mode Contour opens only the exact input list from
the manifest and saves CIF files only under `<staging>/outputs`.

After saving the desired CIF files, use **Kraken → Вернуть результаты в
Kraken** (`Ctrl+Shift+Return`). Only this explicit action writes the result
manifest. Closing Contour without it does not import or alter project data.
Managed V1 jobs require unique input filename stems because the existing
Contour exporter maps each image stem to the matching CIF stem. Symlinks,
paths outside staging, and SHA-256 mismatches are rejected.

## Embedding the widget

`PolygonExtractionWidget` can be hosted inside any PyQt6 application. See
[`examples/contour_integration.py`](examples/contour_integration.py).

## Development

Run the quality gates:

```powershell
ruff check contour tests examples
ruff format --check contour tests examples
mypy contour
$env:QT_QPA_PLATFORM = "offscreen"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full developer workflow and
release process.

### Project layout

```
contour/
  adapters/qt/           Qt-specific adapters (image conversion, preview)
  application/           CLI, bootstrap, model/presenter/view
    services/            Batch / dataset / pipeline / preview / workspace services
    use_cases/           Image processing use cases (preview, processing, autotune)
  domain/                Pure geometry / polygon primitives
  graphics/              Editor scene / view, drawing tools, pure geometry helpers
  infrastructure/        Logging, persisted settings
  resources/             QSS themes, icons
  ui/                    UI builders, retranslate, icons, presets, styles
  widget.py              Top-level PolygonExtractionWidget (composition root)
tests/unit/              Unit tests
tests/integration/       Bootstrap / end-to-end tests
tests/golden/            Golden snapshots (public API surface)
packaging/               PyInstaller spec and Inno Setup script
scripts/                 Build helpers (PowerShell)
```

## Building the Windows installer

Requirements:

- [uv](https://docs.astral.sh/uv/) workspace synced with build extras.
- [Inno Setup 6+](https://jrsoftware.org/isinfo.php) (`iscc` on `PATH`, or the
  default install path such as `C:\Program Files\Inno Setup 7`).

```powershell
cd D:\code\kraken\plugins\contour
.\scripts\build_windows.ps1                 # uv sync + tests + PyInstaller + Inno Setup
.\scripts\build_windows.ps1 -SkipTests      # quick iteration
.\scripts\build_windows.ps1 -SkipSync       # reuse current uv env
.\scripts\build_windows.ps1 -Clean          # remove previous dist/build output first
.\scripts\build_windows.ps1 -PyInstallerOnly # bundle only, no installer
.\scripts\build_windows.ps1 -InstallerOnly   # rebuild installer from existing dist
.\scripts\build_windows.ps1 -Version 0.9.6  # override installer version label
```

The script runs `uv sync --project plugins/contour --extra build --extra dev`
from the repo root, then invokes pytest and PyInstaller through `uv run`.

Outputs:

- application bundle: `dist/Contour/`
- installer: `dist/installer/Contour-setup-<version>.exe`

## License

[MIT](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for release history and
[SECURITY.md](SECURITY.md) for the responsible disclosure policy.
