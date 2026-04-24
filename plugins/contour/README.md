# ViaLaNet Polygon Widget

[![CI](https://github.com/shustinski/ViaLaNet/actions/workflows/ci.yml/badge.svg)](https://github.com/shustinski/ViaLaNet/actions/workflows/ci.yml)
[![Release](https://github.com/shustinski/ViaLaNet/actions/workflows/release.yml/badge.svg)](https://github.com/shustinski/ViaLaNet/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

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
- Python 3.13 for development.
- Optional: Inno Setup 6 (`iscc` on `PATH`) to build the installer.

## Installation

### End user — installer

Download the latest `Contour-setup-<version>.exe` from the
[Releases](https://github.com/shustinski/ViaLaNet/releases) page and run it.
The installer registers a Start Menu entry and optional desktop shortcut.

Logs are written to `%LOCALAPPDATA%\ViaLaNet\Contour\logs\app.log`.

### Developer — editable install

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

```powershell
.\scripts\build_windows.ps1            # lint + tests + installer
.\scripts\build_windows.ps1 -SkipTests # quick iteration
```

The script produces `packaging/Contour-setup-<version>.exe`.

## License

[MIT](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for release history and
[SECURITY.md](SECURITY.md) for the responsible disclosure policy.
