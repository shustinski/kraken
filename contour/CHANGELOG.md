# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] — 2026-04-23

### Added
- Single source of truth for the version at `polygon_widget/__version__.py`.
- Application logging with a rotating file handler under
  `%LOCALAPPDATA%/ViaLaNet/Contour/logs/app.log` and a console handler.
- Global `sys.excepthook` that writes the traceback to the log and shows a
  user-friendly crash dialog.
- CLI flags `--verbose` / `--log-file` / `--version`.
- `ruff`, `mypy`, `pytest`, `pytest-qt`, `pytest-cov`, `pre-commit` wired up via
  `pyproject.toml` with a `dev` extras group.
- `.pre-commit-config.yaml` with ruff, mypy, trailing whitespace, EOF, large
  file, and line-ending hooks.
- GitHub Actions CI (`.github/workflows/ci.yml`) running lint, type check and
  test suite on every push / PR.
- Release workflow (`.github/workflows/release.yml`) that builds the
  PyInstaller bundle, compiles the Inno Setup installer, and uploads it to the
  GitHub Release on tag push (`v*`).
- Reproducible local build script `scripts/build_windows.ps1`.
- `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`.
- `.gitattributes` normalising line endings across the repository.

### Changed
- `pyproject.toml` now derives the package version dynamically from
  `polygon_widget/__version__.py` and declares a `build` extras group for
  PyInstaller.
- `packaging/ViaLaNetPolygonWidget.iss` accepts `MyAppVersion` as an ISCC
  preprocessor argument (`/DMyAppVersion=...`) with a sensible default.
- Example integration script uses the logging module instead of `print`.

### Removed
- Old installer artefacts (`packaging/*.exe`) and the stray `frame_1.cif` in
  the repository root.
- `vialanet_polygon_widget.egg-info/` that was accidentally committed.

[Unreleased]: https://github.com/shustinski/ViaLaNet/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/shustinski/ViaLaNet/releases/tag/v0.4.0
