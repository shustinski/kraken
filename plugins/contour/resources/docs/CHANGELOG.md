# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: Replaced the parallel via detector zoo (gradient / spot / Hough
  / components / contours / morphology / template / blob) with a single
  pipeline: multi-scale TopHat/BlackHat + LoG blob response, ring template
  matching on the gradient, circle-fit verification (radial contrast + Sobel
  edge ring + angular coverage), optional user-template boost, and IoU-based
  NMS. Auto-polarity: if both white and black ranges are disabled, both
  polarities are scanned automatically.
- **BREAKING** (settings): removed `via_detector_*_enabled` flags and all the
  detector-specific thresholds (`via_gradient_min_*`, `via_spot_min_*`,
  `via_hough_*`, `via_component_min_score`, `via_contour_min_score`,
  `via_morphology_peak_scale`, `via_blob_min_circularity`,
  `via_auto_threshold_*`, and the legacy `via_white_threshold*` /
  `via_black_threshold*` / `via_threshold_range_*`). Added unified
  `via_min_score` (0..1), `via_min_contrast` (0..255) and
  `via_min_edge_coverage` (0..1). Old keys are still read from saved
  configurations but ignored.
- **BREAKING** (UI): removed the eight detector checkboxes and their per-detector
  threshold spin boxes; added `Min score`, `Min contrast` and
  `Min edge coverage` controls; the two built-in presets
  (`Bright vias on traces`, `Weak/blurred vias`) now operate on the unified
  fields.
- Decomposed the `PolygonExtractionWidget` god-class: extracted UI builders,
  retranslation, pipeline list widget, toolbar icons, compact stylesheet, via
  preset payloads, and i18n content into dedicated modules under
  `contour/ui/`. `widget.py` shrunk from ~7 100 to ~3 800 lines (~46 %
  reduction) while keeping a stable public API guarded by a smoke test and a
  golden snapshot.
- Split `contour/graphics_view.py` into a `contour/graphics/`
  package (`editor_scene`, `editor_view`, `tools`, `geometry`) with a backward
  compatible shim.
- Split `contour/application/use_cases/processing.py` into a package
  with a `_core` implementation and facade submodules (`requests`, `preview`,
  `binarization`, `via_detection`).
- Extracted reusable services: `application/services/batch_controller.py`,
  `application/services/dataset_exporter.py`,
  `application/services/pipeline_controller.py`,
  `application/services/preview_orchestrator.py`.
- Tightened mypy: removed the project-wide `disable_error_code` and narrowed
  the soft baseline to the modules that interop heavily with PyQt6/OpenCV.
  Domain, services, and infrastructure modules now run with
  `warn_return_any = true` and `strict_optional = true`.

### Fixed
- Via detection accuracy: the old implementation combined ten detectors with
  incompatible score scales, which let the gradient detector dominate and the
  distance-only NMS merged genuinely separate neighbouring vias. The new
  single-pipeline detector uses one unified 0..1 score, IoU-based NMS, and a
  consistent circle-fit verification on both polarities.
- Auto-apply pipeline no longer re-runs processing when the "auto apply"
  checkbox is unchecked (the characterisation test used to be
  `@expectedFailure` — now passes).

### Added
- `tests/golden/widget_public_api.txt` — golden snapshot of the widget's
  public signals and methods; enforced by `tests/unit/test_widget_smoke.py`.
- `tests/integration/test_app_startup.py` — end-to-end bootstrap smoke test.
- `tests/unit/test_services.py` — unit tests for the new service layer.

## [0.4.0] — 2026-04-23

### Added
- Single source of truth for the version at `contour/__version__.py`.
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
  `contour/__version__.py` and declares a `build` extras group for
  PyInstaller.
- `packaging/ViaLaNetContour.iss` accepts `MyAppVersion` as an ISCC
  preprocessor argument (`/DMyAppVersion=...`) with a sensible default.
- Example integration script uses the logging module instead of `print`.

### Removed
- Old installer artefacts (`packaging/*.exe`) and the stray `frame_1.cif` in
  the repository root.
- `contour.egg-info/` that was accidentally committed.

[Unreleased]: https://github.com/shustinski/ViaLaNet/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/shustinski/ViaLaNet/releases/tag/v0.4.0
