"""Shared pytest fixtures for the ViaLaNet Polygon Widget test suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ``pytest`` is the everyday behavioral suite.  Algorithmic image processing,
# geometry/vector output, benchmarks, and broad GUI regression tests stay in
# the explicit ``full`` suite run by the Windows build.
FAST_TEST_FILES = frozenset(
    {
        "test_app_startup.py",
        "test_application_window_sizing.py",
        "test_batch_processor.py",
        "test_contact_feedback.py",
        "test_frame_asset_sync.py",
        "test_frame_drop.py",
        "test_frame_layers.py",
        "test_frame_path_list_model.py",
        "test_frame_prefetch.py",
        "test_i18n.py",
        "test_logging.py",
        "test_processing_use_cases.py",
        "test_processing_v2.py",
        "test_qt_object_validity.py",
        "test_recognition_modes.py",
        "test_runtime_config.py",
        "test_services.py",
        "test_settings_store.py",
        "test_startup_profiler.py",
        "test_status_list_delegate.py",
        "test_styles.py",
        "test_tool_mode_logic.py",
        "test_transition_save_guard.py",
        "test_updater.py",
        "test_utils.py",
        "test_vector_index_controller.py",
        "test_widget_refactor_boundaries.py",
        "test_widget_smoke.py",
        "test_workspace_session.py",
        "test_workspace_use_cases.py",
    }
)

VECTORIZATION_TEST_FILES = frozenset(
    {
        "test_antialias_cif_job.py",
        "test_brush_vector.py",
        "test_cif_klayout_reader.py",
        "test_cif_via_support.py",
        "test_contour_extractor_filters.py",
        "test_deferred_geometry_validation.py",
        "test_fix_internal_contours.py",
        "test_geometry.py",
        "test_keyhole_cut_display.py",
        "test_metal_benchmark_evaluation_crop.py",
        "test_metal_benchmark_metrics.py",
        "test_metal_golden.py",
        "test_metal_recovery.py",
        "test_new_metal_strategies.py",
        "test_polygon_antialiasing.py",
        "test_polygon_creation.py",
        "test_polygon_offset.py",
        "test_structural_watershed.py",
        "test_vector_geometry_postprocess.py",
        "test_via_regression.py",
        "test_vision_sem_mask.py",
        "test_watershed_hierarchy_diagnostic.py",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign every test to full and selected stable behavior tests to fast."""
    for item in items:
        filename = item.path.name
        item.add_marker(pytest.mark.full)
        if filename in FAST_TEST_FILES:
            item.add_marker(pytest.mark.fast)
        if filename in VECTORIZATION_TEST_FILES:
            item.add_marker(pytest.mark.vectorization)


@pytest.fixture(autouse=True)
def _isolated_settings_dir(tmp_path) -> Iterator[None]:
    previous = os.environ.get("VIALANET_SETTINGS_DIR")
    os.environ["VIALANET_SETTINGS_DIR"] = str(tmp_path / "contour-settings")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VIALANET_SETTINGS_DIR", None)
        else:
            os.environ["VIALANET_SETTINGS_DIR"] = previous


@pytest.fixture(autouse=True)
def _restore_profiling_environment() -> Iterator[None]:
    """Prevent profiling switch mutations from leaking between tests."""
    previous = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("CONTOUR_PROFILE") or name == "CONTOUR_PROFILING"
    }
    try:
        yield
    finally:
        for name in list(os.environ):
            if name.startswith("CONTOUR_PROFILE") or name == "CONTOUR_PROFILING":
                os.environ.pop(name, None)
        os.environ.update(previous)


@pytest.fixture(scope="session", autouse=True)
def _qt_application() -> Iterator[object]:
    """Provide a single ``QApplication`` instance for the entire test session.

    Many widget tests instantiate Qt objects directly. Creating a single
    application up-front avoids ``QWidget: Must construct a QApplication``
    errors and keeps tests fast.
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app
    app.processEvents()
