"""Smoke and characterization tests for :class:`PolygonExtractionWidget`.

These tests protect the public API surface of the top-level widget while the
codebase is being refactored. Changes to the public API must be intentional
and require updating ``tests/golden/widget_public_api.txt``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtBoundSignal
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QListWidgetItem, QWidget

from contour.graphics_view import EditorTool
from contour.widget import PolygonExtractionWidget

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "golden" / "widget_public_api.txt"


def _collect_public_api(widget: PolygonExtractionWidget) -> tuple[list[str], list[str]]:
    signals: list[str] = []
    for name in dir(widget):
        if name.startswith("_"):
            continue
        if isinstance(getattr(widget, name, None), pyqtBoundSignal):
            signals.append(name)
    signals = sorted(set(signals) - set(dir(QWidget)))

    qt_inherited = set(dir(QWidget))
    methods = [
        name
        for name in dir(type(widget))
        if not name.startswith("_")
        and callable(getattr(type(widget), name, None))
        and name not in qt_inherited
        and not isinstance(getattr(widget, name, None), pyqtBoundSignal)
    ]
    return sorted(set(signals)), sorted(set(methods))


def _parse_golden(text: str) -> tuple[list[str], list[str]]:
    sections: dict[str, list[str]] = {"signals": [], "methods": []}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            continue
        if current in sections:
            sections[current].append(line)
    return sorted(sections["signals"]), sorted(sections["methods"])


def _send_wheel(target: QWidget, delta: int = -120) -> None:
    center = target.rect().center()
    event = QWheelEvent(
        QPointF(center),
        target.mapToGlobal(QPointF(center)),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(target, event)


class WidgetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_widget_instantiates(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            self.assertIsNotNone(widget)
        finally:
            widget.close()
            widget.deleteLater()

    def test_widget_uses_compact_resizable_layout(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            self.assertLessEqual(widget.main_splitter.widget(0).minimumWidth(), 300)
            self.assertLessEqual(widget.right_tabs.minimumWidth(), 220)
            self.assertTrue(hasattr(widget, "editor_toolbar_scroll"))
            self.assertLessEqual(widget.editor_toolbar_scroll.minimumWidth(), 760)
            self.assertFalse(hasattr(widget, "visual_frame_nav_widget"))
        finally:
            widget.close()
            widget.deleteLater()

    def test_heuristic_via_detector_uses_automatic_preview_controls(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            self.assertFalse(hasattr(widget, "run_bright_via_button"))
            self.assertEqual(widget.heuristic_background_sigma_spin.value(), 25.0)
            settings = widget._current_contour_settings()
            self.assertEqual(settings.heuristic_background_sigma, 25.0)
            self.assertEqual(settings.via_search_mode, "heuristic")
            self.assertEqual(widget._contour_settings_profiles["vias"].via_search_mode, "heuristic")
        finally:
            widget.close()
            widget.deleteLater()

    def test_extraction_and_manual_hole_fill_areas_are_independent(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            widget.min_inner_hole_area_spin.setValue(42.0)
            widget.vector_geom_min_hole_spin.setValue(7.0)

            self.assertEqual(widget._current_contour_settings().min_inner_hole_area, 42.0)
            self.assertEqual(widget._vector_geometry_settings_from_widgets().min_hole_area_to_remove_px2, 7.0)
        finally:
            widget.close()
            widget.deleteLater()

    def test_conductor_display_defaults(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("conductors"))
            self.assertTrue(widget.metal_show_rejected_checkbox.isChecked())
            self.assertFalse(widget.metal_show_border_checkbox.isChecked())
            self.assertFalse(widget.metal_show_mask_checkbox.isChecked())
            self.assertEqual(widget.metal_segmentation_strategy_combo.currentData(), "legacy_otsu")
            self.assertFalse(widget._current_contour_settings().metal_use_wide_conductor_gradient)
            widget.metal_segmentation_strategy_combo.setCurrentIndex(
                widget.metal_segmentation_strategy_combo.findData("gradient_watershed")
            )
            self.assertTrue(widget._current_contour_settings().metal_use_wide_conductor_gradient)
            self.assertEqual(
                widget._current_contour_settings().metal_segmentation_strategy,
                "gradient_watershed",
            )
            widget.metal_segmentation_strategy_combo.setCurrentIndex(
                widget.metal_segmentation_strategy_combo.findData("random_walker")
            )
            self.assertEqual(
                widget._current_contour_settings().metal_segmentation_strategy,
                "random_walker",
            )
            self.assertFalse(widget._current_contour_settings().metal_use_wide_conductor_gradient)
            self.assertEqual(widget.metal_gap_bridge_spin.value(), 1)
            self.assertEqual(widget.metal_speckle_removal_spin.value(), 1)
            self.assertEqual(widget.metal_ws_smoothing_spin.value(), 1.0)
            self.assertEqual(widget.metal_ws_core_margin_spin.value(), 8.0)
            self.assertEqual(widget.metal_ws_groove_margin_spin.value(), 16.0)
            self.assertEqual(widget.metal_ws_rim_probe_spin.value(), 6)
            self.assertEqual(widget.metal_ws_seed_speckle_spin.value(), 1)
            self.assertEqual(widget.metal_ws_valley_span_spin.value(), 5)
            self.assertEqual(widget.metal_ws_valley_depth_spin.value(), 45.0)
            self.assertEqual(widget.metal_rw_beta_spin.value(), 90.0)
            self.assertEqual(widget.metal_rw_iterations_spin.value(), 160)
            self.assertEqual(widget.metal_gc_iterations_spin.value(), 5)
            self.assertEqual(widget.metal_recon_erode_spin.value(), 0)
            self.assertEqual(widget.metal_filter_group.title(), "Фильтрация распознанных")
            self.assertEqual(widget.metal_recognition_params_group.title(), "Параметры распознавания")
            self.assertEqual(widget.metal_watershed_group.title(), "Watershed")
            self.assertEqual(widget.metal_random_walker_group.title(), "Random Walker")
            self.assertEqual(widget.metal_graph_cut_group.title(), "Graph Cut")
            self.assertEqual(widget.metal_reconstruction_group.title(), "Reconstruction")
            widget.metal_rw_beta_spin.setValue(120.0)
            self.assertEqual(widget._current_contour_settings().metal_random_walker_beta, 120.0)
            widget.metal_ws_core_margin_spin.setValue(12.0)
            self.assertEqual(widget._current_contour_settings().metal_watershed_core_margin, 12.0)
            self.assertFalse(widget.metal_preset_widget.isVisible())
            self.assertFalse(widget.metal_preview_mask_button.isVisible())
            self.assertFalse(widget.contour_group.isVisible())
            self.assertGreaterEqual(widget.metal_debug_visual_combo.findData("metal_gradient_x"), 0)
            self.assertGreaterEqual(widget.metal_debug_visual_combo.findData("metal_gradient_y"), 0)
            self.assertGreaterEqual(widget.metal_debug_visual_combo.findData("metal_gradient_field"), 0)
            self.assertTrue(hasattr(widget, "metal_gradient_3d_button"))
            self.assertTrue(widget.metal_hierarchy_combo.currentData() == "full")
        finally:
            widget.close()
            widget.deleteLater()

    def test_combo_and_spin_ignore_mouse_wheel(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            for combo in widget.findChildren(QComboBox):
                if combo.count() < 2:
                    continue
                before = combo.currentIndex()
                _send_wheel(combo, -120)
                _send_wheel(combo, 120)
                self.assertEqual(combo.currentIndex(), before, combo.objectName() or combo.__class__.__name__)
            for spin in widget.findChildren(QAbstractSpinBox):
                before = spin.value()
                _send_wheel(spin, -120)
                _send_wheel(spin, 120)
                self.assertEqual(spin.value(), before, spin.objectName() or spin.__class__.__name__)
        finally:
            widget.close()
            widget.deleteLater()

    def test_conductor_recognition_locks_vector_edit_tools(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("conductors"))

            self.assertTrue(widget.polygon_editor.vector_edits_locked())
            self.assertFalse(widget._tool_buttons[EditorTool.SELECT].isHidden())
            self.assertFalse(widget._tool_buttons[EditorTool.PAN].isHidden())
            self.assertFalse(widget._tool_buttons[EditorTool.RULER].isHidden())
            for tool in (
                EditorTool.ADD_POLYGON,
                EditorTool.BRUSH,
                EditorTool.TRACE_PEN,
                EditorTool.ADD_VIA,
                EditorTool.ADD_VERTEX,
                EditorTool.DELETE_VERTEX,
                EditorTool.MOVE_VERTEX,
                EditorTool.ANTIALIAS,
            ):
                self.assertTrue(widget._tool_buttons[tool].isHidden())
            self.assertFalse(widget.undo_button.isEnabled())
            self.assertFalse(widget.redo_button.isEnabled())
            self.assertFalse(widget.antialias_opened_cif_button.isEnabled())

            widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("disabled"))
            self.assertFalse(widget.polygon_editor.vector_edits_locked())
            self.assertFalse(widget._tool_buttons[EditorTool.ADD_POLYGON].isHidden())
            self.assertTrue(widget.undo_button.isEnabled())
            self.assertTrue(widget.antialias_opened_cif_button.isEnabled())
        finally:
            widget.close()
            widget.deleteLater()

    def test_contour_extraction_group_is_never_shown(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            for mode in ("disabled", "conductors", "via"):
                widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData(mode))
                self.assertTrue(widget.contour_group.isHidden())
                parent_layout = widget.contour_group.parentWidget().layout() if widget.contour_group.parentWidget() else None
                self.assertTrue(parent_layout is None or parent_layout.indexOf(widget.contour_group) < 0)
                self.assertTrue(widget.advanced_extraction_checkbox.isHidden())
            widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
            self.assertFalse(widget.bright_via_group.isHidden())
        finally:
            widget.close()
            widget.deleteLater()

    def test_image_list_selection_selects_thumbnail_grid_item(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            paths = [r"d:\frames\a.png", r"d:\frames\b.png"]
            widget._workspace.replace_image_selection(paths, is_supported_image=lambda _path: True)
            for path in paths:
                image_item = QListWidgetItem(Path(path).stem)
                image_item.setData(Qt.ItemDataRole.UserRole, path)
                widget.image_list.addItem(image_item)

                thumbnail_item = QListWidgetItem()
                thumbnail_item.setData(Qt.ItemDataRole.UserRole, path)
                widget.thumbnail_grid.addItem(thumbnail_item)

            def _fake_load_image(path: str) -> None:
                widget._workspace._current_image_path = path

            widget.load_image = _fake_load_image  # type: ignore[method-assign]
            widget.image_list.setCurrentRow(1)

            self.assertEqual(widget.thumbnail_grid.currentRow(), 1)
            self.assertEqual(widget.thumbnail_grid.currentItem().data(Qt.ItemDataRole.UserRole), paths[1])
        finally:
            widget.close()
            widget.deleteLater()

    def test_image_list_selection_scrolls_thumbnail_grid_to_item(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            paths = [fr"d:\frames\frame_{index:03d}.png" for index in range(40)]
            widget._workspace.replace_image_selection(paths, is_supported_image=lambda _path: True)
            widget.neighbor_columns_spin.setValue(4)
            for path in paths:
                image_item = QListWidgetItem(Path(path).stem)
                image_item.setData(Qt.ItemDataRole.UserRole, path)
                widget.image_list.addItem(image_item)

                thumbnail_item = QListWidgetItem()
                thumbnail_item.setData(Qt.ItemDataRole.UserRole, path)
                widget.thumbnail_grid.addItem(thumbnail_item)

            widget._configure_thumbnail_grid_geometry()
            widget.thumbnail_grid_scroll_area.setFixedSize(4 * 64 + 4, 2 * 48 + 4)
            widget.show()
            QApplication.processEvents()

            def _fake_load_image(path: str) -> None:
                widget._workspace._current_image_path = path

            widget.load_image = _fake_load_image  # type: ignore[method-assign]
            widget.image_list.setCurrentRow(24)
            QApplication.processEvents()

            self.assertEqual(widget.thumbnail_grid.currentRow(), 24)
            self.assertGreater(widget.thumbnail_grid_scroll_area.verticalScrollBar().value(), 0)
        finally:
            widget.close()
            widget.deleteLater()

    def test_public_api_matches_golden_snapshot(self) -> None:
        widget = PolygonExtractionWidget()
        try:
            signals, methods = _collect_public_api(widget)
        finally:
            widget.close()
            widget.deleteLater()

        expected_signals, expected_methods = _parse_golden(GOLDEN_PATH.read_text(encoding="utf-8"))

        missing_signals = sorted(set(expected_signals) - set(signals))
        extra_signals = sorted(set(signals) - set(expected_signals))
        missing_methods = sorted(set(expected_methods) - set(methods))
        extra_methods = sorted(set(methods) - set(expected_methods))

        self.assertEqual(
            (missing_signals, extra_signals, missing_methods, extra_methods),
            ([], [], [], []),
            msg=(
                "Public API drifted from golden snapshot. "
                "If intentional, update tests/golden/widget_public_api.txt.\n"
                f"Missing signals: {missing_signals}\n"
                f"Extra signals:   {extra_signals}\n"
                f"Missing methods: {missing_methods}\n"
                f"Extra methods:   {extra_methods}"
            ),
        )


def regenerate_snapshot() -> None:
    """Utility entry point: rewrite the golden snapshot from the current widget."""
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        signals, methods = _collect_public_api(widget)
    finally:
        widget.close()
        widget.deleteLater()

    lines = [
        "# Golden snapshot of PolygonExtractionWidget public API.",
        "# Regenerate via tests/unit/test_widget_smoke.py::regenerate_snapshot() when a",
        "# change is intentional; otherwise any diff is a regression.",
        "#",
        "# Format: one SIGNAL/METHOD name per line. Inherited Qt members are excluded.",
        "",
        "[signals]",
        *signals,
        "",
        "[methods]",
        *methods,
        "",
    ]
    GOLDEN_PATH.write_text("\n".join(lines), encoding="utf-8")
    _ = app


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        regenerate_snapshot()
    else:
        unittest.main()
