"""Smoke and characterization tests for :class:`PolygonExtractionWidget`.

These tests protect the public API surface of the top-level widget while the
codebase is being refactored. Changes to the public API must be intentional
and require updating ``tests/golden/widget_public_api.txt``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from PyQt6.QtCore import pyqtBoundSignal
from PyQt6.QtWidgets import QApplication, QWidget

from polygon_widget.widget import PolygonExtractionWidget

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
