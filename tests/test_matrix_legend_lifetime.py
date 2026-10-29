"""Thumbnail completions must not stack deferred legend widgets over new ones."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtCore import Qt

from kraken_manager.presentation.qt.matrix_legend import MatrixLegendPanel


def test_changed_legend_hides_old_widgets_before_deferred_deletion():
    app = QApplication.instance() or QApplication([])
    panel = MatrixLegendPanel()
    panel.resize(240, 700)
    panel.show()
    app.processEvents()
    old = panel.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
    panel.set_rows(({"status": "vectorized", "performer_name": "Deployment", "modified_at": "2026-10-27"},))
    assert all(not widget.isVisible() for widget in old)
    titles = [w for w in panel.findChildren(QLabel, "matrixLegendTitle") if w.isVisible()]
    assert len(titles) == 4
    top = [w.mapTo(panel, w.rect().topLeft()).y() for w in titles]
    assert all(right >= left + titles[index].height() for index, (left, right) in enumerate(zip(top, top[1:])))
    panel.close()


def test_thumbnail_only_refresh_does_not_accumulate_legend_widgets():
    app = QApplication.instance() or QApplication([])
    panel = MatrixLegendPanel()
    panel.show()
    app.processEvents()
    rows = ({"status": "vectorized"},)
    panel.set_rows(rows)
    initial = panel.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
    for _ in range(100):
        panel.set_rows(rows)
    assert panel.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly) == initial
    panel.close()
