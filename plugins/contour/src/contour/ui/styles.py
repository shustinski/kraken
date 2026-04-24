"""Reusable Qt stylesheets for the polygon extraction widget."""

from __future__ import annotations

COMPACT_UI_STYLE = """
#polygonExtractionWidget {
    font-size: 12px;
}
#polygonExtractionWidget QLabel,
#polygonExtractionWidget QCheckBox,
#polygonExtractionWidget QGroupBox {
    font-size: 12px;
}
#polygonExtractionWidget QPushButton {
    min-height: 28px;
    padding: 4px 10px;
    font-size: 12px;
}
#polygonExtractionWidget QToolButton {
    padding: 2px;
}
#polygonExtractionWidget QLineEdit,
#polygonExtractionWidget QComboBox,
#polygonExtractionWidget QSpinBox,
#polygonExtractionWidget QDoubleSpinBox {
    min-height: 26px;
    padding: 2px 6px;
    font-size: 12px;
}
#polygonExtractionWidget QTabBar::tab {
    min-height: 24px;
    padding: 4px 10px;
    font-size: 12px;
}
#polygonExtractionWidget QListWidget {
    font-size: 12px;
}
#polygonExtractionWidget QProgressBar {
    min-height: 18px;
    max-height: 18px;
}
"""

__all__ = ["COMPACT_UI_STYLE"]
