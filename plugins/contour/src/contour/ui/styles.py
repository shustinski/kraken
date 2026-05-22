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
#polygonExtractionWidget QGroupBox {
    margin-top: 4px;
    padding-top: 4px;
}
#polygonExtractionWidget QPushButton {
    min-height: 26px;
    padding: 3px 8px;
    font-size: 12px;
}
#polygonExtractionWidget QToolButton {
    padding: 2px;
}
#polygonExtractionWidget QToolButton:checked {
    background-color: #16A34A;
    border: 2px solid #86EFAC;
    border-radius: 4px;
}
#polygonExtractionWidget QToolButton:checked:hover {
    background-color: #15803D;
}
#polygonExtractionWidget QLineEdit,
#polygonExtractionWidget QComboBox,
#polygonExtractionWidget QSpinBox,
#polygonExtractionWidget QDoubleSpinBox {
    min-height: 24px;
    padding: 1px 4px;
    font-size: 12px;
}
#polygonExtractionWidget QTabBar::tab {
    min-height: 22px;
    padding: 3px 8px;
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
