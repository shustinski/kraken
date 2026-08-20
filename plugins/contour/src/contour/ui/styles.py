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
    margin-top: 2px;
    padding-top: 2px;
}
#polygonExtractionWidget QPushButton {
    min-height: 24px;
    padding: 2px 6px;
    font-size: 12px;
}
#polygonExtractionWidget QToolButton {
    padding: 0;
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
    min-height: 22px;
    padding: 1px 3px;
    font-size: 12px;
}
#polygonExtractionWidget QTabBar::tab {
    min-height: 20px;
    padding: 2px 6px;
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

RECOGNITION_SCENE_FRAME_STYLE = """
#editorSceneFrame {
    border: 3px solid #DC2626;
}
"""

__all__ = ["COMPACT_UI_STYLE", "RECOGNITION_SCENE_FRAME_STYLE"]
