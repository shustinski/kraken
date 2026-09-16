from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


def add_compact_row(
    layout: QVBoxLayout,
    *,
    checkbox=None,
    label_text: str = '',
    label_tooltip: str = '',
    controls: tuple[object, ...] = (),
    stretch_label: bool = True,
) -> QWidget:
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)

    if checkbox is not None:
        row_layout.addWidget(checkbox, 0, Qt.AlignmentFlag.AlignTop)

    if label_text:
        label = QLabel(label_text, row)
        label.setWordWrap(True)
        if label_tooltip:
            label.setToolTip(label_tooltip)
        row_layout.addWidget(label, 1 if stretch_label else 0)

    for control in controls:
        row_layout.addWidget(control, 0)

    if stretch_label and not label_text:
        row_layout.addStretch(1)

    layout.addWidget(row)
    return row
