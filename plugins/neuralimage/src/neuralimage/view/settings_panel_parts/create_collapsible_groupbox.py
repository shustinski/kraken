"""settings_panel: create collapsible groupbox."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QLabel,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

from neuralimage.view.settings_panel_parts.module_state import (
    CONTENT_LAYOUT_MARGINS,
    CONTENT_LAYOUT_SPACING,
    FIELD_DESCRIPTION_ROW_SPACING,
)


class CreateCollapsibleGroupboxMixin:
    def _get_desc_label(self, key: str) -> QLabel:
        label = self._desc_labels.get(key)
        if label is not None:
            return label
        label = QLabel('')
        label.setWordWrap(True)
        label.setMaximumWidth(190)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet('')
        self._desc_labels[key] = label
        return label

    def _field_with_description(self, field: QWidget, key: str) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(FIELD_DESCRIPTION_ROW_SPACING)
        row_layout.addWidget(self._get_desc_label(key), 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        row_layout.addWidget(field, 1, Qt.AlignmentFlag.AlignVCenter)
        self._field_rows[field] = row_widget
        self._desc_fields[key] = field
        return row_widget

    def _add_labeled_row(self, form: QFormLayout, field: QWidget, key: str) -> None:
        """Add a single visible label inside the row to avoid duplicate captions."""
        form.addRow(self._field_with_description(field, key))

    def _create_collapsible_groupbox(self) -> tuple[QGroupBox, QVBoxLayout, QWidget]:
        groupbox = QGroupBox('')
        groupbox.setCheckable(True)
        groupbox.setChecked(False)
        groupbox_layout = QVBoxLayout(groupbox)
        groupbox_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        groupbox_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        content_widget.setVisible(False)
        groupbox_layout.addWidget(content_widget)
        groupbox.toggled.connect(content_widget.setVisible)
        return groupbox, content_layout, content_widget

    @staticmethod
    def _apply_tooltip_to_widget_and_children(widget: QWidget, text: str) -> None:
        widget.setToolTip(text)
