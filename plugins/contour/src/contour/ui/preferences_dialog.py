"""Tools → Preferences: the display settings that used to live in the Display dock.

The dialog does not build its own controls. It moves the widgets that
``build_display_tab`` created into its pages, so their signals, the restore of
persisted display settings and the settings save keep working unchanged.
"""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..application.processing import VIA_DISPLAY_MODE_CIRCLE, VIA_DISPLAY_MODE_RECTANGLE

# (key, fields) per section. A field is (label attribute or None, field attribute).
_SECTIONS: tuple[tuple[str, tuple[tuple[str | None, str], ...]], ...] = (
    (
        "contours",
        (
            ("external_color_label_widget", "external_color_button"),
            ("hole_color_label_widget", "hole_color_button"),
            ("selected_color_label_widget", "selected_color_button"),
            ("conductor_hover_highlight_label_widget", "conductor_hover_highlight_color_button"),
            ("line_width_label_widget", "line_width_spin"),
            ("fill_opacity_label_widget", "fill_opacity_spin"),
            (None, "random_object_colors_checkbox"),
        ),
    ),
    (
        "vertices",
        (
            ("vertex_color_label_widget", "vertex_color_button"),
            ("vertex_size_label_widget", "vertex_size_spin"),
            (None, "show_vertices_checkbox"),
            (None, "show_labels_checkbox"),
        ),
    ),
    (
        "contacts",
        (
            ("via_selection_color_label_widget", "via_selection_color_button"),
            ("via_display_mode_label_widget", "via_display_mode_combo"),
        ),
    ),
    (
        "frame_matrix",
        (
            (None, "show_frame_matrix_checkbox"),
            (None, "show_frame_matrix_thumbnails_checkbox"),
        ),
    ),
    (
        "neighbors",
        (
            (None, "show_neighbor_frames_checkbox"),
            (None, "show_neighbor_vectors_checkbox"),
            ("neighbor_columns_label_widget", "neighbor_columns_spin"),
            ("neighbor_max_grid_label_widget", "neighbor_max_grid_spin"),
            ("neighbor_opacity_label_widget", "neighbor_opacity_spin"),
            ("neighbor_overlap_label_widget", "neighbor_overlap_spin"),
        ),
    ),
)

SECTION_KEYS: tuple[str, ...] = tuple(key for key, _fields in _SECTIONS)
FIELD_MAX_WIDTH = 240


class ContourPreferencesDialog(QDialog):
    """Non-modal settings window; every change applies at once."""

    def __init__(self, widget: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._widget = widget
        self.setObjectName("contourPreferencesDialog")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setSizeGripEnabled(True)

        self.section_list = QListWidget(self)
        self.section_list.setObjectName("preferencesSectionList")
        self.section_list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.section_list.setFixedWidth(190)
        self.section_list.setSpacing(1)
        self.pages = QStackedWidget(self)
        self.pages.setObjectName("preferencesPages")
        self._page_titles: dict[str, QLabel] = {}

        self._detach_display_tab()
        for key, fields in _SECTIONS:
            self.section_list.addItem(QListWidgetItem(""))
            self.pages.addWidget(self._build_page(key, fields))
        self.section_list.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.section_list.setCurrentRow(0)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.button_box.rejected.connect(self.close)
        close_button = self.button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setObjectName("preferencesCloseButton")

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self.section_list)
        body.addWidget(self.pages, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        root.addLayout(body, 1)
        root.addWidget(self.button_box)
        self.resize(620, 400)
        self.retranslate()

    def _detach_display_tab(self) -> None:
        # The display tab no longer shows anywhere. It keeps the controls that
        # are not preferences (frame autosave) until they get a home of their own.
        tab = getattr(self._widget, "display_tab", None)
        if tab is None:
            return
        take_control_tab = getattr(self._widget, "_take_control_tab", None)
        if callable(take_control_tab):
            take_control_tab(tab)
        tab.setParent(self._widget)
        tab.hide()

    def _build_page(self, key: str, fields: tuple[tuple[str | None, str], ...]) -> QWidget:
        page = QWidget()
        page.setObjectName(f"preferencesPage_{key}")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(10)
        title = QLabel(page)
        title_font = QFont(title.font())
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() * 1.15)
        title.setFont(title_font)
        self._page_titles[key] = title
        layout.addWidget(title)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        old_form = getattr(self._widget, "display_form", None)
        for label_name, field_name in fields:
            field = getattr(self._widget, field_name, None)
            if field is None:
                continue
            label = getattr(self._widget, label_name, None) if label_name else None
            if old_form is not None:
                if label is not None:
                    old_form.removeWidget(label)
                old_form.removeWidget(field)
            if label is not None:
                # Center the label on its field; QFormLayout pins it to the top.
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                label.setMinimumHeight(field.sizeHint().height())
                field.setMaximumWidth(FIELD_MAX_WIDTH)
                form.addRow(label, field)
            else:
                form.addRow(field)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def show_section(self, key: str) -> None:
        if key in SECTION_KEYS:
            self.section_list.setCurrentRow(SECTION_KEYS.index(key))

    def open_preferences(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def retranslate(self) -> None:
        tr = self._widget._tr
        self.setWindowTitle(tr("preferences_title"))
        section_titles = {
            "contours": tr("preferences_section_contours"),
            "vertices": tr("preferences_section_vertices"),
            "contacts": tr("preferences_section_contacts"),
            "frame_matrix": tr("preferences_section_frame_matrix"),
            "neighbors": tr("preferences_section_neighbors"),
        }
        for row, key in enumerate(SECTION_KEYS):
            item = self.section_list.item(row)
            if item is not None:
                item.setText(section_titles[key])
            self._page_titles[key].setText(section_titles[key])
        close_button = self.button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setText(tr("preferences_close"))
        self._retranslate_via_display_mode(tr)

    def _retranslate_via_display_mode(self, tr: Any) -> None:
        label = getattr(self._widget, "via_display_mode_label_widget", None)
        if label is not None:
            label.setText(tr("preferences_via_display_mode"))
        combo = getattr(self._widget, "via_display_mode_combo", None)
        if combo is None:
            return
        item_texts = {
            VIA_DISPLAY_MODE_CIRCLE: tr("preferences_via_display_circle"),
            VIA_DISPLAY_MODE_RECTANGLE: tr("preferences_via_display_rectangle"),
        }
        for index in range(combo.count()):
            text = item_texts.get(combo.itemData(index))
            if text:
                combo.setItemText(index, text)
