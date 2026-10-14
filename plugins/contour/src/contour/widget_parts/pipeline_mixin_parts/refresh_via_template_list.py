"""pipeline_mixin: refresh via template list."""
from __future__ import annotations
from contour.widget_parts._imports import (
    QDoubleSpinBox,
    QLabel,
    QPixmap,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    Qt,
    cv_to_qimage,
)


class RefreshViaTemplateListMixin:
    def _refresh_via_template_list(self) -> None:
        if not hasattr(self, "via_template_table"):
            return
        self._ensure_via_template_metadata()
        table = self.via_template_table
        table.clearSpans()
        table.setRowCount(0)
        if not self._via_template_images:
            table.insertRow(0)
            table.setRowHeight(0, 52)
            table.setSpan(0, 0, 1, 5)
            empty_label = QLabel(
                "Чтобы добавить шаблон удерживайте Ctrl и выделите область"
                if self._ui_language == "ru"
                else "Hold Ctrl and select an area to add a template"
            )
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setWordWrap(True)
            table.setCellWidget(0, 0, empty_label)
        for index, template in enumerate(self._via_template_images, start=1):
            row = index - 1
            table.insertRow(row)
            table.setRowHeight(row, 64)

            order_spin = QSpinBox()
            order_spin.setRange(1, len(self._via_template_images))
            order_spin.setValue(index)
            order_spin.setMinimumWidth(0)
            order_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            order_spin.valueChanged.connect(
                lambda value, source_row=row: self._move_via_template(source_row, int(value) - 1)
            )
            table.setCellWidget(row, 0, order_spin)

            preview_pixmap = QPixmap.fromImage(cv_to_qimage(template)).scaled(
                56,
                56,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            preview_label = QLabel()
            preview_label.setPixmap(preview_pixmap)
            preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview_label.setFixedSize(60, 60)
            table.setCellWidget(row, 1, preview_label)

            similarity_spin = QDoubleSpinBox()
            similarity_spin.setRange(0.0, 1.0)
            similarity_spin.setDecimals(3)
            similarity_spin.setSingleStep(0.01)
            similarity_spin.setValue(self._via_template_min_scores[row])
            similarity_spin.setMinimumWidth(0)
            similarity_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            similarity_spin.valueChanged.connect(
                lambda value, template_row=row: self._set_via_template_similarity(template_row, float(value))
            )
            table.setCellWidget(row, 2, similarity_spin)

            diameter_spin = QSpinBox()
            diameter_spin.setRange(1, 10_000)
            diameter_spin.setSuffix(" px")
            diameter_spin.setValue(self._via_template_diameters[row])
            diameter_spin.setMinimumWidth(0)
            diameter_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            diameter_spin.valueChanged.connect(
                lambda value, template_row=row: self._set_via_template_diameter(template_row, int(value))
            )
            table.setCellWidget(row, 3, diameter_spin)

            remove_button = QPushButton("-")
            remove_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            remove_button.setMinimumSize(0, 0)
            remove_button.setStyleSheet(
                "QPushButton { background-color: #DC2626; color: white; font-weight: 700; "
                "border-radius: 0; padding: 0; }"
            )
            remove_button.setToolTip("Удалить шаблон" if self._ui_language == "ru" else "Remove template")
            remove_button.clicked.connect(
                lambda _checked=False, template_row=row: self._remove_via_template(template_row)
            )
            table.setCellWidget(row, 4, remove_button)
        table.setColumnWidth(1, 64)
        visible_rows = max(1, min(5, len(self._via_template_images)))
        row_height = 64 if self._via_template_images else 52
        header_height = max(24, table.horizontalHeader().sizeHint().height())
        table_height = header_height + visible_rows * row_height + table.frameWidth() * 2 + 2
        table.setMinimumHeight(table_height)
        table.setMaximumHeight(table_height)
