"""pipeline_mixin: render color binarize parameters."""
from __future__ import annotations
from contour.widget_parts._imports import (
    QAbstractItemView,
    QColor,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    Qt,
    QVBoxLayout,
    QWidget,
)


class RenderColorBinarizeParametersMixin:
    def _render_color_binarize_parameters(self, row: int) -> None:
        entries = self._color_selection_entries(row)
        group = QGroupBox(
            self._tr(
                "color_binarize_group_title",
                "Цвета для бинаризации" if self._ui_language == "ru" else "Colors for binarization",
            )
        )
        layout = QVBoxLayout(group)
        hint = QLabel(
            self._tr(
                "color_binarize_hint",
                "Включите выбор и кликните по изображению, чтобы добавить цвет. Галочкой можно временно отключить цвет."
                if self._ui_language == "ru"
                else "Enable picking and click the image to add a color. Uncheck an item to disable it temporarily.",
            )
        )
        hint.setWordWrap(True)
        hint.setToolTip(
            "Цвета из списка используются для построения бинарной маски; допуск задается параметром delta."
            if self._ui_language == "ru"
            else "Colors in the list are used to build the binary mask; tolerance is controlled by delta."
        )
        layout.addWidget(hint)
        color_list = QListWidget()
        color_list.setToolTip(
            "Отмеченные цвета участвуют в бинаризации. Снимите галочку, чтобы временно исключить цвет из маски."
            if self._ui_language == "ru"
            else "Checked colors participate in binarization. Uncheck a color to temporarily exclude it from the mask."
        )
        color_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for entry in entries:
            rgb = entry["rgb"]
            item = QListWidgetItem(f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}")
            item.setToolTip(
                "Этот цвет добавляет похожие пиксели в маску; галочка включает или выключает его."
                if self._ui_language == "ru"
                else "This color adds similar pixels to the mask; the checkbox enables or disables it."
            )
            item.setData(Qt.ItemDataRole.UserRole, list(rgb))
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(Qt.CheckState.Checked if entry.get("enabled", True) else Qt.CheckState.Unchecked)
            item.setBackground(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            brightness = int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114
            item.setForeground(QColor("#111111" if brightness > 150 else "#F8FAFC"))
            color_list.addItem(item)
        color_list.itemChanged.connect(
            lambda item, row_index=row, widget=color_list: self._on_color_entry_changed(row_index, widget, item)
        )
        layout.addWidget(color_list)

        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        pick_button = QPushButton(
            self._tr("pick_colors_button", "Выбор с изображения" if self._ui_language == "ru" else "Pick from image")
        )
        pick_button.setCheckable(True)
        pick_button.setToolTip(
            "Включает выбор цвета с изображения: кликните по нужному пикселю, чтобы добавить его в список."
            if self._ui_language == "ru"
            else "Enables picking from the image: click a pixel to add its color to the list."
        )
        pick_button.setChecked(self._color_pick_pipeline_row == row)
        pick_button.toggled.connect(
            lambda checked, row_index=row: self._set_color_pick_active(row_index if checked else None)
        )
        remove_button = QPushButton(
            self._tr(
                "remove_selected_color_button", "Удалить выбранный" if self._ui_language == "ru" else "Remove selected"
            )
        )
        remove_button.setToolTip(
            "Удаляет выбранный цвет из списка бинаризации."
            if self._ui_language == "ru"
            else "Removes the selected color from the binarization list."
        )
        remove_button.clicked.connect(
            lambda _checked=False, row_index=row, widget=color_list: self._remove_selected_color_entry(
                row_index, widget
            )
        )
        clear_button = QPushButton(
            self._tr("clear_colors_button", "Очистить список" if self._ui_language == "ru" else "Clear list")
        )
        clear_button.setToolTip(
            "Очищает весь список цветов для этого шага бинаризации."
            if self._ui_language == "ru"
            else "Clears the whole color list for this binarization step."
        )
        clear_button.clicked.connect(lambda _checked=False, row_index=row: self._clear_color_entries(row_index))
        buttons_layout.addWidget(pick_button)
        buttons_layout.addWidget(remove_button)
        buttons_layout.addWidget(clear_button)
        layout.addWidget(buttons_row)
        self.parameters_form.addRow(group)
