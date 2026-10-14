"""main_window: build metric settings widget."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QComboBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from ...ui.ui_constants import (
    METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH,
    METRIC_SETTINGS_LABEL_MIN_WIDTH,
)


class BuildMetricSettingsWidgetMixin:
    def _build_metric_settings_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        for combo in (self.metric_scope_combo, self.metric_combo):
            combo.setMinimumContentsLength(max(8, min(10, METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH)))
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._metric_scope_row = self._build_setting_row(
            self._t("analysis.confidence_model"),
            self.metric_scope_combo,
            label_min_width=METRIC_SETTINGS_LABEL_MIN_WIDTH,
            compact=True,
        )
        layout.addWidget(self._metric_scope_row)
        self._metric_select_row = self._build_setting_row(
            self._t("menu.metric.select"),
            self.metric_combo,
            label_min_width=METRIC_SETTINGS_LABEL_MIN_WIDTH,
            compact=True,
        )
        layout.addWidget(self._metric_select_row)
        return widget

    def _update_language_toggle_button(self) -> None:
        current_language = str(self._i18n.language or "en").lower()
        next_language = "RU" if current_language == "en" else "EN"
        self.language_toggle_button.setText(next_language)
        self.language_toggle_button.setToolTip(self._t("language.toggle_tooltip"))
