"""main_window: build setting row."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from ...ui.ui_constants import (
    SETTINGS_LABEL_MIN_WIDTH,
)


class BuildSettingRowMixin:
    def _toggle_language(self) -> None:
        current_language = str(self._i18n.language or "en").lower()
        language = "ru" if current_language == "en" else "en"
        self._i18n.set_language(language)
        self._settings_service.save_language(language)
        self.retranslate_ui()

    def _build_setting_row(
        self,
        title: str,
        control: QWidget,
        *,
        label_min_width: int = SETTINGS_LABEL_MIN_WIDTH,
        compact: bool = False,
    ) -> QWidget:
        row = QWidget(self)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2 if compact else 4)
        label = QLabel(title, row)
        label.setMinimumWidth(0)
        label.setMaximumWidth(int(label_min_width * 3))
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        control.setMinimumWidth(0)
        if not compact:
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(label)
        layout.addWidget(control)
        row._title_label = label  # type: ignore[attr-defined]
        return row
