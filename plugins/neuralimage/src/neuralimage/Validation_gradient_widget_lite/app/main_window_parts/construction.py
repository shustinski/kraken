"""main_window: construction."""
from __future__ import annotations
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QWidget,
)
from ...infra.services import ValidationGradientLiteSettingsService
from ...ui.i18n import Translator, set_current_language
from ...ui.ui_constants import (
    EXTEND_ROOT_OBJECT_NAME,
    EXTEND_WIDGET_STYLESHEET,
    SETTINGS_APP,
    SETTINGS_ORG,
)
from neuralimage.Validation_gradient_widget_lite.app.presenter import ValidationGradientExtendPresenter


class ConstructionMixin:
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(EXTEND_ROOT_OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._theme = "dark"
        self.setStyleSheet(EXTEND_WIDGET_STYLESHEET)
        self._settings_service = ValidationGradientLiteSettingsService(QSettings(SETTINGS_ORG, SETTINGS_APP))
        language = self._settings_service.load_language()
        set_current_language(language)
        self._i18n = Translator(language)
        self._t = self._i18n.tr

        self._build_ui()
        self._setup_menu_bar()

        self._presenter = ValidationGradientExtendPresenter(self, self._settings_service)
        self._connect_signals()
        self._presenter._restore_persisted_state()
        self._presenter._refresh_folder_rows()
        self._presenter._sync_action_buttons()
