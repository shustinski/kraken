"""main_window: create menubar."""
from __future__ import annotations
from PyQt6.QtGui import QAction, QActionGroup
from neuralimage.view.changelog_dialog import show_changelog_dialog
from neuralimage.view.help_dialog import show_help_dialog

from neuralimage.view.main_window_parts.responsive_preview_label import _load_menu_icon


class CreateMenubarMixin:
    def _create_menubar(self, t: dict[str, str]):
        menubar = self.menuBar()
        if menubar is None:
            return
        file_menu = menubar.addMenu(t.get("menu_file", "Файл"))
        settings_menu = menubar.addMenu(t["menu_settings"])
        self._view_menu = menubar.addMenu(t["menu_view"])
        view_menu = self._view_menu
        self._plugins_menu = menubar.addMenu(t.get("menu_plugins", "Plugins"))
        plugins_menu = self._plugins_menu
        info_menu = menubar.addMenu(t["menu_help"])
        self._file_menu = file_menu
        self._settings_menu = settings_menu
        self._info_menu = info_menu
        if file_menu is None or settings_menu is None or view_menu is None or plugins_menu is None or info_menu is None:
            return

        self._open_config_action = QAction(t.get("menu_open_config", "Открыть"), self)
        file_menu.addAction(self._open_config_action)
        menu_icon = _load_menu_icon()
        self._settings_sample_action = QAction(menu_icon, t["menu_sample"], self)
        self._settings_train_action = QAction(menu_icon, t["menu_train"], self)
        self._settings_pred_action = QAction(menu_icon, t["menu_pred"], self)
        settings_menu.addAction(self._settings_sample_action)
        settings_menu.addAction(self._settings_train_action)
        settings_menu.addAction(self._settings_pred_action)
        metrics_action = self.metrics_panel.toggleViewAction()
        if metrics_action is not None:
            metrics_action.setText(t["menu_metrics"])
            view_menu.addAction(metrics_action)
            self._metrics_toggle_action = metrics_action
        log_action = self.log_dock.toggleViewAction()
        if log_action is not None:
            log_action.setText(t.get("menu_log_panel", "Панель лога"))
            view_menu.addAction(log_action)
            self._log_toggle_action = log_action
        queue_action = self.queue_dock.toggleViewAction()
        if queue_action is not None:
            queue_action.setText(t.get('menu_queue_panel', t['queue']))
            view_menu.addAction(queue_action)
            self._queue_toggle_action = queue_action
        if self.settings_dock is not None:
            settings_action = self.settings_dock.toggleViewAction()
            if settings_action is not None:
                settings_action.setText(t["menu_settings_panel"])
                view_menu.addAction(settings_action)
                self._settings_toggle_action = settings_action
        self.batch_preview_action = QAction(t["menu_batch_preview"], self)
        self.batch_preview_action.setCheckable(True)
        self.batch_preview_action.setChecked(True)
        view_menu.addAction(self.batch_preview_action)
        self.release_memory_action = QAction(t["menu_release_memory"], self)
        view_menu.addAction(self.release_memory_action)
        self.open_tic_tac_toe_action = QAction(
            t.get("menu_open_tic_tac_toe", "Крестики-нолики (нейросеть)"),
            self,
        )
        view_menu.addAction(self.open_tic_tac_toe_action)
        self._open_validation_gradient_action = QAction(
            t.get("menu_open_validation_gradient", "Open Validation gradient"),
            self,
        )
        plugins_menu.addAction(self._open_validation_gradient_action)
        view_menu.addSeparator()
        self._language_menu = view_menu.addMenu(t.get("menu_language", "Язык"))
        language_group = QActionGroup(self)
        language_group.setExclusive(True)
        self.ui_language_ru_action = QAction(t.get("lang_ru", "Русский"), self)
        self.ui_language_ru_action.setCheckable(True)
        self.ui_language_ru_action.setData("ru")
        language_group.addAction(self.ui_language_ru_action)
        self._language_menu.addAction(self.ui_language_ru_action)
        self.ui_language_en_action = QAction(t.get("lang_en", "English"), self)
        self.ui_language_en_action.setCheckable(True)
        self.ui_language_en_action.setData("en")
        language_group.addAction(self.ui_language_en_action)
        self._language_menu.addAction(self.ui_language_en_action)
        self._theme_menu = menubar.addMenu(t.get("menu_theme", "Тема"))
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self.theme_dark_action = QAction(t.get("theme_dark", "Темная"), self)
        self.theme_dark_action.setCheckable(True)
        self.theme_dark_action.setData("dark")
        theme_group.addAction(self.theme_dark_action)
        self._theme_menu.addAction(self.theme_dark_action)
        self.theme_light_action = QAction(t.get("theme_light", "Светлая"), self)
        self.theme_light_action.setCheckable(True)
        self.theme_light_action.setData("light")
        theme_group.addAction(self.theme_light_action)
        self._theme_menu.addAction(self.theme_light_action)
        self._ui_mode_menu = view_menu.addMenu(t.get("menu_ui_mode", "Interface mode"))
        ui_mode_group = QActionGroup(self)
        ui_mode_group.setExclusive(True)
        self._ui_mode_simple_action = QAction(t.get("menu_ui_mode_simple", "Simple"), self)
        self._ui_mode_simple_action.setCheckable(True)
        self._ui_mode_simple_action.setData("simple")
        ui_mode_group.addAction(self._ui_mode_simple_action)
        self._ui_mode_menu.addAction(self._ui_mode_simple_action)
        self._ui_mode_advanced_action = QAction(t.get("menu_ui_mode_advanced", "Advanced"), self)
        self._ui_mode_advanced_action.setCheckable(True)
        self._ui_mode_advanced_action.setData("advanced")
        ui_mode_group.addAction(self._ui_mode_advanced_action)
        self._ui_mode_menu.addAction(self._ui_mode_advanced_action)
        self._sync_language_menu_checks()
        self._sync_theme_menu_checks()
        self._sync_ui_mode_menu_checks()
        help_action = QAction(t["menu_open_help"], self)
        help_action.triggered.connect(lambda: show_help_dialog(self))
        info_menu.addAction(help_action)
        self._help_action = help_action
        changelog_action = QAction(t.get("menu_open_changelog", "Список изменений"), self)
        changelog_action.triggered.connect(lambda: show_changelog_dialog(self))
        info_menu.addAction(changelog_action)
        self._changelog_action = changelog_action
        self._update_menu = info_menu.addMenu(t.get("menu_update", "Update"))
        check_updates_action = QAction(t.get("menu_check_updates", "Проверить обновления"), self)
        self._update_menu.addAction(check_updates_action)
        self._check_updates_action = check_updates_action
        self._update_channel_menu = self._update_menu.addMenu(t.get("menu_update_channel", "Канал обновлений"))
        self.configure_update_channels(self._available_update_channels or ['stable'], self._selected_update_channel)
        menu_action = info_menu.menuAction()
        if menu_action is not None:
            menu_action.setVisible(True)
