"""window: construction 2."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,

)
from krona.presentation.qt.ui_strings import available_languages
from krona.version import __version__
from kraken_core.theme import add_theme_menu
from updater.qt import QtUpdateController


class Construction2Mixin:
    def _init_ui(self) -> None:
        add_theme_menu(
            self,
            initial_theme=self._current_theme.lower(),
            on_theme_changed=lambda theme: self._on_theme_changed("Light" if theme == "light" else "Dark"),
        )
        help_menu = self.menuBar().addMenu(self._t("menu.help", "Help"))
        self._update_controller = QtUpdateController(
            self,
            app_id="krona",
            app_name="Krona",
            current_version=__version__,
            status_callback=lambda message: self.statusBar().showMessage(message, 5000),
        )
        self._update_controller.add_menu_action(
            help_menu,
            self._t("menu.check_updates", "Check updates"),
            submenu_title=self._t("menu.update", "Update"),
        )
        root = QWidget(self)
        root_layout = QVBoxLayout(root)

        toolbar = QHBoxLayout()
        self.path_label = QLabel(self._t("path.no_file", "No EDF file loaded"))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_btn = QPushButton(self._t("button.open_edf", "Open EDF"))
        self.open_btn.clicked.connect(self.on_open_file)
        self.zoom_in_btn = QPushButton(self._t("button.zoom_in", "Zoom In"))
        self.zoom_in_btn.clicked.connect(lambda: self._apply_zoom(1.2))
        self.zoom_out_btn = QPushButton(self._t("button.zoom_out", "Zoom Out"))
        self.zoom_out_btn.clicked.connect(lambda: self._apply_zoom(1 / 1.2))
        self.zoom_reset_btn = QPushButton(self._t("button.reset_zoom", "Reset Zoom"))
        self.zoom_reset_btn.clicked.connect(self._fit_scene_in_view)
        self.hitbox_toggle_btn = QPushButton(self._t("button.show_hitboxes", "Show Hitboxes"))
        self.hitbox_toggle_btn.setCheckable(True)
        self.hitbox_toggle_btn.setChecked(False)
        self.hitbox_toggle_btn.toggled.connect(self._on_hitbox_toggle_changed)
        self.language_label = QLabel(self._t("label.language", "Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(available_languages())
        if self._current_language in available_languages():
            self.language_combo.setCurrentText(self._current_language)
        self.language_combo.currentTextChanged.connect(self._on_language_changed)
        self.theme_label = QLabel(self._t("label.theme", "Theme:"))
        self.theme_combo = QComboBox()
        theme_items = self._theme_manager.available_themes() if self._theme_manager else ["Dark"]
        self.theme_combo.addItems(theme_items)
        if self._current_theme in theme_items:
            self.theme_combo.setCurrentText(self._current_theme)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        toolbar.addWidget(self.open_btn)
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.zoom_reset_btn)
        toolbar.addWidget(self.hitbox_toggle_btn)
        toolbar.addWidget(self.language_label)
        toolbar.addWidget(self.language_combo)
        toolbar.addWidget(self.theme_label)
        toolbar.addWidget(self.theme_combo)
        toolbar.addWidget(self.path_label, 1)
        root_layout.addLayout(toolbar)

        self.summary_group = QGroupBox(self._t("group.summary", "Top-Level Summary"))
        summary_layout = QFormLayout(self.summary_group)
        self.design_value = QLabel("-")
        self.library_value = QLabel("-")
        self.cell_value = QLabel("-")
        self.view_value = QLabel("-")
        self.instance_count_value = QLabel("0")
        self.net_count_value = QLabel("0")
        summary_layout.addRow(self._t("summary.design", "Design:"), self.design_value)
        summary_layout.addRow(self._t("summary.library", "Library:"), self.library_value)
        summary_layout.addRow(self._t("summary.cell", "Cell:"), self.cell_value)
        summary_layout.addRow(self._t("summary.view", "View:"), self.view_value)
        summary_layout.addRow(self._t("summary.instances", "Instances:"), self.instance_count_value)
        summary_layout.addRow(self._t("summary.nets", "Nets:"), self.net_count_value)
        root_layout.addWidget(self.summary_group)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_nets_tab(), self._t("tab.nets", "Nets"))
        self.tabs.addTab(self._build_instances_tab(), self._t("tab.instances", "Instances"))
        self.tabs.addTab(self._build_logic_tab(), self._t("tab.logic", "Logic"))
        self.tabs.addTab(self._build_diagnostics_tab(), self._t("tab.diagnostics", "Diagnostics"))
        self.tabs.addTab(self._build_scene_tab(), self._t("tab.scene", "Scene"))
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)
