"""Build the mismatch-only lite widget shell and its standalone host window."""
from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QProgressBar,
    QSplitter,
    QSpinBox,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..core.domain import BuildResult, ComparisonMode
from ..ui.matrix_view import GradientPresetSelectorWidget, GradientRangeSelectorWidget, MatrixListWidget, MatrixMiniMapWidget

from ..ui.i18n import Translator, set_current_language
from .presenter import ValidationGradientLitePresenter
from ..infra.services import ValidationGradientLiteSettingsService
from .state import LiteMatrixTabState, LitePreviewPanel
from ..ui.ui_constants import (
    CONTROL_PANEL_SPLITTER_SIZES,
    DEFAULT_CELL_SIZE,
    DEFAULT_COMPARISON_MODE,
    DEFAULT_ERROR_WINDOW,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_GRADIENT_NAME,
    DEFAULT_SCORE_VIEW_MODE,
    DEFAULT_TOTAL_FRAMES,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    FRAMES_PER_ROW_RANGE,
    LITE_LANGUAGE_BUTTON_OBJECT_NAME,
    LITE_ROOT_OBJECT_NAME,
    LITE_WIDGET_STYLESHEET,
    MATRIX_COLUMNS_RANGE,
    MATRIX_ROWS_RANGE,
    OVERVIEW_PANEL_MAX_WIDTH,
    SETTINGS_APP,
    SETTINGS_LABEL_MIN_WIDTH,
    SETTINGS_ORG,
    THUMBNAIL_SIZE_RANGE,
    TOTAL_FRAMES_RANGE,
)


class ValidationGradientLiteWidget(QWidget):
    """Build the embeddable mismatch-only lite widget UI."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(LITE_ROOT_OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(LITE_WIDGET_STYLESHEET)
        self._settings_service = ValidationGradientLiteSettingsService(QSettings(SETTINGS_ORG, SETTINGS_APP))
        language = self._settings_service.load_language()
        set_current_language(language)
        self._i18n = Translator(language)
        self._t = self._i18n.tr

        self._build_ui()
        self._setup_menu_bar()

        self._presenter = ValidationGradientLitePresenter(self, self._settings_service)
        self._connect_signals()
        self._presenter._restore_persisted_state()
        self._presenter._sync_layout_control_state()
        self._presenter._sync_action_buttons()
        self._presenter._refresh_folder_rows()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._menu_bar = QMenuBar(self)
        root_layout.addWidget(self._menu_bar)

        content = QWidget(self)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.addWidget(content, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal, content)
        content_layout.addWidget(splitter)

        control_panel = QWidget(splitter)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)

        folders_group = QGroupBox(self._t("folders.group"), control_panel)
        self.folders_group = folders_group
        folders_layout = QVBoxLayout(folders_group)
        folders_info = QLabel(self._t("folders.info"), folders_group)
        self.folders_info_label = folders_info
        folders_info.setWordWrap(True)
        folders_layout.addWidget(folders_info)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(4)

        style = self.style()
        self.btn_add_folder = QToolButton(folders_group)
        self.btn_add_folder.setAutoRaise(True)
        self.btn_add_folder.setProperty('liteToolbarButton', True)
        self.btn_add_folder.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self.btn_add_folder.setToolTip(self._t("folders.add"))
        toolbar_layout.addWidget(self.btn_add_folder)

        self.btn_clear_folders = QToolButton(folders_group)
        self.btn_clear_folders.setAutoRaise(True)
        self.btn_clear_folders.setProperty('liteToolbarButton', True)
        self.btn_clear_folders.setText("x")
        self.btn_clear_folders.setToolTip(self._t("folders.clear_all"))
        toolbar_layout.addWidget(self.btn_clear_folders)

        self.btn_set_base = QToolButton(folders_group)
        self.btn_set_base.setAutoRaise(True)
        self.btn_set_base.setProperty('liteToolbarButton', True)
        self.btn_set_base.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.btn_set_base.setToolTip(self._t("folders.set_base"))
        toolbar_layout.addWidget(self.btn_set_base)

        self.btn_clear_base = QToolButton(folders_group)
        self.btn_clear_base.setAutoRaise(True)
        self.btn_clear_base.setProperty('liteToolbarButton', True)
        self.btn_clear_base.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.btn_clear_base.setToolTip(self._t("folders.clear_base"))
        toolbar_layout.addWidget(self.btn_clear_base)

        self.btn_build = QToolButton(folders_group)
        self.btn_build.setAutoRaise(True)
        self.btn_build.setProperty('liteToolbarButton', True)
        self.btn_build.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_build.setToolTip(self._t("folders.build"))
        toolbar_layout.addWidget(self.btn_build)

        self.btn_compute = QToolButton(folders_group)
        self.btn_compute.setAutoRaise(True)
        self.btn_compute.setProperty('liteToolbarButton', True)
        self.btn_compute.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.btn_compute.setToolTip(self._t("folders.compute_mismatch"))
        toolbar_layout.addWidget(self.btn_compute)

        self.btn_cancel = QToolButton(folders_group)
        self.btn_cancel.setAutoRaise(True)
        self.btn_cancel.setProperty('liteToolbarButton', True)
        self.btn_cancel.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        self.btn_cancel.setToolTip(self._t("folders.cancel_build"))
        toolbar_layout.addWidget(self.btn_cancel)
        toolbar_layout.addStretch(1)
        folders_layout.addLayout(toolbar_layout)

        self.build_progress = QProgressBar(folders_group)
        self.build_progress.setTextVisible(True)
        self.build_progress.setRange(0, 1)
        self.build_progress.setValue(0)
        self.build_progress.hide()
        folders_layout.addWidget(self.build_progress)

        self.folder_list = QListWidget(folders_group)
        self.folder_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.folder_list.setSpacing(2)
        folders_layout.addWidget(self.folder_list, stretch=1)
        control_layout.addWidget(folders_group, stretch=1)

        self.matrix_tabs = QTabWidget(splitter)
        self.matrix_tabs.setTabsClosable(True)
        self.matrix_tabs.setMovable(True)
        self.matrix_tabs.setDocumentMode(True)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(CONTROL_PANEL_SPLITTER_SIZES))

        self.mode_combo = QComboBox(self)
        self._populate_mode_combo(DEFAULT_COMPARISON_MODE)

        self.thumbnail_size_spin = QSpinBox(self)
        self.thumbnail_size_spin.setRange(*THUMBNAIL_SIZE_RANGE)
        self.thumbnail_size_spin.setValue(DEFAULT_CELL_SIZE)

        self.layout_mode_combo = QComboBox(self)
        self._populate_layout_mode_combo(DEFAULT_MATRIX_LAYOUT_MODE)

        self.total_frames_spin = QSpinBox(self)
        self.total_frames_spin.setRange(*TOTAL_FRAMES_RANGE)
        self.total_frames_spin.setValue(DEFAULT_TOTAL_FRAMES)
        self.total_frames_spin.setToolTip(self._t("matrix.total_frames.tooltip"))

        self.frames_per_row_spin = QSpinBox(self)
        self.frames_per_row_spin.setRange(*FRAMES_PER_ROW_RANGE)
        self.frames_per_row_spin.setValue(DEFAULT_FRAMES_PER_ROW)
        self.frames_per_row_spin.setToolTip(self._t("matrix.frames_per_row.tooltip"))

        self.matrix_rows_spin = QSpinBox(self)
        self.matrix_rows_spin.setRange(*MATRIX_ROWS_RANGE)
        self.matrix_rows_spin.setValue(DEFAULT_MATRIX_ROWS)
        self.matrix_rows_spin.setToolTip(self._t("matrix.rows.tooltip"))

        self.matrix_columns_spin = QSpinBox(self)
        self.matrix_columns_spin.setRange(*MATRIX_COLUMNS_RANGE)
        self.matrix_columns_spin.setValue(DEFAULT_MATRIX_COLUMNS)
        self.matrix_columns_spin.setToolTip(self._t("matrix.columns.tooltip"))

        self.gradient_selector = GradientPresetSelectorWidget(self)
        self.gradient_range_selector = GradientRangeSelectorWidget(self)
        self.gradient_selector.set_selected_gradient(DEFAULT_GRADIENT_NAME, emit_signal=False)
        self.gradient_range_selector.set_gradient_name(DEFAULT_GRADIENT_NAME)
        self.gradient_range_selector.set_error_window(*DEFAULT_ERROR_WINDOW)

        self.error_score_view_combo = QComboBox(self)
        self.error_score_view_combo.addItem(self._t("matrix.score_view.relative"), "relative")
        self.error_score_view_combo.addItem(self._t("matrix.score_view.absolute"), "absolute")
        self.error_score_view_combo.setCurrentIndex(self.error_score_view_combo.findData(DEFAULT_SCORE_VIEW_MODE))
        self.error_score_view_combo.setEnabled(False)

        self.language_toggle_button = QToolButton(self._menu_bar)
        self.language_toggle_button.setAutoRaise(True)
        self.language_toggle_button.setObjectName(LITE_LANGUAGE_BUTTON_OBJECT_NAME)
        self.language_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_language_toggle_button()

    def _setup_menu_bar(self) -> None:
        self._menu_bar.clear()
        settings_menu = self._menu_bar.addMenu(self._t("menu.settings"))
        settings_menu.setToolTipsVisible(True)
        self._add_menu_widget(settings_menu.addMenu(self._t("menu.matrix")), self._build_matrix_settings_widget())

        error_view_menu = self._menu_bar.addMenu(self._t("menu.error_view"))
        error_view_menu.setToolTipsVisible(True)
        self._add_menu_widget(error_view_menu, self._build_error_view_settings_widget())

        self._menu_bar.setCornerWidget(self.language_toggle_button, Qt.Corner.TopRightCorner)

    def _build_matrix_settings_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self._matrix_pixel_size_row = self._build_setting_row(self._t("matrix.pixel_size"), self.thumbnail_size_spin)
        self._matrix_operation_row = self._build_setting_row(self._t("matrix.operation"), self.mode_combo)
        self._matrix_layout_mode_row = self._build_setting_row(self._t("matrix.layout_mode"), self.layout_mode_combo)
        self._matrix_total_frames_row = self._build_setting_row(self._t("matrix.total_frames"), self.total_frames_spin)
        self._matrix_frames_per_row_row = self._build_setting_row(self._t("matrix.frames_per_row"), self.frames_per_row_spin)
        self._matrix_rows_row = self._build_setting_row(self._t("matrix.rows"), self.matrix_rows_spin)
        self._matrix_columns_row = self._build_setting_row(self._t("matrix.columns"), self.matrix_columns_spin)

        for row_widget in (
            self._matrix_operation_row,
            self._matrix_pixel_size_row,
            self._matrix_layout_mode_row,
            self._matrix_total_frames_row,
            self._matrix_frames_per_row_row,
            self._matrix_rows_row,
            self._matrix_columns_row,
        ):
            layout.addWidget(row_widget)
        layout.addStretch(1)
        return widget

    def _build_error_view_settings_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(self._build_setting_row(self._t("matrix.score_view"), self.error_score_view_combo))
        layout.addWidget(self.gradient_selector)
        layout.addWidget(self.gradient_range_selector)
        layout.addStretch(1)
        return widget

    def _build_setting_row(self, title: str, control: QWidget) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel(title, row)
        label.setMinimumWidth(SETTINGS_LABEL_MIN_WIDTH)
        layout.addWidget(label)
        layout.addWidget(control, stretch=1)
        row._title_label = label  # type: ignore[attr-defined]
        return row

    def _build_preview_panel(self, parent: QWidget) -> LitePreviewPanel:
        group = QGroupBox(self._t("matrix.preview.group"), parent)
        form = QFormLayout(group)

        frame_title = QLabel(self._t("matrix.preview.frame"), group)
        frame_value = QLabel("-", group)
        frame_value.setWordWrap(True)
        form.addRow(frame_title, frame_value)

        absolute_title = QLabel(self._t("matrix.preview.absolute"), group)
        absolute_value = QLabel("-", group)
        absolute_value.setWordWrap(True)
        form.addRow(absolute_title, absolute_value)

        relative_title = QLabel(self._t("matrix.preview.relative"), group)
        relative_value = QLabel("-", group)
        relative_value.setWordWrap(True)
        form.addRow(relative_title, relative_value)

        return LitePreviewPanel(
            group=group,
            frame_title=frame_title,
            frame_value=frame_value,
            absolute_title=absolute_title,
            absolute_value=absolute_value,
            relative_title=relative_title,
            relative_value=relative_value,
        )

    def _retranslate_preview_panel(self, state: LiteMatrixTabState) -> None:
        preview = state.preview
        if preview is None:
            return
        preview.group.setTitle(self._t("matrix.preview.group"))
        preview.frame_title.setText(self._t("matrix.preview.frame"))
        preview.absolute_title.setText(self._t("matrix.preview.absolute"))
        preview.relative_title.setText(self._t("matrix.preview.relative"))

    def _set_row_label(self, row: QWidget | None, title: str) -> None:
        if row is None:
            return
        label = getattr(row, "_title_label", None)
        if isinstance(label, QLabel):
            label.setText(title)

    def _add_menu_widget(self, menu: QMenu, widget: QWidget) -> None:
        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)

    def _update_language_toggle_button(self) -> None:
        current_language = str(self._i18n.language or "en").lower()
        next_language = "RU" if current_language == "en" else "EN"
        self.language_toggle_button.setText(next_language)
        self.language_toggle_button.setToolTip(self._t("language.toggle_tooltip"))

    def _populate_mode_combo(self, selected_mode: ComparisonMode) -> None:
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItem(self._t("mode.first_minus_second"), ComparisonMode.FIRST_MINUS_SECOND)
        self.mode_combo.addItem(self._t("mode.second_minus_first"), ComparisonMode.SECOND_MINUS_FIRST)
        self.mode_combo.addItem(self._t("mode.disagreement"), ComparisonMode.DISAGREEMENT)
        self.mode_combo.addItem(self._t("mode.grayscale_diff"), ComparisonMode.GRAYSCALE_DIFF)
        index = self.mode_combo.findData(selected_mode)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.mode_combo.blockSignals(False)

    def _populate_layout_mode_combo(self, selected_mode: str) -> None:
        self.layout_mode_combo.blockSignals(True)
        self.layout_mode_combo.clear()
        self.layout_mode_combo.addItem(self._t("matrix.layout_mode.indexed"), "indexed_grid")
        self.layout_mode_combo.addItem(self._t("matrix.layout_mode.manual"), "manual_grid")
        index = self.layout_mode_combo.findData(str(selected_mode or DEFAULT_MATRIX_LAYOUT_MODE))
        self.layout_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.layout_mode_combo.blockSignals(False)

    def retranslate_ui(self) -> None:
        self.folders_group.setTitle(self._t("folders.group"))
        self.folders_info_label.setText(self._t("folders.info"))
        self.btn_add_folder.setToolTip(self._t("folders.add"))
        self.btn_clear_folders.setToolTip(self._t("folders.clear_all"))
        self.btn_set_base.setToolTip(self._t("folders.set_base"))
        self.btn_clear_base.setToolTip(self._t("folders.clear_base"))
        self.btn_build.setToolTip(self._t("folders.build"))
        self.btn_compute.setToolTip(self._t("folders.compute_mismatch"))
        self.btn_cancel.setToolTip(self._t("folders.cancel_build"))
        self.total_frames_spin.setToolTip(self._t("matrix.total_frames.tooltip"))
        self.frames_per_row_spin.setToolTip(self._t("matrix.frames_per_row.tooltip"))
        self.matrix_rows_spin.setToolTip(self._t("matrix.rows.tooltip"))
        self.matrix_columns_spin.setToolTip(self._t("matrix.columns.tooltip"))
        self._update_language_toggle_button()
        self.gradient_selector.retranslate_ui()
        self.gradient_range_selector.retranslate_ui()
        self._populate_mode_combo(self.mode_combo.currentData() or DEFAULT_COMPARISON_MODE)
        self._populate_layout_mode_combo(str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE))
        current_score_view = str(self.error_score_view_combo.currentData() or DEFAULT_SCORE_VIEW_MODE)
        self.error_score_view_combo.blockSignals(True)
        self.error_score_view_combo.clear()
        self.error_score_view_combo.addItem(self._t("matrix.score_view.relative"), "relative")
        self.error_score_view_combo.addItem(self._t("matrix.score_view.absolute"), "absolute")
        self.error_score_view_combo.setCurrentIndex(self.error_score_view_combo.findData(current_score_view))
        self.error_score_view_combo.blockSignals(False)
        self._set_row_label(self._matrix_pixel_size_row, self._t("matrix.pixel_size"))
        self._set_row_label(self._matrix_operation_row, self._t("matrix.operation"))
        self._set_row_label(self._matrix_layout_mode_row, self._t("matrix.layout_mode"))
        self._set_row_label(self._matrix_total_frames_row, self._t("matrix.total_frames"))
        self._set_row_label(self._matrix_frames_per_row_row, self._t("matrix.frames_per_row"))
        self._set_row_label(self._matrix_rows_row, self._t("matrix.rows"))
        self._set_row_label(self._matrix_columns_row, self._t("matrix.columns"))
        self._setup_menu_bar()
        if self.window() is not None:
            self.window().setWindowTitle(self._t("window.title"))
        if hasattr(self, "_presenter"):
            self._presenter._refresh_folder_rows()
            for state in self._presenter._tab_states.values():
                self._retranslate_preview_panel(state)
                self._presenter._update_matrix_preview(state)
            self._presenter._sync_layout_control_state()
            self._presenter._sync_action_buttons()

    def _toggle_language(self) -> None:
        current_language = str(self._i18n.language or "en").lower()
        language = "ru" if current_language == "en" else "en"
        self._i18n.set_language(language)
        self._t = self._i18n.tr
        self._settings_service.save_language(language)
        self._settings_service.sync()
        self.retranslate_ui()

    def _connect_signals(self) -> None:
        self.btn_add_folder.clicked.connect(self._presenter._add_folder)
        self.btn_clear_folders.clicked.connect(self._presenter._clear_folders)
        self.btn_set_base.clicked.connect(self._presenter._set_base_folder)
        self.btn_clear_base.clicked.connect(self._presenter._clear_base_folder)
        self.btn_build.clicked.connect(self._presenter._start_build)
        self.btn_compute.clicked.connect(self._presenter._start_compute_mismatches)
        self.btn_cancel.clicked.connect(self._presenter._request_cancel_build)

        self.mode_combo.currentIndexChanged.connect(self._presenter._on_comparison_mode_changed)
        self.layout_mode_combo.currentIndexChanged.connect(self._presenter._on_matrix_layout_mode_changed)
        self.thumbnail_size_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.total_frames_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.frames_per_row_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.matrix_rows_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.matrix_columns_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.gradient_selector.gradientChanged.connect(self._presenter._on_gradient_preset_changed)
        self.gradient_range_selector.rangeChanged.connect(self._presenter._on_error_window_changed)
        self.error_score_view_combo.currentIndexChanged.connect(self._presenter._on_error_score_view_changed)
        self.language_toggle_button.clicked.connect(self._toggle_language)

        self.matrix_tabs.currentChanged.connect(self._presenter._on_current_tab_changed)
        self.matrix_tabs.tabCloseRequested.connect(self._presenter._close_matrix_tab)

    def _create_matrix_tab(self, build_result: BuildResult, snapshot: dict[str, object]) -> LiteMatrixTabState:
        host = QWidget(self.matrix_tabs)
        matrix_layout = QHBoxLayout(host)
        matrix_layout.setContentsMargins(0, 0, 0, 0)

        matrix_view = MatrixListWidget(host)
        matrix_layout.addWidget(matrix_view, stretch=1)

        overview_host = QWidget(host)
        overview_layout = QVBoxLayout(overview_host)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        mini_map = MatrixMiniMapWidget(overview_host)
        overview_layout.addWidget(mini_map)

        preview = self._build_preview_panel(overview_host)
        overview_layout.addWidget(preview.group)
        overview_layout.addStretch(1)
        overview_host.setMaximumWidth(OVERVIEW_PANEL_MAX_WIDTH)
        matrix_layout.addWidget(overview_host)

        state = LiteMatrixTabState(
            widget=host,
            matrix_view=matrix_view,
            mini_map=mini_map,
            build_result=build_result,
            cell_size=int(snapshot["cell_size"]),
            layout_config=snapshot["layout_config"],
            gradient_name=str(snapshot["gradient_name"]),
            error_window=tuple(snapshot["error_window"]),
            score_view_mode=str(snapshot.get("score_view_mode") or DEFAULT_SCORE_VIEW_MODE),
            preview=preview,
        )

        matrix_view.recordSelected.connect(lambda record, s=state: self._presenter._on_record_selected(s, record))
        matrix_view.recordActivated.connect(lambda record, s=state: self._presenter._open_record_details(record, s))
        matrix_view.contextMenuRequested.connect(lambda record, pos, s=state: self._presenter._show_matrix_context_menu(s, record, pos))
        matrix_view.overviewChanged.connect(
            lambda image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position, s=state:
            self._presenter._on_matrix_overview_changed(
                s,
                image,
                visible_rect,
                selected_position,
                selected_blink_on,
                processing_positions,
                reference_position,
            )
        )
        self._presenter._refresh_score_view_controls(state)
        return state

    def menu_bar_widget(self) -> QMenuBar:
        return self._menu_bar

    def shutdown(self) -> None:
        self._presenter.shutdown()

    def closeEvent(self, event) -> None:
        self._presenter.shutdown()
        super().closeEvent(event)


class ValidationGradientLiteMainWindow(QMainWindow):
    """Provide a standalone host window for the lite widget."""

    def __init__(self) -> None:
        super().__init__()
        self._widget = ValidationGradientLiteWidget(self)
        self.setWindowTitle(self._widget._t("window.title"))
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setCentralWidget(self._widget)

    def plugin_widget(self) -> ValidationGradientLiteWidget:
        return self._widget

    def closeEvent(self, event) -> None:
        self._widget.shutdown()
        super().closeEvent(event)




