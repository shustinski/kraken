"""builders: build files tab."""
from __future__ import annotations
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from contour.ui.frame_matrix_view import FrameMatrixGraphicsView
from contour.ui.frame_path_list_model import FramePathListView
from contour.ui.status_list_delegate import attach_status_row_delegate

def build_files_tab(self) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)

    self.files_scan_progress_bar = QProgressBar()
    self.files_scan_progress_bar.setRange(0, 100)
    self.files_scan_progress_bar.setValue(0)
    self.files_scan_progress_bar.setTextVisible(True)
    self.files_scan_progress_bar.setVisible(False)

    self.image_list = FramePathListView()
    attach_status_row_delegate(self.image_list)
    # Without uniform sizes QListView probes every row for layout (~N data()/rowCount).
    self.image_list.setUniformItemSizes(True)
    self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    self.image_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    self.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.image_list.customContextMenuRequested.connect(self._on_image_list_context_menu)

    self.image_vector_list = QListWidget()
    attach_status_row_delegate(self.image_vector_list)
    self.image_vector_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    self.image_vector_list.itemClicked.connect(self._on_asset_image_item_clicked)
    self.image_vector_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.image_vector_list.customContextMenuRequested.connect(
        lambda pos, list_widget=self.image_vector_list: self._on_asset_image_list_context_menu(list_widget, pos)
    )
    self.image_only_list = QListWidget()
    attach_status_row_delegate(self.image_only_list)
    self.image_only_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    self.image_only_list.itemClicked.connect(self._on_asset_image_item_clicked)
    self.image_only_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.image_only_list.customContextMenuRequested.connect(
        lambda pos, list_widget=self.image_only_list: self._on_asset_image_list_context_menu(list_widget, pos)
    )
    self.vector_only_list = QListWidget()
    attach_status_row_delegate(self.vector_only_list)
    self.vector_only_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    self.vector_only_list.itemClicked.connect(self._on_asset_vector_item_clicked)

    self.asset_view_tabs = QTabWidget()
    self.asset_view_tabs.setUsesScrollButtons(True)
    self.asset_view_tabs.addTab(self.image_list, "All")
    self.asset_view_tabs.addTab(self.image_vector_list, "Image+Vector")
    self.asset_view_tabs.addTab(self.image_only_list, "Image only")
    self.asset_view_tabs.addTab(self.vector_only_list, "Vector only")

    self.thumbnail_grid_label = QLabel("")
    self.thumbnail_grid_label.setVisible(False)
    self.thumbnail_grid = FrameMatrixGraphicsView()
    self.thumbnail_grid.setViewMode(QListWidget.ViewMode.IconMode)
    self.thumbnail_grid.setResizeMode(QListWidget.ResizeMode.Fixed)
    self.thumbnail_grid.setMovement(QListWidget.Movement.Static)
    self.thumbnail_grid.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    self.thumbnail_grid.setUniformItemSizes(True)
    self.thumbnail_grid.setWrapping(True)
    self.thumbnail_grid.setSpacing(0)
    self.thumbnail_grid.setIconSize(QSize(64, 48))
    self.thumbnail_grid.setContentsMargins(0, 0, 0, 0)
    self.thumbnail_grid.setFrameShape(QFrame.Shape.NoFrame)
    self.thumbnail_grid.setStyleSheet(
        "QListWidget { padding: 0px; border: 0px; }QListWidget::item { margin: 0px; padding: 0px; border: 0px; }"
    )
    self.thumbnail_grid.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.thumbnail_grid.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.thumbnail_grid.frameNavigationRequested.connect(self._on_frame_navigation_requested)
    self.thumbnail_grid.frameContextMenuRequested.connect(self._on_frame_matrix_context_menu)
    self.thumbnail_grid.installEventFilter(self)
    self.thumbnail_grid.viewport().installEventFilter(self)
    self.thumbnail_grid_scroll_area = self.thumbnail_grid
    self.thumbnail_grid_scroll_area.setWidgetResizable(False)
    self.thumbnail_grid_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
    self.thumbnail_grid_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.thumbnail_grid_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.thumbnail_grid_scroll_area.setMinimumHeight(96)
    thumbnail_h_scroll = self.thumbnail_grid_scroll_area.horizontalScrollBar()
    thumbnail_v_scroll = self.thumbnail_grid_scroll_area.verticalScrollBar()
    if thumbnail_h_scroll is not None:
        thumbnail_h_scroll.valueChanged.connect(self._on_thumbnail_viewport_changed)
    if thumbnail_v_scroll is not None:
        thumbnail_v_scroll.valueChanged.connect(self._on_thumbnail_viewport_changed)
    self.thumbnail_grid.thumbnailLodChanged.connect(self._on_thumbnail_lod_changed)
    self.thumbnail_matrix_panel = QWidget()
    thumbnail_matrix_layout = QVBoxLayout(self.thumbnail_matrix_panel)
    thumbnail_matrix_layout.setContentsMargins(0, 0, 0, 0)
    thumbnail_matrix_layout.setSpacing(4)
    thumbnail_matrix_layout.addWidget(self.thumbnail_grid_scroll_area, 1)

    # Kept as an internal CIF-status model for reload/status logic; it is no longer a visible sidebar mode.
    self.vector_list = QListWidget()
    attach_status_row_delegate(self.vector_list)
    self.vector_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    self.vector_list.itemClicked.connect(self._on_vector_item_navigate_request)
    self.vector_list.setVisible(False)
    overlay_buttons_row = QWidget()
    overlay_buttons_layout = QVBoxLayout(overlay_buttons_row)
    overlay_buttons_layout.setContentsMargins(0, 0, 0, 0)
    overlay_buttons_layout.setSpacing(6)
    self.reload_cif_selected_button = QPushButton()
    self.reload_cif_selected_button.clicked.connect(self._reload_cif_overlays_for_selected_vectors)
    self.reload_cif_for_frames_button = QPushButton()
    self.reload_cif_for_frames_button.clicked.connect(self._reload_cif_overlays_for_selected_images)
    overlay_buttons_layout.addWidget(self.reload_cif_selected_button)
    overlay_buttons_layout.addWidget(self.reload_cif_for_frames_button)
    self.reload_cif_selected_button.setVisible(False)

    layout.addWidget(self.files_scan_progress_bar)
    layout.addWidget(self.asset_view_tabs, 1)
    layout.addWidget(self.thumbnail_matrix_panel, 0)
    layout.addWidget(overlay_buttons_row)

    self.extra_layers_group = QGroupBox("Additional layers")
    self.extra_layers_form = QFormLayout(self.extra_layers_group)
    self._configure_compact_form(self.extra_layers_form)
    self.extra_layers_widget = QWidget()
    extra_layers_layout = QVBoxLayout(self.extra_layers_widget)
    extra_layers_layout.setContentsMargins(0, 0, 0, 0)
    extra_layers_layout.setSpacing(6)
    self.extra_layers_list = QListWidget()
    self.extra_layers_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
    self.extra_layers_list.setDefaultDropAction(Qt.DropAction.MoveAction)
    self.extra_layers_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    self.extra_layers_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.extra_layers_list.setMaximumHeight(96)
    extra_layers_layout.addWidget(self.extra_layers_list)
    self.add_extra_layers_button = QPushButton("+")
    self.add_extra_layers_button.setMinimumHeight(28)
    self.add_extra_layers_button.setStyleSheet(
        "QPushButton { background-color: #16A34A; color: white; font-weight: 700; border-radius: 4px; padding: 2px 8px; }"
    )
    extra_layers_layout.addWidget(self.add_extra_layers_button)

    self.add_extra_layers_button.clicked.connect(self._load_extra_layers)
    extra_layers_model = self.extra_layers_list.model()
    if extra_layers_model is not None:
        extra_layers_model.rowsMoved.connect(self._on_extra_layers_rows_moved)

    self.extra_layers_form.addRow(self.extra_layers_widget)
    self.extra_layers_label_widget = None
    self.extra_layer_path_label_widget = None
    self.extra_layer_opacity_label_widget = None
    self.extra_layer_dx_label_widget = None
    self.extra_layer_dy_label_widget = None
    layout.addWidget(self.extra_layers_group)

    # The containing QDockWidget is titled "Run" / "Обработка".
    self.run_group = QWidget()
    run_layout = QGridLayout(self.run_group)
    run_layout.setContentsMargins(0, 0, 0, 0)
    self.process_current_button = QPushButton()
    self.process_current_button.clicked.connect(self._process_current_image_and_save)
    self.batch_button = QPushButton()
    self.batch_button.clicked.connect(self._confirm_and_start_batch_processing)
    self.stop_batch_button = QPushButton()
    self.stop_batch_button.clicked.connect(self.stop_batch_processing)
    self.antialias_opened_cif_button = QPushButton()
    self.antialias_opened_cif_button.setIcon(self._create_editor_action_icon("antialias_all"))
    self.antialias_opened_cif_button.clicked.connect(self._antialias_opened_cif_files)
    self.fix_internal_contours_button = QPushButton()
    self.fix_internal_contours_button.clicked.connect(self._fix_internal_contours_all_cifs)
    self._configure_icon_only_button(
        self.process_current_button,
        self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
    )
    self._configure_icon_only_button(
        self.batch_button,
        self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward),
    )
    self._configure_icon_only_button(
        self.stop_batch_button,
        self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
    )
    self.save_current_button = QPushButton("Save current result")
    self.save_current_button.clicked.connect(self.save_current_result)
    self.export_dataset_button = QPushButton("Export frame to dataset")
    self.export_dataset_button.clicked.connect(self.export_current_frame_to_dataset)
    self.dataset_mode_checkbox = QCheckBox("Dataset mode")
    run_buttons_row = QWidget()
    run_buttons_layout = QHBoxLayout(run_buttons_row)
    run_buttons_layout.setContentsMargins(0, 0, 0, 0)
    run_buttons_layout.setSpacing(6)
    run_buttons_layout.addWidget(self.process_current_button)
    run_buttons_layout.addWidget(self.batch_button)
    run_buttons_layout.addWidget(self.stop_batch_button)
    run_buttons_layout.addStretch(1)
    run_layout.addWidget(run_buttons_row, 0, 0, 1, 2)
    self.batch_progress_bar = QProgressBar()
    self.batch_progress_bar.setRange(0, 100)
    self.batch_progress_bar.setValue(0)
    self.batch_progress_bar.setFormat("%p% (%v/%m)")
    self.batch_progress_bar.setTextVisible(True)
    self.batch_progress_bar.setVisible(False)
    run_layout.addWidget(self.save_current_button, 1, 0, 1, 2)
    run_layout.addWidget(self.export_dataset_button, 2, 0, 1, 2)
    run_layout.addWidget(self.dataset_mode_checkbox, 3, 0, 1, 2)
    run_layout.addWidget(self.antialias_opened_cif_button, 4, 0, 1, 2)
    run_layout.addWidget(self.fix_internal_contours_button, 5, 0, 1, 2)
    run_layout.addWidget(self.batch_progress_bar, 6, 0, 1, 2)
    run_layout.addWidget(self.save_group, 7, 0, 1, 2)
    run_layout.setRowStretch(8, 1)
    layout.addWidget(self.run_group)
    return tab
