"""builders: build visual panel."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from contour.graphics_view import PolygonEditorView

def build_visual_panel(self) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)

    # Keep a compatibility handle, but the scene itself is no longer wrapped
    # in a titled QGroupBox.
    self.editor_group = panel
    editor_layout = layout
    self.polygon_editor = PolygonEditorView()
    self.polygon_editor.installEventFilter(self)
    self.polygon_editor.viewport().installEventFilter(self)
    self.polygon_editor.set_frame_navigation_guard(self.confirm_ok_to_leave_current_vectors)
    self.polygon_editor.polygonsEdited.connect(self._on_polygons_edited)
    self.polygon_editor.polygonsEdited.connect(self._update_editor_selection_status)
    self.polygon_editor.activePolygonChanged.connect(self._update_editor_selection_status)
    self.polygon_editor.logRequested.connect(self._append_log)
    self.polygon_editor.imageClicked.connect(self._on_editor_image_clicked)
    self.polygon_editor.imageRegionSelected.connect(self._on_editor_image_region_selected)
    self.polygon_editor.rulerMeasurementChanged.connect(self._update_ruler_status)
    self.polygon_editor.toolChanged.connect(self._on_editor_tool_changed)
    self.polygon_editor.availableToolsChanged.connect(self._on_available_editor_tools_changed)
    self.polygon_editor.neighborFrameActivated.connect(self._on_neighbor_frame_activated)
    self.polygon_editor.frameNavigationRequested.connect(self._on_frame_navigation_requested)
    self.polygon_editor.currentFrameChanged.connect(self._on_editor_current_frame_changed)
    self.polygon_editor.viaDebugRequested.connect(self._on_via_debug_requested)
    self.polygon_editor.contactPlacementHotkeyPressed.connect(self._start_contact_placement_profile)
    self.polygon_editor.contactPlacementAttemptStarted.connect(self._ensure_contact_placement_profile)
    self.polygon_editor.contactPlacementAttemptFinished.connect(self._on_contact_placement_attempt_finished)
    self.polygon_editor.contactMultiSelectionStarted.connect(self._start_contact_multi_selection_profile)
    self.polygon_editor.contactMultiSelectionApplyStarted.connect(self._mark_contact_multi_selection_apply_started)
    self.polygon_editor.contactMultiSelectionFinished.connect(self._finish_contact_multi_selection_profile)
    self.polygon_editor.contactDeletionStarted.connect(self._start_contact_deletion_profile)
    self.polygon_editor.contactDeletionFinished.connect(self._on_contact_deletion_finished)
    self.polygon_editor.contactCopyStarted.connect(self._start_contact_copy_profile)
    self.polygon_editor.contactCopyFinished.connect(self._finish_contact_copy_profile)
    self.polygon_editor.contactPasteStarted.connect(self._start_contact_paste_profile)
    self.polygon_editor.contactPasteFinished.connect(self._finish_contact_paste_profile)
    self.polygon_editor.contactUndoStarted.connect(self._start_contact_undo_profile)
    self.polygon_editor.contactUndoFinished.connect(self._finish_contact_undo_profile)
    self.polygon_editor.contactRedoStarted.connect(self._start_contact_redo_profile)
    self.polygon_editor.contactRedoFinished.connect(self._finish_contact_redo_profile)
    self.polygon_editor.manualViaAdded.connect(self._on_manual_via_added)
    self.polygon_editor.recognizedViasDeleted.connect(self._on_recognized_vias_deleted)
    self.polygon_editor.metalOverlayDetailRequested.connect(self._on_metal_overlay_detail_requested)
    self.polygon_editor.filterPreviewHoldChanged.connect(self._on_filter_preview_hold_changed)
    self.polygon_editor.effectivePolygonCreateModeChanged.connect(self._on_effective_polygon_create_mode_changed)
    self.polygon_editor.polygonCreateModeChanged.connect(self._sync_polygon_mode_combo)
    self.polygon_editor.brushModeChanged.connect(self._sync_brush_mode_combo)
    self.polygon_editor.traceModeChanged.connect(self._sync_trace_mode_combo)
    self.polygon_editor.deleteVertexModeChanged.connect(self._sync_delete_vertex_mode_combo)
    self.editor_toolbar = self._build_editor_toolbar()
    self.editor_toolbar_scroll = QScrollArea()
    self.editor_toolbar_scroll.setObjectName("editorToolbarScroll")
    self.editor_toolbar_scroll.setWidget(self.editor_toolbar)
    self.editor_toolbar_scroll.setWidgetResizable(False)
    self.editor_toolbar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.editor_toolbar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.editor_toolbar_scroll.setFrameShape(QFrame.Shape.NoFrame)
    self.editor_toolbar_scroll.setMinimumWidth(min(760, max(180, self.editor_toolbar.sizeHint().width())))
    self.editor_toolbar_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    toolbar_scrollbar = self.editor_toolbar_scroll.horizontalScrollBar()
    toolbar_scrollbar_height = toolbar_scrollbar.sizeHint().height() if toolbar_scrollbar is not None else 0
    toolbar_height = self.editor_toolbar.sizeHint().height() + toolbar_scrollbar_height + 2
    self.editor_toolbar_scroll.setMinimumHeight(toolbar_height)
    self.editor_toolbar_scroll.setMaximumHeight(toolbar_height)
    editor_layout.addWidget(self.editor_toolbar_scroll)
    self.editor_scene_frame = QFrame()
    self.editor_scene_frame.setObjectName("editorSceneFrame")
    self.editor_scene_frame.setFrameShape(QFrame.Shape.NoFrame)
    scene_frame_layout = QVBoxLayout(self.editor_scene_frame)
    scene_frame_layout.setContentsMargins(0, 0, 0, 0)
    scene_frame_layout.setSpacing(0)
    scene_frame_layout.addWidget(self.polygon_editor)
    editor_layout.addWidget(self.editor_scene_frame, 1)

    return panel
