"""builders: build ui."""
from __future__ import annotations
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from ...gamification.ui import GamificationPanel

from contour.ui.builders_parts.module_state import USE_GAMIFICATION

def _connect_line_edit_with_delay(self, line_edit: QLineEdit, slot, *, delay_ms: int = 500) -> None:
    if not hasattr(self, "_line_edit_debounce_timers"):
        self._line_edit_debounce_timers = []

    timer = QTimer(self)
    timer.setSingleShot(True)
    timer.setInterval(delay_ms)
    timer.timeout.connect(slot)
    self._line_edit_debounce_timers.append(timer)

    line_edit.textEdited.connect(lambda *_args, _timer=timer: _timer.start())

def build_ui(self) -> None:
    root_layout = QVBoxLayout(self)

    self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
    self.main_splitter.setChildrenCollapsible(False)
    self.main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
    root_layout.addWidget(self.main_splitter, 1)

    self.left_controls_scroll = QScrollArea()
    self.left_controls_scroll.setWidgetResizable(True)
    self.left_controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
    self.left_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.left_controls_scroll.setMinimumWidth(280)
    self.left_controls_scroll.setMaximumWidth(560)
    self.left_controls_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    controls_container = QWidget()
    self.left_controls_scroll.setWidget(controls_container)
    controls_layout = QVBoxLayout(controls_container)
    self.control_tabs = self._build_tabs()
    controls_layout.addWidget(self.control_tabs, 1)
    self.main_splitter.addWidget(self.left_controls_scroll)
    self.visual_panel = self._build_visual_panel()
    self.main_splitter.addWidget(self.visual_panel)
    self.right_tabs = QTabWidget()
    self.right_tabs.setUsesScrollButtons(True)
    self.right_tabs.setMinimumWidth(220)
    self.right_tabs.setMaximumWidth(380)
    self.right_tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    self.files_tab = self._build_files_tab()
    self.right_tabs.addTab(self.files_tab, "Files")
    if hasattr(self, "_gamification_profile_service") and USE_GAMIFICATION:
        self.gamification_panel = GamificationPanel(self._gamification_profile_service, parent=self.right_tabs)
        self.gamification_panel.messageRequested.connect(self._append_log)
        self.right_tabs.addTab(self.gamification_panel, "Питомец")
    self.main_splitter.addWidget(self.right_tabs)
    self.main_splitter.setStretchFactor(0, 0)
    self.main_splitter.setStretchFactor(1, 1)
    self.main_splitter.setStretchFactor(2, 0)
    self.main_splitter.setSizes([340, 700, 240])

def _apply_size_hint_geometry(widget: QWidget, *, width: bool = True, height: bool = True) -> None:
    hint = widget.sizeHint()
    if width and hint.width() > 0:
        widget.setMinimumWidth(hint.width())
    if height and hint.height() > 0:
        widget.setMinimumHeight(hint.height())
