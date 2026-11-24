"""builders: build path panel (sources tree and result folders)."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from contour.application.sources import SOURCE_IMAGES
from contour.ui.path_card import ROLE_DATASET, ROLE_OUTPUT, PathCard
from contour.ui.sources_panel import SourcesPanel


def _section_header(title_label: QLabel, trailing: QWidget | None = None) -> QWidget:
    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 4, 0, 0)
    header_layout.setSpacing(6)
    title_label.setObjectName("pathSectionTitle")
    title_label.setStyleSheet("QLabel#pathSectionTitle { font-weight: 600; }")
    header_layout.addWidget(title_label)
    header_layout.addStretch(1)
    if trailing is not None:
        header_layout.addWidget(trailing)
    return header


def _on_add_source_requested(self, kind: str, origin: str) -> None:
    if kind == SOURCE_IMAGES:
        if origin == "folder":
            self._select_input_directory()
        else:
            self._select_input_image_files()
    elif origin == "folder":
        self._select_cif_directory()
    else:
        self._merge_cif_files_dialog()


def build_path_panel(self) -> QWidget:
    # "Sources" tree at the top of the Files dock (see build_files_tab).
    self.path_group = QWidget()
    layout = QVBoxLayout(self.path_group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    self.sources_panel = SourcesPanel(self._tr)
    self.sources_panel.addRequested.connect(lambda kind, origin: _on_add_source_requested(self, kind, origin))
    self.sources_panel.removeRequested.connect(self.remove_source)
    self.sources_panel.refreshRequested.connect(self.refresh_image_list)
    self.sources_panel.sourceActivated.connect(self.set_source_filter)
    self.sources_panel.pathsDropped.connect(self._ingest_dropped_source_paths)
    self.refresh_button = self.sources_panel.refresh_button
    layout.addWidget(self.sources_panel)

    # Result folders go to the "Run & save" dock next to the save buttons.
    self.output_dir_edit = PathCard(ROLE_OUTPUT, self._tr)
    self.dataset_dir_edit = PathCard(ROLE_DATASET, self._tr)
    self.output_dir_edit.pathCommitted.connect(lambda _path: self._apply_output_directory_edit())
    self.dataset_dir_edit.pathCommitted.connect(lambda _path: self._apply_dataset_directory_edit())
    self.output_dir_edit.browseDirectoryRequested.connect(self._select_output_directory)
    self.dataset_dir_edit.browseDirectoryRequested.connect(self._select_dataset_directory)
    for card in (self.output_dir_edit, self.dataset_dir_edit):
        card.set_recent_provider(lambda role=card.role: self._recent_paths_for_role(role))
    self.path_outputs_label = QLabel()
    self.path_outputs_panel = QWidget()
    outputs_layout = QVBoxLayout(self.path_outputs_panel)
    outputs_layout.setContentsMargins(0, 0, 0, 0)
    outputs_layout.setSpacing(6)
    outputs_layout.addWidget(_section_header(self.path_outputs_label))
    outputs_layout.addWidget(self.output_dir_edit)
    outputs_layout.addWidget(self.dataset_dir_edit)
    return self.path_group
