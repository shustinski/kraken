"""builders: build path panel."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from contour.ui.builders_parts.build_ui import _connect_line_edit_with_delay

def build_path_panel(self) -> QWidget:
    # The dock already provides the "Paths" title; do not repeat it in an
    # additional root group box.
    self.path_group = QWidget()
    layout = QVBoxLayout(self.path_group)
    layout.setContentsMargins(0, 0, 0, 0)

    self.input_dir_edit = QLineEdit()
    self.cif_dir_edit = QLineEdit()
    self.output_dir_edit = QLineEdit()
    self.dataset_dir_edit = QLineEdit()
    self.input_dir_label = QLabel("Input directory")
    self.cif_dir_label = QLabel("CIF overlay directory")
    self.output_dir_label = QLabel("Output directory")
    self.dataset_dir_label = QLabel("Dataset directory")
    self.browse_input_button = QPushButton()
    self.browse_cif_button = QPushButton()
    self.browse_output_button = QPushButton()
    self.browse_dataset_button = QPushButton()
    self.refresh_button = QPushButton()
    folder_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
    for button in (
        self.browse_input_button,
        self.browse_cif_button,
        self.browse_output_button,
        self.browse_dataset_button,
    ):
        self._configure_icon_only_button(button, folder_icon)
    self._configure_icon_only_button(self.refresh_button, self._refresh_files_icon())

    self.browse_input_button.clicked.connect(self._select_input_directory)
    self.browse_cif_button.clicked.connect(self._select_cif_directory)
    self.browse_output_button.clicked.connect(self._select_output_directory)
    self.browse_dataset_button.clicked.connect(self._select_dataset_directory)
    self.refresh_button.clicked.connect(self.refresh_image_list)
    self.pick_input_files_button = QPushButton()
    file_pick_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogStart)
    self._configure_icon_only_button(self.pick_input_files_button, file_pick_icon)
    self.pick_input_files_button.clicked.connect(self._select_input_image_files)
    self.merge_cif_files_button = QPushButton()
    self._configure_icon_only_button(self.merge_cif_files_button, file_pick_icon)
    self.merge_cif_files_button.clicked.connect(self._merge_cif_files_dialog)
    _connect_line_edit_with_delay(self, self.input_dir_edit, self._apply_input_directory_edit)
    _connect_line_edit_with_delay(self, self.cif_dir_edit, self._apply_cif_directory_edit)
    _connect_line_edit_with_delay(self, self.output_dir_edit, self._apply_output_directory_edit)
    _connect_line_edit_with_delay(self, self.dataset_dir_edit, self._apply_dataset_directory_edit)

    for label, edit, button in [
        (self.input_dir_label, self.input_dir_edit, self.browse_input_button),
        (self.output_dir_label, self.output_dir_edit, self.browse_output_button),
        (self.dataset_dir_label, self.dataset_dir_edit, self.browse_dataset_button),
    ]:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(edit, 1)
        if button is self.browse_input_button:
            row_layout.addWidget(self.pick_input_files_button)
        row_layout.addWidget(button)
        layout.addWidget(label)
        layout.addWidget(row)

    cif_row = QWidget()
    cif_row_layout = QHBoxLayout(cif_row)
    cif_row_layout.setContentsMargins(0, 0, 0, 0)
    cif_row_layout.setSpacing(6)
    cif_row_layout.addWidget(self.cif_dir_edit, 1)
    cif_row_layout.addWidget(self.merge_cif_files_button)
    cif_row_layout.addWidget(self.browse_cif_button)
    layout.addWidget(self.cif_dir_label)
    layout.addWidget(cif_row)

    refresh_row = QWidget()
    refresh_row_layout = QHBoxLayout(refresh_row)
    refresh_row_layout.setContentsMargins(0, 0, 0, 0)
    refresh_row_layout.setSpacing(6)
    refresh_row_layout.addWidget(self.refresh_button)
    refresh_row_layout.addStretch(1)
    layout.addWidget(refresh_row)
    return self.path_group
