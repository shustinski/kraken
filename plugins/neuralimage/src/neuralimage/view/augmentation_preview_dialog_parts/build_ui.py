"""augmentation_preview_dialog: build ui."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)

from neuralimage.view.augmentation_preview_dialog_parts.preview_label import _PreviewLabel


class BuildUiMixin:
    def _build_ui(self) -> None:
        self.setWindowTitle(str(self._texts.get('window_title', 'Augmentation preview')))
        self.resize(1420, 860)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        left_widget = QWidget(self)
        left_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(6)
        self.prev_button = QPushButton(str(self._texts.get('prev_button', 'Previous')))
        self.next_button = QPushButton(str(self._texts.get('next_button', 'Next')))
        self.resample_button = QPushButton(str(self._texts.get('resample_button', 'Resample')))
        self.apply_to_main_button = QPushButton(
            str(
                self._texts.get(
                    'apply_to_main_button',
                    'Перенести в основное окно' if self._is_russian_ui else 'Apply to main window',
                )
            )
        )
        self.full_image_check_box = QCheckBox(
            str(
                self._texts.get(
                    'full_image_toggle',
                    'Показывать целиком' if self._is_russian_ui else 'Show full image',
                )
            )
        )
        self.full_image_check_box.setChecked(True)
        self.sample_label = QLabel('')
        self.sample_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        nav_row.addWidget(self.prev_button)
        nav_row.addWidget(self.next_button)
        nav_row.addWidget(self.resample_button)
        nav_row.addWidget(self.apply_to_main_button)
        nav_row.addWidget(self.full_image_check_box)
        nav_row.addWidget(self.sample_label, 1)
        left_layout.addLayout(nav_row)

        self.mode_label = QLabel('')
        self.mode_label.setWordWrap(True)
        left_layout.addWidget(self.mode_label)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(8)

        self.sample_list_group = QGroupBox(
            str(
                self._texts.get(
                    'sample_list_group',
                    'Изображения' if self._is_russian_ui else 'Images',
                )
            )
        )
        sample_list_layout = QVBoxLayout(self.sample_list_group)
        sample_list_layout.setContentsMargins(6, 6, 6, 6)
        self.sample_list_widget = QListWidget(self.sample_list_group)
        self.sample_list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.sample_list_widget.setMinimumWidth(140)
        sample_list_layout.addWidget(self.sample_list_widget)

        preview_widget = QWidget(self)
        preview_row = QHBoxLayout(preview_widget)
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(10)

        self.image_group = QGroupBox(str(self._texts.get('image_group', 'Image')))
        image_layout = QVBoxLayout(self.image_group)
        self.image_preview = _PreviewLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(180, 220)
        self.image_preview.setStyleSheet('border: 1px solid #666; background: #111;')
        image_layout.addWidget(self.image_preview)

        self.label_group = QGroupBox(str(self._texts.get('label_group', 'Label')))
        label_layout = QVBoxLayout(self.label_group)
        self.label_preview = _PreviewLabel()
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview.setMinimumSize(180, 220)
        self.label_preview.setStyleSheet('border: 1px solid #666; background: #111;')
        label_layout.addWidget(self.label_preview)

        preview_row.addWidget(self.image_group, 1)
        preview_row.addWidget(self.label_group, 1)
        self.content_splitter.addWidget(self.sample_list_group)
        self.content_splitter.addWidget(preview_widget)
        self.content_splitter.setStretchFactor(0, 0)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([260, 900])
        left_layout.addWidget(self.content_splitter, 1)

        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        right_scroll = QScrollArea(self)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(260)
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        normalization_group, augmentation_group = self._build_shared_training_transform_groups()
        self._attach_sidebar_widget(normalization_group, right_layout)
        self._attach_sidebar_widget(augmentation_group, right_layout)
        right_layout.addStretch(1)
        right_scroll.setWidget(right_content)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(right_scroll)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([1060, 360])

        self._populate_sample_list()
        root_layout.addWidget(self.main_splitter, 1)
