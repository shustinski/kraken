from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCloseEvent, QImage, QKeySequence, QMouseEvent, QPainter, QPen, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lib.rare_patch_masks import (
    collect_matching_sample_label_pairs,
    load_rare_patch_mask,
    save_rare_patch_mask,
)
from lib.ui_texts import get_ui_section


class RarePatchCanvas(QWidget):
    mask_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 420)
        self._base_image: QImage | None = None
        self._mask_overlay_image: QImage | None = None
        self._rare_mask = np.zeros((1, 1), dtype=np.uint8)
        self._undo_stack: list[np.ndarray] = []
        self._max_undo_steps = 100
        self._active_button = Qt.MouseButton.NoButton
        self._drag_start_pos: QPointF | None = None
        self._drag_current_pos: QPointF | None = None

    def set_base_image(self, image: QImage) -> None:
        self._base_image = image.copy()
        if self._rare_mask.shape != (image.height(), image.width()):
            resized = Image.fromarray(self._rare_mask, mode='L').resize(
                (image.width(), image.height()),
                resample=Image.Resampling.NEAREST,
            )
            self._rare_mask = np.array(resized, dtype=np.uint8, copy=True)
            self._rebuild_mask_overlay()
        self.update()

    def set_rare_mask(self, mask: np.ndarray) -> None:
        normalized = np.asarray(mask)
        if normalized.ndim == 3:
            normalized = normalized[..., 0]
        normalized = np.where(normalized > 0, 255, 0).astype(np.uint8, copy=False)
        if self._base_image is not None and normalized.shape != (self._base_image.height(), self._base_image.width()):
            resized = Image.fromarray(normalized, mode='L').resize(
                (self._base_image.width(), self._base_image.height()),
                resample=Image.Resampling.NEAREST,
            )
            normalized = np.array(resized, dtype=np.uint8, copy=True)
        self._rare_mask = normalized.copy()
        self._undo_stack.clear()
        self._rebuild_mask_overlay()
        self.update()

    def rare_mask(self) -> np.ndarray:
        return self._rare_mask.copy()

    def clear_mask(self) -> None:
        if self._rare_mask.size == 0:
            return
        if not np.any(self._rare_mask):
            return
        self._push_undo_state()
        self._rare_mask.fill(0)
        self._rebuild_mask_overlay()
        self.mask_changed.emit()
        self.update()

    def undo_last_action(self) -> None:
        if not self._undo_stack:
            return
        self._rare_mask = self._undo_stack.pop()
        self._rebuild_mask_overlay()
        self.mask_changed.emit()
        self.update()

    def selected_pixels(self) -> int:
        return int(np.count_nonzero(self._rare_mask))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        target_rect = self._target_image_rect()
        if self._base_image is not None and not target_rect.isEmpty():
            painter.drawImage(target_rect, self._base_image)
            if self._mask_overlay_image is not None:
                painter.drawImage(target_rect, self._mask_overlay_image)
        preview_rect = self._selection_widget_rect()
        if preview_rect is not None:
            pen = QPen(QColor(255, 255, 255, 220))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(preview_rect)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        if self._base_image is None or not self._target_image_rect().contains(event.position()):
            return
        self._active_button = event.button()
        self._drag_start_pos = event.position()
        self._drag_current_pos = event.position()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._active_button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._drag_current_pos = event.position()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == self._active_button and self._drag_start_pos is not None:
            self._drag_current_pos = event.position()
            self._apply_selection_rect(erase=event.button() == Qt.MouseButton.RightButton)
            self._active_button = Qt.MouseButton.NoButton
            self._drag_start_pos = None
            self._drag_current_pos = None
        self.update()
        event.accept()

    def leaveEvent(self, _event) -> None:
        if self._active_button == Qt.MouseButton.NoButton:
            self._drag_start_pos = None
            self._drag_current_pos = None
        self.update()

    def apply_selection_rect(self, left: int, top: int, right: int, bottom: int, *, erase: bool = False) -> None:
        image_rect = self._normalize_image_rect(left, top, right, bottom)
        if image_rect is None:
            return
        left, top, right, bottom = image_rect
        current = self._rare_mask[top:bottom, left:right]
        new_value = 0 if erase else 255
        if current.size == 0 or bool(np.all(current == new_value)):
            return
        self._push_undo_state()
        self._rare_mask[top:bottom, left:right] = new_value
        self._rebuild_mask_overlay()
        self.mask_changed.emit()
        self.update()

    def _apply_selection_rect(self, *, erase: bool) -> None:
        image_rect = self._current_selection_image_rect()
        if image_rect is None:
            return
        self.apply_selection_rect(*image_rect, erase=erase)

    def _rebuild_mask_overlay(self) -> None:
        if self._rare_mask.size == 0:
            self._mask_overlay_image = None
            return
        rgba = np.zeros((self._rare_mask.shape[0], self._rare_mask.shape[1], 4), dtype=np.uint8)
        selected = self._rare_mask > 0
        rgba[selected] = (255, 196, 0, 110)
        self._mask_overlay_image = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()

    def _push_undo_state(self) -> None:
        self._undo_stack.append(self._rare_mask.copy())
        if len(self._undo_stack) > self._max_undo_steps:
            self._undo_stack.pop(0)

    def _target_image_rect(self) -> QRectF:
        if self._base_image is None:
            return QRectF()
        bounds = QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -8.0)
        if bounds.width() <= 0 or bounds.height() <= 0:
            return QRectF()
        image_width = float(self._base_image.width())
        image_height = float(self._base_image.height())
        scale = min(bounds.width() / image_width, bounds.height() / image_height)
        width = image_width * scale
        height = image_height * scale
        x = bounds.x() + ((bounds.width() - width) / 2.0)
        y = bounds.y() + ((bounds.height() - height) / 2.0)
        return QRectF(x, y, width, height)

    def _map_widget_to_image(self, position: QPointF) -> tuple[int, int] | None:
        if self._base_image is None:
            return None
        target_rect = self._target_image_rect()
        if target_rect.isEmpty():
            return None
        image_width = self._base_image.width()
        image_height = self._base_image.height()
        clamped_x = min(max(position.x(), target_rect.x()), target_rect.right())
        clamped_y = min(max(position.y(), target_rect.y()), target_rect.bottom())
        rel_x = (clamped_x - target_rect.x()) / max(1.0, target_rect.width())
        rel_y = (clamped_y - target_rect.y()) / max(1.0, target_rect.height())
        pixel_x = int(round(rel_x * max(0, image_width - 1)))
        pixel_y = int(round(rel_y * max(0, image_height - 1)))
        pixel_x = min(max(0, pixel_x), image_width - 1)
        pixel_y = min(max(0, pixel_y), image_height - 1)
        return pixel_x, pixel_y

    def _selection_widget_rect(self) -> QRectF | None:
        image_rect = self._current_selection_image_rect()
        if image_rect is None or self._base_image is None:
            return None
        left, top, right, bottom = image_rect
        target_rect = self._target_image_rect()
        scale_x = target_rect.width() / float(self._base_image.width())
        scale_y = target_rect.height() / float(self._base_image.height())
        return QRectF(
            target_rect.x() + (left * scale_x),
            target_rect.y() + (top * scale_y),
            max(1.0, (right - left) * scale_x),
            max(1.0, (bottom - top) * scale_y),
        )

    def _current_selection_image_rect(self) -> tuple[int, int, int, int] | None:
        if self._drag_start_pos is None or self._drag_current_pos is None:
            return None
        start = self._map_widget_to_image(self._drag_start_pos)
        end = self._map_widget_to_image(self._drag_current_pos)
        if start is None or end is None:
            return None
        return self._normalize_image_rect(start[0], start[1], end[0] + 1, end[1] + 1)

    def _normalize_image_rect(self, left: int, top: int, right: int, bottom: int) -> tuple[int, int, int, int] | None:
        if self._base_image is None:
            return None
        image_width = self._base_image.width()
        image_height = self._base_image.height()
        normalized_left = min(left, right)
        normalized_top = min(top, bottom)
        normalized_right = max(left, right)
        normalized_bottom = max(top, bottom)
        normalized_left = min(max(0, normalized_left), image_width)
        normalized_top = min(max(0, normalized_top), image_height)
        normalized_right = min(max(0, normalized_right), image_width)
        normalized_bottom = min(max(0, normalized_bottom), image_height)
        if normalized_left >= normalized_right or normalized_top >= normalized_bottom:
            return None
        return normalized_left, normalized_top, normalized_right, normalized_bottom


class RarePatchEditorDialog(QDialog):
    def __init__(
        self,
        sample_folder: str | Path,
        label_folder: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample_folder = Path(sample_folder)
        self._label_folder = Path(label_folder)
        self._texts = get_ui_section('rare_patch_editor')
        self._pairs, self._load_error = collect_matching_sample_label_pairs(
            self._sample_folder,
            self._label_folder,
        )
        self._current_index = -1
        self._current_sample_image: Image.Image | None = None
        self._current_label_image: Image.Image | None = None
        self._current_sample_array: np.ndarray | None = None
        self._current_label_array: np.ndarray | None = None

        self.setModal(True)
        self.resize(1180, 820)
        self._build_ui()

        if self._load_error is not None:
            self._set_error_state(self._load_error)
        elif self._pairs:
            self._load_pair(0)
        else:
            self._set_error_state(self._text('empty_error', 'No matched sample/label pairs were found.'))

    def accept(self) -> None:
        self._save_current_mask()
        super().accept()

    def reject(self) -> None:
        self._save_current_mask()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_current_mask()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle(self._text('title', 'Rare patch oversampling'))

        self.canvas = RarePatchCanvas(self)
        self.canvas.mask_changed.connect(self._update_status_labels)

        self.prev_button = QPushButton(self._text('prev_button', 'Previous'))
        self.next_button = QPushButton(self._text('next_button', 'Next'))
        self.image_label = QLabel('')
        self.selection_label = QLabel('')
        self.hint_label = QLabel(
            self._text(
                'hint',
                'Drag with the left mouse button to add a rectangle, or with the right mouse button to erase it.',
            )
        )
        self.hint_label.setWordWrap(True)

        self.opacity_label = QLabel(self._text('opacity_label', 'Label opacity'))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)

        self.clear_button = QPushButton(self._text('clear_button', 'Clear current image'))
        self.save_button = QPushButton(self._text('save_button', 'Save'))
        self.close_button = QPushButton(self._text('close_button', 'Close'))
        self._previous_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        for shortcut in (self._previous_shortcut, self._next_shortcut, self._save_shortcut, self._undo_shortcut):
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

        top_row = QHBoxLayout()
        top_row.addWidget(self.prev_button)
        top_row.addWidget(self.next_button)
        top_row.addWidget(self.image_label, 1)
        top_row.addWidget(self.selection_label)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.opacity_label)
        controls_row.addWidget(self.opacity_slider, 1)
        controls_row.addWidget(self.clear_button)
        controls_row.addWidget(self.save_button)
        controls_row.addWidget(self.close_button)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.hint_label)
        layout.addLayout(controls_row)
        self.setLayout(layout)

        self.prev_button.clicked.connect(self._go_to_previous)
        self.next_button.clicked.connect(self._go_to_next)
        self.opacity_slider.valueChanged.connect(self._refresh_composite_image)
        self.clear_button.clicked.connect(self.canvas.clear_mask)
        self.save_button.clicked.connect(self._save_current_mask)
        self.close_button.clicked.connect(self.accept)
        self._previous_shortcut.activated.connect(self._go_to_previous)
        self._next_shortcut.activated.connect(self._go_to_next)
        self._save_shortcut.activated.connect(self._save_current_mask)
        self._undo_shortcut.activated.connect(self.canvas.undo_last_action)

    def _set_error_state(self, message: str) -> None:
        self.image_label.setText(self._text('error_label', 'Cannot open editor'))
        self.selection_label.setText('')
        self.hint_label.setText(message)
        for widget in (
            self.prev_button,
            self.next_button,
            self.opacity_slider,
            self.clear_button,
            self.save_button,
            self.canvas,
        ):
            widget.setEnabled(False)

    def _go_to_previous(self) -> None:
        if self._current_index <= 0:
            return
        self._save_current_mask()
        self._load_pair(self._current_index - 1)

    def _go_to_next(self) -> None:
        if self._current_index < 0 or self._current_index >= len(self._pairs) - 1:
            return
        self._save_current_mask()
        self._load_pair(self._current_index + 1)

    def _load_pair(self, index: int) -> None:
        if not (0 <= index < len(self._pairs)):
            return
        sample_path, label_path = self._pairs[index]
        with Image.open(sample_path) as sample_image:
            sample_image.load()
            self._current_sample_image = sample_image.convert('RGB').copy()
        with Image.open(label_path) as label_image:
            label_image.load()
            label_copy = label_image.convert('L').copy()
        if label_copy.size != self._current_sample_image.size:
            label_copy = label_copy.resize(self._current_sample_image.size, resample=Image.Resampling.NEAREST)
        self._current_label_image = label_copy
        self._current_sample_array = np.asarray(self._current_sample_image, dtype=np.uint8)
        self._current_label_array = np.asarray(self._current_label_image, dtype=np.uint8)
        rare_mask = load_rare_patch_mask(
            self._sample_folder,
            sample_path.stem,
            self._current_sample_image.size,
        )
        self.canvas.set_rare_mask(np.asarray(rare_mask, dtype=np.uint8))
        self._current_index = index
        self._refresh_composite_image()
        self._update_status_labels()
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self._pairs) - 1)

    def _refresh_composite_image(self) -> None:
        if self._current_sample_array is None or self._current_label_array is None:
            return
        sample = self._current_sample_array.astype(np.float32)
        label = self._current_label_array.astype(np.float32) / 255.0
        alpha = float(self.opacity_slider.value()) / 100.0
        overlay_color = np.array([255.0, 72.0, 72.0], dtype=np.float32)
        composed = sample * (1.0 - (label[..., None] * alpha)) + overlay_color * (label[..., None] * alpha)
        rgb = np.clip(composed, 0.0, 255.0).astype(np.uint8, copy=False)
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self.canvas.set_base_image(qimage)

    def _save_current_mask(self) -> None:
        if not (0 <= self._current_index < len(self._pairs)):
            return
        sample_path, _label_path = self._pairs[self._current_index]
        save_rare_patch_mask(self._sample_folder, sample_path.stem, self.canvas.rare_mask())
        self._update_status_labels()

    def _update_status_labels(self) -> None:
        if not (0 <= self._current_index < len(self._pairs)):
            return
        sample_path, _label_path = self._pairs[self._current_index]
        selected_pixels = self.canvas.selected_pixels()
        total_pixels = int(self.canvas.rare_mask().size) or 1
        percent = (selected_pixels / total_pixels) * 100.0
        self.image_label.setText(
            self._text(
                'image_template',
                '{index}/{total}: {name}',
            ).format(
                index=self._current_index + 1,
                total=len(self._pairs),
                name=sample_path.name,
            )
        )
        self.selection_label.setText(
            self._text(
                'selection_template',
                'Selected: {pixels} px ({percent:.2f}%)',
            ).format(
                pixels=selected_pixels,
                percent=percent,
            )
        )

    def _text(self, key: str, default: str) -> str:
        value = self._texts.get(key, default)
        return str(value if value is not None else default)
