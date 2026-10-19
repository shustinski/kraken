"""main_window: update memory runtime label."""
from __future__ import annotations
import time
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QProgressBar,
)
import numpy as np

from neuralimage.view.main_window_parts.responsive_preview_label import ResponsivePreviewLabel


class UpdateMemoryRuntimeLabelMixin:
    def _update_memory_runtime_label(self) -> None:
        t = self._main_texts()
        no_runtime_data = (
            self._ram_mb is None
            and self._vram_alloc_mb is None
            and self._vram_reserved_mb is None
            and self._vram_used_mb is None
            and self._vram_total_mb is None
            and self._train_speed_batches_per_sec is None
        )
        if no_runtime_data:
            self.memory_usage_label.setText(str(t.get("memory_label_default", "Память: —")))
            return

        memory_unit = str(t.get("memory_unit", "МБ"))
        speed_unit = str(t.get("speed_unit", "batch/s"))
        ram_label = str(t.get("runtime_ram_label", "RAM"))
        vram_label = str(t.get("runtime_vram_label", "VRAM"))
        speed_label = str(t.get("runtime_speed_label", "Скорость"))

        ram_text = f"{ram_label}: {self._ram_mb:.0f} {memory_unit}" if self._ram_mb is not None else f"{ram_label}: —"
        if self._vram_used_mb is not None:
            total_text = f"/{self._vram_total_mb:.0f}" if self._vram_total_mb is not None else ""
            vram_text = f"{vram_label}: {self._vram_used_mb:.0f}{total_text} {memory_unit}"
        elif self._vram_alloc_mb is None:
            vram_text = f"{vram_label}: —"
        else:
            reserved_text = f"/{self._vram_reserved_mb:.0f}" if self._vram_reserved_mb is not None else ""
            vram_text = f"{vram_label}: {self._vram_alloc_mb:.0f}{reserved_text} {memory_unit}"
        speed_text = (
            f"{speed_label}: {self._train_speed_batches_per_sec:.2f} {speed_unit}"
            if self._train_speed_batches_per_sec is not None
            else f"{speed_label}: — {speed_unit}"
        )
        self.memory_usage_label.setText(f"{ram_text} | {vram_text} | {speed_text}")

    def _update_recognition_speed(self, current: int, total: int) -> None:
        if total <= 0:
            self._recognition_speed_images_per_sec = None
            self._recognition_started_at = None
            self._recognition_last_current = 0
            self._recognition_last_total = 0
            self._update_recognition_speed_label()
            return

        now = time.perf_counter()
        run_restarted = (
            self._recognition_started_at is None
            or total != self._recognition_last_total
            or current < self._recognition_last_current
            or current == 0
        )
        if run_restarted:
            self._recognition_started_at = now
            self._recognition_speed_images_per_sec = None
        elif self._recognition_started_at is not None and current > 0:
            elapsed_seconds = max(1e-6, now - self._recognition_started_at)
            self._recognition_speed_images_per_sec = current / elapsed_seconds

        self._recognition_last_current = current
        self._recognition_last_total = total
        self._update_recognition_speed_label()

    def _update_recognition_speed_label(self) -> None:
        t = self._main_texts()
        label_fallback = "Recognition speed" if self._ui_language == "en" else "Скорость распознавания"
        default_fallback = "Recognition speed: —" if self._ui_language == "en" else "Скорость распознавания: —"
        unit_fallback = "img/s" if self._ui_language == "en" else "изобр./с"
        default_text = str(t.get("recognition_speed_default", default_fallback))
        label = str(t.get("recognition_speed_label", label_fallback))
        unit = str(t.get("recognition_speed_unit", unit_fallback))
        if self._recognition_speed_images_per_sec is None:
            self.recognition_speed_label.setText(default_text)
            return
        self.recognition_speed_label.setText(
            f"{label}: {self._recognition_speed_images_per_sec:.2f} {unit}"
        )

    def _show_progress_mode(self, mode: str) -> None:
        resolved_mode = str(mode or "idle").strip().lower()
        if resolved_mode not in {"train", "recognition"}:
            resolved_mode = "idle"
        self._active_progress_mode = resolved_mode
        self.training_progress_widget.setVisible(resolved_mode == "train")
        self.recognition_progress_widget.setVisible(resolved_mode == "recognition")
        self.validation_quality_label.setVisible(True)
        self.performance_label.setVisible(True)

    def _set_training_batch_progress_bar(self) -> None:
        current, total = self._last_batch_progress
        suffix = ""
        if self._train_speed_batches_per_sec is not None:
            speed_unit = str(self._main_texts().get("speed_unit", "batch/s"))
            suffix = f" | {self._train_speed_batches_per_sec:.2f} {speed_unit}"
        self._set_progress_bar(self.batch_progress_bar, current, total, suffix=suffix)

    def _set_recognition_progress_bar(self) -> None:
        current, total = self._last_recognition_progress
        suffix = ""
        if self._recognition_speed_images_per_sec is not None:
            unit = str(self._main_texts().get("recognition_speed_unit", "img/s"))
            suffix = f" | {self._recognition_speed_images_per_sec:.2f} {unit}"
        self._set_progress_bar(self.recognition_progress_bar, current, total, suffix=suffix)

    def _set_preview_frame_name(self, sample_name: str) -> None:
        self._current_preview_sample_name = str(sample_name).strip()
        if self._current_preview_sample_name:
            template = str(self._main_texts().get("preview_current_frame", "Frame: {name}"))
            self.preview_frame_name_label.setText(template.format(name=self._current_preview_sample_name))
            return
        self.preview_frame_name_label.setText(
            str(self._main_texts().get("preview_current_frame_default", "Frame: —"))
        )

    def _set_preview_mode(self, mode: str) -> None:
        resolved_mode = "recognition" if str(mode).strip().lower() == "recognition" else "train"
        self._current_preview_mode = resolved_mode
        preview_label_column = getattr(self, "preview_label_column_widget", None)
        if preview_label_column is not None:
            preview_label_column.setVisible(resolved_mode != "recognition")
        self.preview_image_title_label.setText(self._main_texts()["preview_image"])
        self.preview_label_title_label.setText(self._main_texts()["preview_label"])
        self.preview_output_title_label.setText(self._main_texts()["preview_output"])

    def _reset_runtime_metrics(self) -> None:
        self._ram_mb = None
        self._vram_alloc_mb = None
        self._vram_reserved_mb = None
        self._vram_used_mb = None
        self._vram_total_mb = None
        self._train_speed_batches_per_sec = None
        self._recognition_speed_images_per_sec = None
        self._recognition_started_at = None
        self._recognition_last_current = 0
        self._recognition_last_total = 0
        self._last_validation_metrics = None
        self._last_performance_metrics = None

    @staticmethod
    def _sparsify_batch_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) > 1000:
            return points[::2]
        return points

    @staticmethod
    def _set_progress_bar(progress_bar: QProgressBar, current: int, total: int, *, suffix: str = ""):
        if total <= 0:
            progress_bar.setValue(0)
            progress_bar.setFormat("0%")
            return
        value = max(0, min(100, int((current / total) * 100)))
        progress_bar.setValue(value)
        progress_bar.setFormat(f"{value}% ({current}/{total}){suffix}")

    @staticmethod
    def _set_preview_image(widget: QLabel, image_data):
        if not isinstance(image_data, np.ndarray) or image_data.size == 0:
            return
        arr = np.ascontiguousarray(image_data)
        if arr.ndim == 2:
            qimg = QImage(arr.tobytes(), arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_Grayscale8).copy()
        elif arr.ndim == 3 and arr.shape[2] == 3:
            qimg = QImage(arr.tobytes(), arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_RGB888).copy()
        else:
            return
        widget.setText("")
        pix = QPixmap.fromImage(qimg)
        if isinstance(widget, ResponsivePreviewLabel):
            widget.set_source_pixmap(pix)
            return
        widget.setPixmap(
            pix.scaled(
                widget.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _set_start_enabled(self, ok: bool):
        self.btn_start.setEnabled(True)
        self.btn_start.setToolTip('' if ok else 'Нажмите кнопку, чтобы увидеть причину невозможности запуска.')
        self.btn_start.setStyleSheet(
            """background-color: #4CAF50; color: white; padding: 8px 20px;
               border: none; border-radius: 5px;"""
            if ok
            else """background-color: #C62828; color: white; padding: 8px 20px;
               border: none; border-radius: 5px;"""
        )
