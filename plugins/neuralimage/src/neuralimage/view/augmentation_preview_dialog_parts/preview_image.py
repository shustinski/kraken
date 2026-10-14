"""augmentation_preview_dialog: preview image."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
)
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)


class PreviewImageMixin:
    def _resolve_tip(self, key: str) -> str:
        return str(
            self._texts.get(
                f'{key}_tip',
                self._settings_texts.get(
                    f'{key}_tip',
                    self._settings_form_tooltips.get(key, ''),
                ),
            )
        )

    @staticmethod
    def _bounded_int(value: object, lower: int, upper: int) -> int:
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            resolved = int(lower)
        return max(int(lower), min(int(upper), resolved))

    @staticmethod
    def _bounded_float(value: object, lower: float, upper: float) -> float:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            resolved = float(lower)
        return max(float(lower), min(float(upper), resolved))

    @staticmethod
    def _set_preview_image(
        widget: QLabel,
        image_data: np.ndarray | None,
        *,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> None:
        if not isinstance(image_data, np.ndarray) or image_data.size == 0:
            widget.clear()
            return
        if image_data.ndim == 2:
            contiguous = np.ascontiguousarray(image_data)
            qimg = QImage(
                contiguous.tobytes(),
                contiguous.shape[1],
                contiguous.shape[0],
                contiguous.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()
        elif image_data.ndim == 3 and image_data.shape[2] == 3:
            contiguous = np.ascontiguousarray(image_data)
            qimg = QImage(
                contiguous.tobytes(),
                contiguous.shape[1],
                contiguous.shape[0],
                contiguous.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
        else:
            widget.clear()
            return
        scaled_width = max(1, int(target_width or widget.width()))
        scaled_height = max(1, int(target_height or widget.height()))
        pixmap = QPixmap.fromImage(qimg).scaled(
            scaled_width,
            scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        widget.setPixmap(pixmap)
