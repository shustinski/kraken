from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtGui import QImage

from ...i18n import tr
from ...utils import ensure_uint8


def cv_to_qimage(image: np.ndarray | None) -> QImage:
    if image is None:
        return QImage()
    data = ensure_uint8(image)
    if data.ndim == 2:
        height, width = data.shape
        qimage = QImage(data.data, width, height, data.strides[0], QImage.Format.Format_Grayscale8)
        return qimage.copy()
    if data.ndim == 3 and data.shape[2] == 3:
        rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        return qimage.copy()
    raise ValueError(tr("unsupported_qimage_shape", shape=data.shape))
