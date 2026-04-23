from __future__ import annotations

from PyQt6.QtCore import QObject, QRectF, QRunnable, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget

from ...application.use_cases.autotune import auto_tune_pipeline
from ...application.use_cases.processing import (
    PreparedImageRequest,
    PreviewProcessingRequest,
    prepare_image_for_preview,
    process_image_path,
)
from .image_conversion import cv_to_qimage


class PreviewImageView(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        scene = QGraphicsScene()
        super().__init__(scene, parent)
        self._scene = scene
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.GlobalColor.black)

    def set_image(self, image) -> None:
        if image is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(0, 0, 1, 1)
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_to_view()

    def fit_to_view(self) -> None:
        rect = self._scene.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()


class PreviewProcessingSignals(QObject):
    result = pyqtSignal(int, object)
    error = pyqtSignal(int, str)
    finished = pyqtSignal(int)


class PreparedImageSignals(QObject):
    result = pyqtSignal(int, str, object, object)
    error = pyqtSignal(int, str)
    finished = pyqtSignal(int)


class AutoTuneSignals(QObject):
    result = pyqtSignal(int, object)
    error = pyqtSignal(int, str)
    finished = pyqtSignal(int)


class PreviewProcessingRunnable(QRunnable):
    def __init__(self, request_id: int, request: PreviewProcessingRequest) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.request = PreviewProcessingRequest(
            image_path=request.image_path,
            pipeline_config=dict(request.pipeline_config),
            contour_settings=request.contour_settings,
            source_image=request.source_image,
            preprocessed_image=request.preprocessed_image,
        )
        self.signals = PreviewProcessingSignals()

    def run(self) -> None:
        try:
            result = process_image_path(
                image_path=self.request.image_path,
                pipeline_config=self.request.pipeline_config,
                contour_settings=self.request.contour_settings,
                source_image=self.request.source_image,
                preprocessed_image=self.request.preprocessed_image,
            )
            self.signals.result.emit(self.request_id, result)
        except Exception as exc:
            self.signals.error.emit(self.request_id, str(exc))
        finally:
            self.signals.finished.emit(self.request_id)


class PreparedImageRunnable(QRunnable):
    def __init__(self, request_id: int, request: PreparedImageRequest) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.request = PreparedImageRequest(
            image_path=request.image_path,
            source_image=request.source_image.copy(),
            pipeline_config=dict(request.pipeline_config),
        )
        self.signals = PreparedImageSignals()

    def run(self) -> None:
        try:
            preprocessed_image = prepare_image_for_preview(
                source_image=self.request.source_image,
                pipeline_config=self.request.pipeline_config,
            )
            self.signals.result.emit(
                self.request_id,
                self.request.image_path,
                preprocessed_image,
                self.request.pipeline_config,
            )
        except Exception as exc:
            self.signals.error.emit(self.request_id, str(exc))
        finally:
            self.signals.finished.emit(self.request_id)


class AutoTuneRunnable(QRunnable):
    def __init__(self, request_id: int, image_path: str, source_image, reference_polygons: list) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.image_path = str(image_path)
        self.source_image = source_image.copy()
        self.reference_polygons = [polygon.clone() for polygon in reference_polygons]
        self.signals = AutoTuneSignals()

    def run(self) -> None:
        try:
            result = auto_tune_pipeline(
                source_image=self.source_image,
                reference_polygons=self.reference_polygons,
            )
            self.signals.result.emit(self.request_id, result)
        except Exception as exc:
            self.signals.error.emit(self.request_id, str(exc))
        finally:
            self.signals.finished.emit(self.request_id)
