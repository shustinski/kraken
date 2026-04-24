from __future__ import annotations

import logging

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from contour import PolygonExtractionWidget

_LOGGER = logging.getLogger(__name__)


class HostWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Polygon Widget Integration Example")

        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.contour = PolygonExtractionWidget()
        layout.addWidget(self.contour)

        self.contour.logMessage.connect(_LOGGER.info)
        self.contour.imageProcessed.connect(self._on_image_processed)

        self.contour.set_input_directory(".")
        self.contour.set_output_directory("./polygon_outputs")

    def _on_image_processed(self, image_path: str, polygons: list) -> None:
        _LOGGER.info("%s: %d polygons", image_path, len(polygons))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication([])
    window = HostWindow()
    window.resize(1600, 900)
    window.show()
    app.exec()
