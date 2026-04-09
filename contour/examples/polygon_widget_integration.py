from __future__ import annotations

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from polygon_widget import PolygonExtractionWidget


class HostWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Polygon Widget Integration Example")

        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.polygon_widget = PolygonExtractionWidget()
        layout.addWidget(self.polygon_widget)

        self.polygon_widget.logMessage.connect(print)
        self.polygon_widget.imageProcessed.connect(self._on_image_processed)

        self.polygon_widget.set_input_directory(".")
        self.polygon_widget.set_output_directory("./polygon_outputs")

    def _on_image_processed(self, image_path: str, polygons: list) -> None:
        print(f"{image_path}: {len(polygons)} polygons")


if __name__ == "__main__":
    app = QApplication([])
    window = HostWindow()
    window.resize(1600, 900)
    window.show()
    app.exec()
