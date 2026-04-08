from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Sequence

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow

from .i18n import active_language
from .widget import PolygonExtractionWidget


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polygon-widget",
        description="Standalone launcher for PolygonExtractionWidget.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional image files or a single directory to load on startup.",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        help="Input image directory.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory for exported results.",
    )
    parser.add_argument(
        "--cif-dir",
        dest="cif_dir",
        help="Directory with CIF overlays.",
    )
    parser.add_argument(
        "--pipeline-json",
        dest="pipeline_json",
        help="Path to pipeline JSON config.",
    )
    parser.add_argument(
        "--language",
        choices=("ru", "en"),
        default=None,
        help="UI language override.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1680,
        help="Initial window width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=980,
        help="Initial window height.",
    )
    parser.add_argument(
        "--no-qss",
        action="store_true",
        help="Do not apply the main application QSS theme.",
    )
    return parser


def _try_apply_app_qss(app: QApplication) -> None:
    try:
        from controller.app_controller import load_qss_from_resource
    except Exception:
        return
    try:
        app.setStyleSheet(load_qss_from_resource())
    except Exception:
        return


def _try_apply_app_icon(window: QMainWindow) -> None:
    try:
        from lib.runtime_paths import resolve_internal_path
    except Exception:
        return
    icon_path = resolve_internal_path("icon.png")
    if Path(icon_path).exists():
        window.setWindowIcon(QIcon(str(icon_path)))


class PolygonWidgetStandaloneWindow(QMainWindow):
    def __init__(self, *, language: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Polygon Extraction")
        self.widget = PolygonExtractionWidget(self)
        self.setCentralWidget(self.widget)
        self.widget.set_ui_language(active_language(language))
        self.widget.logMessage.connect(self.statusBar().showMessage)
        self.widget.imageProcessed.connect(self._on_image_processed)
        _try_apply_app_icon(self)

    def apply_startup_configuration(
        self,
        *,
        input_dir: str | None = None,
        output_dir: str | None = None,
        cif_dir: str | None = None,
        pipeline_json: str | None = None,
        paths: list[str] | None = None,
    ) -> None:
        if output_dir:
            self.widget.set_output_directory(output_dir)
        if cif_dir:
            self.widget.set_cif_directory(cif_dir)
        if pipeline_json:
            pipeline_path = Path(pipeline_json)
            payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
            self.widget.set_pipeline(payload)
        if input_dir:
            self.widget.set_input_directory(input_dir)
        if paths:
            normalized_paths = [str(Path(path)) for path in paths]
            file_paths = [path for path in normalized_paths if Path(path).is_file()]
            directory_paths = [path for path in normalized_paths if Path(path).is_dir()]
            if file_paths:
                self.widget.load_images(file_paths)
            elif directory_paths:
                self.widget.set_input_directory(directory_paths[0])

    def _on_image_processed(self, image_path: str, polygons: list) -> None:
        self.statusBar().showMessage(f"{Path(image_path).name}: {len(polygons)} polygons", 5000)


def build_application(argv: Sequence[str] | None = None) -> tuple[QApplication, PolygonWidgetStandaloneWindow]:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    app = QApplication.instance() or QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setOrganizationName("NeuralImage")
    app.setApplicationName("PolygonWidget")
    if not args.no_qss:
        _try_apply_app_qss(app)

    window = PolygonWidgetStandaloneWindow(language=args.language)
    window.resize(max(640, args.width), max(480, args.height))
    window.apply_startup_configuration(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cif_dir=args.cif_dir,
        pipeline_json=args.pipeline_json,
        paths=list(args.paths),
    )
    return app, window


def main(argv: Sequence[str] | None = None) -> None:
    app, window = build_application(argv)
    window.show()
    app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    main()
