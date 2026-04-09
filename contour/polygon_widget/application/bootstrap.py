from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

from PyQt6.QtWidgets import QApplication

from .model import PolygonWidgetApplicationModel, StartupConfiguration
from .presenter import PolygonWidgetPresenter
from .styles import load_shared_stylesheet
from .view import PolygonWidgetStandaloneWindow


@dataclass(slots=True)
class PolygonWidgetApplicationComponents:
    app: QApplication
    model: PolygonWidgetApplicationModel
    view: PolygonWidgetStandaloneWindow
    presenter: PolygonWidgetPresenter


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
    stylesheet = load_shared_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)


def _build_model(args: argparse.Namespace) -> PolygonWidgetApplicationModel:
    return PolygonWidgetApplicationModel(
        language=args.language,
        width=args.width,
        height=args.height,
        startup=StartupConfiguration.from_cli(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cif_dir=args.cif_dir,
            pipeline_json=args.pipeline_json,
            paths=args.paths,
        ),
    )


def assemble_application(argv: Sequence[str] | None = None) -> PolygonWidgetApplicationComponents:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication.instance() or QApplication(qt_argv)
    app.setOrganizationName("ViaLaNet")
    app.setApplicationName("PolygonWidget")
    if not args.no_qss:
        _try_apply_app_qss(app)

    model = _build_model(args)
    view = PolygonWidgetStandaloneWindow()
    presenter = PolygonWidgetPresenter(model=model, view=view)
    view.set_presenter(presenter)
    presenter.initialize()
    return PolygonWidgetApplicationComponents(
        app=app,
        model=model,
        view=view,
        presenter=presenter,
    )


def build_application(argv: Sequence[str] | None = None) -> tuple[QApplication, PolygonWidgetStandaloneWindow]:
    components = assemble_application(argv)
    return components.app, components.view
