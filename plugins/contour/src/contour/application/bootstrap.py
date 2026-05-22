from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from kraken_core.qt import configure_application_identity
from PyQt6.QtWidgets import QApplication, QMessageBox

from ..__version__ import __version__
from ..infrastructure import WidgetAppearanceSettingsStore
from ..infrastructure.logging import configure_logging
from .model import ContourApplicationModel, StartupConfiguration
from .presenter import ContourPresenter
from .styles import load_stylesheet
from .view import ContourStandaloneWindow

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ContourApplicationComponents:
    app: QApplication
    model: ContourApplicationModel
    view: ContourStandaloneWindow
    presenter: ContourPresenter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contour",
        description="Standalone launcher for Contour.",
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging to console and file.",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        default=None,
        help="Path to the log file. Defaults to %%LOCALAPPDATA%%/ViaLaNet/Contour/logs/app.log on Windows.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _try_apply_app_qss(app: QApplication) -> None:
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)


def _build_model(
    args: argparse.Namespace,
    appearance_settings_store: WidgetAppearanceSettingsStore,
) -> ContourApplicationModel:
    return ContourApplicationModel(
        language=args.language or appearance_settings_store.load_language(),
        theme=None if args.no_qss else appearance_settings_store.load_theme(),
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


def _install_global_excepthook(log_file: Path) -> None:
    """Install ``sys.excepthook`` that logs the traceback and shows a dialog."""
    previous_hook = sys.excepthook

    def _hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, exc_tb)
            return

        _LOGGER.critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            app = QApplication.instance()
            if app is not None:
                box = QMessageBox()
                box.setIcon(QMessageBox.Icon.Critical)
                box.setWindowTitle("Contour — unexpected error")
                box.setText("An unexpected error occurred.")
                box.setInformativeText(f"Details were written to:\n{log_file}")
                box.setDetailedText(formatted)
                box.setStandardButtons(QMessageBox.StandardButton.Close)
                box.exec()
        except Exception:
            _LOGGER.exception("Failed to display crash dialog")

    sys.excepthook = _hook


def assemble_application(argv: Sequence[str] | None = None) -> ContourApplicationComponents:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    log_file = configure_logging(verbose=args.verbose, log_file=args.log_file)
    _install_global_excepthook(log_file)
    _LOGGER.info("Starting Contour %s (log file: %s)", __version__, log_file)

    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication.instance() or QApplication(qt_argv)
    assert isinstance(app, QApplication)
    app.setOrganizationName("ViaLaNet")
    app.setApplicationName("Contour")
    app.setApplicationVersion(__version__)
    configure_application_identity(app, app_id="ViaLaNet.Contour", icon_name="contour")
    if not args.no_qss:
        _try_apply_app_qss(app)

    appearance_settings_store = WidgetAppearanceSettingsStore()
    model = _build_model(args, appearance_settings_store)
    view = ContourStandaloneWindow(appearance_settings_store=appearance_settings_store)
    presenter = ContourPresenter(model=model, view=view)
    view.set_presenter(presenter)
    presenter.initialize()
    return ContourApplicationComponents(
        app=app,
        model=model,
        view=view,
        presenter=presenter,
    )


def build_application(argv: Sequence[str] | None = None) -> tuple[QApplication, ContourStandaloneWindow]:
    components = assemble_application(argv)
    return components.app, components.view
