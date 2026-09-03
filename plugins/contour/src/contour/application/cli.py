from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from collections.abc import Sequence
from time import perf_counter


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]
    if any(item in {"-h", "--help"} for item in args):
        from ..infrastructure.runtime_config import config_int

        parser = argparse.ArgumentParser(prog="contour", description="Standalone launcher for Contour.")
        parser.add_argument("paths", nargs="*", help="Optional image files or a single directory to load on startup.")
        parser.add_argument("--input-dir", help="Input image directory.")
        parser.add_argument("--output-dir", help="Output directory for exported results.")
        parser.add_argument("--dataset-dir", help="Directory for the prepared training dataset.")
        parser.add_argument("--cif-dir", help="Directory with CIF overlays.")
        parser.add_argument("--pipeline-json", help="Path to pipeline JSON config.")
        parser.add_argument("--language", choices=("ru", "en"), default=None, help="UI language override.")
        parser.add_argument(
            "--width",
            type=int,
            default=config_int("window", "width", 1680, minimum=640),
            help="Initial window width.",
        )
        parser.add_argument(
            "--height",
            type=int,
            default=config_int("window", "height", 980, minimum=480),
            help="Initial window height.",
        )
        parser.add_argument("--no-qss", action="store_true", help="Do not apply the main application QSS theme.")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
        parser.add_argument("--log-file", default=None, help="Path to the log file.")
        parser.add_argument("--kraken-job-manifest", help="Kraken Agent job manifest (managed mode).")
        parser.add_argument("--kraken-result-manifest", help="Kraken Agent result manifest (managed mode).")
        parser.add_argument("--kraken-staging-root", help="Kraken Agent staging workspace (managed mode).")
        parser.add_argument(
            "--kraken-workspace-context",
            help="Kraken two-root local workspace context.",
        )
        parser.print_help()
        return

    from ..infrastructure.startup_profiler import StartupProfile

    startup_profile = StartupProfile.begin()
    try:
        batch_import_started_at = perf_counter()
        from ..batch_processor import configure_batch_runtime

        if startup_profile is not None:
            startup_profile.mark_interval("batch_runtime_import", batch_import_started_at)
            startup_profile.measure("batch_runtime_configure", configure_batch_runtime)
        else:
            configure_batch_runtime()

        bridge_import_started_at = perf_counter()
        from ..kraken_bridge import prepare_contour_launch

        if startup_profile is not None:
            startup_profile.mark_interval("launch_bridge_import", bridge_import_started_at)

        if startup_profile is None:
            kraken_session, args = prepare_contour_launch(args)
        else:
            kraken_session, args = startup_profile.measure("prepare_launch", lambda: prepare_contour_launch(args))

        import_started_at = perf_counter()
        from .bootstrap import build_application

        if startup_profile is not None:
            startup_profile.mark_interval("bootstrap_import", import_started_at)
            app, window = startup_profile.measure("application_build", lambda: build_application(args))
        else:
            app, window = build_application(args)

        if kraken_session is not None:
            if startup_profile is None:
                kraken_session.attach_return_action(window)
            else:
                startup_profile.measure("session_attach", lambda: kraken_session.attach_return_action(window))

        show_started_at = perf_counter()
        window.show()
        if startup_profile is not None:
            startup_profile.mark_interval("window_show", show_started_at)
            event_loop_started_at = perf_counter()

            def _finish_startup_profile() -> None:
                startup_profile.mark_interval("first_event_loop", event_loop_started_at)
                startup_profile.finish(status="interactive")

            from PyQt6.QtCore import QTimer

            QTimer.singleShot(0, _finish_startup_profile)
        app.exec()
        if startup_profile is not None:
            startup_profile.finish(status="stopped")
    except BaseException:
        if startup_profile is not None:
            startup_profile.finish(status="failed")
        raise


if __name__ == "__main__":
    mp.freeze_support()
    main()
