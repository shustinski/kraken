"""Zoom smoothness: time of one wheel step in the matrix (50 000 frames) and in the frame window.

    python plugins/karakal/benchmarks/benchmark_zoom.py [--frames 50000] [--size 2000]

For every wheel notch it prints how long the event handling and the repaint of the visible
area took. Smooth zoom needs every animation frame under ~16 ms.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication


def _wheel(view, delta: int, *, ctrl: bool) -> QWheelEvent:
    center = QPointF(view.viewport().width() / 2.0, view.viewport().height() / 2.0)
    return QWheelEvent(
        center,
        QPointF(view.viewport().mapToGlobal(center.toPoint())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _settle(app: QApplication, ms: int) -> None:
    end = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.002)


def _run_steps(app, view, steps: int, delta: int, *, ctrl: bool) -> dict[str, list[float]]:
    """Wheel notches ~8 per second; every real paint of the view is timed."""

    paints: list[float] = view.paint_ms
    paints.clear()
    events: list[float] = []
    run_started = time.perf_counter()
    for _ in range(steps):
        started = time.perf_counter()
        view.wheelEvent(_wheel(view, delta, ctrl=ctrl))
        events.append((time.perf_counter() - started) * 1000.0)
        until = time.perf_counter() + 0.12
        while time.perf_counter() < until:
            app.processEvents()
            time.sleep(0.001)
    elapsed = time.perf_counter() - run_started
    return {"events": events, "paints": paints, "fps": [len(paints) / elapsed]}


def _timed(view_class):
    """The view class with every paint timed (Qt only sees paintEvent defined on a class)."""

    class Timed(view_class):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.paint_ms: list[float] = []

        def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
            started = time.perf_counter()
            super().paintEvent(event)
            self.paint_ms.append((time.perf_counter() - started) * 1000.0)

    return Timed


def _report(name: str, timings: dict[str, list[float]]) -> None:
    events, paints = timings["events"], timings["paints"] or [0.0]
    print(
        f"{name:28s} wheel ms: median {statistics.median(events):5.1f} max {max(events):5.1f} | "
        f"paint ms: median {statistics.median(paints):5.1f} p95 {sorted(paints)[int(len(paints) * 0.95)]:5.1f} "
        f"max {max(paints):5.1f} | frames/s {timings['fps'][0]:5.1f}"
    )


def bench_matrix(app, frames: int) -> None:
    from karakal.core.domain import FrameRecord
    from karakal.core.grid_anomaly import GridFrameAnalysisResult
    from karakal.ui.matrix_view import MatrixLayoutConfig, MatrixListWidget

    view = _timed(MatrixListWidget)()
    view.resize(1400, 900)
    view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    view.show()
    records = [FrameRecord(f"F_{index:05d}.png", f"F_{index:05d}.png") for index in range(frames)]
    rng = np.random.default_rng(0)
    payloads = {
        record.key: GridFrameAnalysisResult(
            record.key, "", 2000, 2000, 0, 0, 100, 100, 90, 4, 3, 0, 3,
            float(rng.random()), "low",
        )
        for record in records
    }
    view.set_grid_inspection_visual_mode(True)
    view.set_layout_config(MatrixLayoutConfig(mode="indexed_grid", frames_per_row=250))
    started = time.perf_counter()
    view.set_records(records, sort_mode="name", reset_view=True)
    view.set_grid_inspection_payloads(payloads, enabled=True)
    view.finalize_grid_inspection_payloads()
    _settle(app, 500)
    print(f"matrix {frames} frames ready in {time.perf_counter() - started:.1f} s")
    _report("matrix zoom in (40 notches)", _run_steps(app, view, 40, 120, ctrl=True))
    _report("matrix zoom out (40 notches)", _run_steps(app, view, 40, -120, ctrl=True))
    view.close()


def bench_details(app, size: int) -> None:
    from karakal.ui.details_dialog import _OverlayGraphicsView
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import QGraphicsPixmapItem

    view = _timed(_OverlayGraphicsView)()
    view.resize(1200, 900)
    view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    view.show()
    rng = np.random.default_rng(1)
    for z in range(3):
        gray = (rng.random((size, size)) * 255).astype(np.uint8)
        gray = cv2.GaussianBlur(gray, (0, 0), 3)
        rgba = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGBA)
        rgba[..., 3] = 255 if z == 0 else 110
        image = QImage(rgba.data, size, size, size * 4, QImage.Format.Format_RGBA8888).copy()
        item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setZValue(z)
        view.scene().addItem(item)
    view.fitInView(view.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
    _settle(app, 300)
    _report("frame zoom in (25 notches)", _run_steps(app, view, 25, 120, ctrl=False))
    _report("frame zoom out (25 notches)", _run_steps(app, view, 25, -120, ctrl=False))
    view.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=50_000)
    parser.add_argument("--size", type=int, default=2000)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory():
        bench_matrix(app, args.frames)
        bench_details(app, args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
