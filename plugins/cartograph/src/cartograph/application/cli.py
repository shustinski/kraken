from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cartograph import __version__
from cartograph.application.pipeline import RunLocalVerticalSlice, VerticalSliceRequest
from cartograph.domain.coordinates import GridCoordinate, NominalPlacementMode, PixelSize
from cartograph.application.nominal import PlacementSettings
from cartograph.infrastructure.persistence import solution_to_dict
from cartograph.infrastructure.render import BlendMode

_LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.grid is None:
        from .bootstrap import build_application

        components = build_application([], apply_qss=not args.no_qss)
        components.window.show()
        return int(components.app.exec())

    if args.center_row is None or args.center_col is None:
        parser.error("--center-row and --center-col are required with --grid")

    pixel_size = None
    if args.pixel_size is not None:
        pixel_size = PixelSize(args.pixel_size, args.pixel_size)
    request = VerticalSliceRequest(
        path=Path(args.grid),
        center=GridCoordinate(args.center_row, args.center_col),
        overlap_x=args.overlap_x,
        overlap_y=args.overlap_y,
        placement=PlacementSettings(mode=NominalPlacementMode(args.placement), pixel_size=pixel_size),
        blend=BlendMode(args.blend),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    result = RunLocalVerticalSlice().execute(request)
    if args.output:
        _write_image(Path(args.output), result.mosaic.pixels)
    if args.diagnostics:
        Path(args.diagnostics).write_text(
            json.dumps(solution_to_dict(result.solution), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    _LOGGER.info(
        "registered 3x3 at %s status=%s cache=%s edges=%s",
        request.center,
        result.solution.status.value,
        result.outcome.from_cache,
        len(result.solution.graph.edges),
    )
    return 0


def _write_image(path: Path, pixels: np.ndarray) -> None:
    import cv2

    array = np.clip(pixels, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"failed to write mosaic: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cartograph", description="Local SEM tile stitching for Kraken.")
    parser.add_argument("--grid", help="Tile folder or grid.json manifest. Omit to open the diagnostic UI.")
    parser.add_argument("--center-row", type=int)
    parser.add_argument("--center-col", type=int)
    parser.add_argument("--overlap-x", type=float, default=0.1)
    parser.add_argument("--overlap-y", type=float, default=0.1)
    parser.add_argument(
        "--placement",
        choices=[item.value for item in NominalPlacementMode],
        default=NominalPlacementMode.REGULAR_GRID.value,
    )
    parser.add_argument("--pixel-size", type=float, help="Stage units per pixel for SEM_STAGE/HYBRID.")
    parser.add_argument("--blend", choices=[item.value for item in BlendMode], default=BlendMode.FEATHERED.value)
    parser.add_argument("--output", help="Write the local mosaic as an 8-bit image.")
    parser.add_argument("--diagnostics", help="Write registration diagnostics JSON.")
    parser.add_argument("--cache-dir", help="Directory for persisted local-block JSON.")
    parser.add_argument("--no-qss", action="store_true", help="Do not apply the shared Kraken stylesheet.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser
