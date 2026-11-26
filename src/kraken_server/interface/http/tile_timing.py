"""Opt-in timing of matrix tile requests on the server.

Enabled by ``matrix_tile_timing`` in ``kraken.ini``; lines go to
``profiling-server.log`` so they do not mix with the desktop's file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from kraken_core.frame_matrix import profiling

TIMING = "matrix_tile_timing"
SERVER_LOG = "profiling-server.log"
_LOGGER = logging.getLogger(__name__)


def configure_server_profiling() -> None:
    """Read ``kraken.ini`` when one exists; never create it on a server install."""

    if profiling.configured():
        return
    explicit = os.environ.get("KRAKEN_CONFIG", "").strip()
    ini_path = Path(explicit) if explicit else profiling.project_directory() / "kraken.ini"
    if not ini_path.is_file():
        return
    try:
        profiling.configure(ini_path.parent, log_file=SERVER_LOG)
    except Exception:
        _LOGGER.warning("Matrix tile timing is off: %s could not be read", ini_path, exc_info=True)


def rect_label(params: Any) -> str:
    return ":".join(str(params.get(name, "")) for name in ("x1", "y1", "x2", "y2"))


def register_tile_timing(app: Any) -> None:
    """Time each ``viewport-mosaic`` request from arrival to response, auth included."""

    if not profiling.enabled(TIMING):
        return

    @app.middleware("http")
    async def time_tile_requests(request, call_next):
        if not request.url.path.endswith("/viewport-mosaic"):
            return await call_next(request)
        started = perf_counter()
        response = await call_next(request)
        profiling.timing(
            TIMING,
            "http_request",
            side="server",
            rect=rect_label(request.query_params),
            status=response.status_code,
            ms=(perf_counter() - started) * 1000,
        )
        return response


__all__ = ["TIMING", "configure_server_profiling", "register_tile_timing", "rect_label"]
