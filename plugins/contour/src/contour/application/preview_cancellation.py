"""Cooperative cancellation for interactive preview (UI supersedes in-flight work)."""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Generator

_preview_cancel: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "contour_preview_cancel", default=None
)


class PreviewProcessingCancelled(Exception):
    """Raised when a newer preview was queued and this run should stop."""


@contextmanager
def use_preview_cancellation_event(event: threading.Event | None) -> Generator[None, None, None]:
    if event is None:
        yield
        return
    token = _preview_cancel.set(event)
    try:
        yield
    finally:
        _preview_cancel.reset(token)


def preview_cancelled() -> bool:
    ev = _preview_cancel.get()
    return ev is not None and ev.is_set()


def raise_if_preview_cancelled() -> None:
    if preview_cancelled():
        raise PreviewProcessingCancelled
