"""Contract notes for future append-only archive thumbnail stores.

An archive adapter is expected to use immutable segments, an independently
replaceable key-to-offset index, tombstones, recovery of an incomplete final
segment and background compaction. ZIP/TAR behaviour is intentionally absent
from the public storage port.
"""

from __future__ import annotations

from typing import Protocol

from .storage import ThumbnailStore


class ArchiveThumbnailStore(ThumbnailStore, Protocol):
    """Marker protocol for future segmented archive adapters."""


__all__ = ["ArchiveThumbnailStore"]
