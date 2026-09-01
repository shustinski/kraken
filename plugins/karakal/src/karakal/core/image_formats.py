"""Canonical image extensions supported by Karakal."""

from __future__ import annotations


SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
SUPPORTED_IMAGE_EXTENSION_SET = frozenset(SUPPORTED_IMAGE_EXTENSIONS)


__all__ = ["SUPPORTED_IMAGE_EXTENSIONS", "SUPPORTED_IMAGE_EXTENSION_SET"]
