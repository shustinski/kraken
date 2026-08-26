"""Domain errors for Cartograph. Callers must not swallow these."""

from __future__ import annotations


class CartographError(Exception):
    """Base error for Cartograph domain and application failures."""


class GridLoadError(CartographError):
    """Raised when a tile grid cannot be loaded from disk or a manifest."""


class PlacementError(CartographError):
    """Raised when nominal placement cannot be computed from the given coordinates."""


class RegistrationError(CartographError):
    """Raised when local registration cannot run (invalid window, missing images)."""


class PersistenceError(CartographError):
    """Raised when a local-block sidecar cannot be read or written."""
