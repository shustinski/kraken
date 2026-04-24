"""Vision / analysis backend (SEM topology): modular pipelines and stable data schemas.

This package is the integration target for the next-generation backend. Older code
(``contour_extractor``, ``application.use_cases.processing._core``) remains in
place; new entrypoints live here and can be called from a thin application layer
or from legacy adapters (see :mod:`contour.vision.integration`).
"""

from __future__ import annotations

from . import contour_extraction, integration, io_normalize, preprocessing, schemas
from .via import CompositeViaDetector, ViaRunConfig

__all__ = [
    "CompositeViaDetector",
    "ViaRunConfig",
    "contour_extraction",
    "integration",
    "io_normalize",
    "preprocessing",
    "schemas",
]
