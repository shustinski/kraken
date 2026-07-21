"""Processing use-cases.

This used to be a single 1600-line ``processing.py`` module. During the
production-ready refactor it became a sub-package so concrete logic can be
migrated gradually into concern-specific submodules; this ``__init__`` keeps
the historical import surface intact for existing callers.
"""

from __future__ import annotations

from ._core import *  # noqa: F403
from ._core import (  # noqa: F401
    PreparedImageRequest,
    PreviewProcessingRequest,
    _resolve_conductor_edge_method,
    _resolve_via_edge_method,
    _via_grayscale,
    apply_via_vectorization_mask,
    build_conductor_vectorization_mask,
    build_detection_debug_maps,
    build_prepared_image_signature,
    build_preview_request_signature,
    build_via_vectorization_mask,
    prepare_image_for_preview,
    process_image_path,
    process_image_path_timed,
)
