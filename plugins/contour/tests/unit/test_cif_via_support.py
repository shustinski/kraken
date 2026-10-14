from __future__ import annotations
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
from contour.contour_extractor import extract_polygons
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    clear_cif_parse_cache,
    export_dataset_frame,
    load_polygons_cif,
    load_polygons_cv,
    load_polygons_vector,
    save_polygons_cif,
    save_polygons_cv,
    save_result_bundle,
)

from test_cif_via_support_parts.artifact_path import ArtifactPathMixin
from test_cif_via_support_parts.via_profile_applies import ViaProfileAppliesMixin
from test_cif_via_support_parts.contour_settings_parse import ContourSettingsParseMixin
from test_cif_via_support_parts.cif_loader_reads import CifLoaderReadsMixin
from test_cif_via_support_parts.dataset_export_writes import DatasetExportWritesMixin
from test_cif_via_support_parts.cif_round_trip import CifRoundTripMixin
from test_cif_via_support_parts.cif_loader_matches import CifLoaderMatchesMixin
from test_cif_via_support_parts.cif_loader_keeps import CifLoaderKeepsMixin
from test_cif_via_support_parts.cif_loader_opt import CifLoaderOptMixin
from test_cif_via_support_parts.cif_loader_splits import CifLoaderSplitsMixin
from test_cif_via_support_parts.cv_saves_polygon import CvSavesPolygonMixin
from test_cif_via_support_parts.result_bundle_can import ResultBundleCanMixin
from test_cif_via_support_parts.lib_open_cif import (
    _longest_axis_aligned_edge,
    _EnvVar,
    _rectangle_polygon,
    TestKLayoutCifLoader,
    TestLibOpenCifLoader,
)
from test_cif_via_support_parts.boundary_vertex_snap_tests import BoundaryVertexSnapTests

class CifViaSupportTests(
    ArtifactPathMixin,
    ViaProfileAppliesMixin,
    ContourSettingsParseMixin,
    CifLoaderReadsMixin,
    DatasetExportWritesMixin,
    CifRoundTripMixin,
    CifLoaderMatchesMixin,
    CifLoaderKeepsMixin,
    CifLoaderOptMixin,
    CifLoaderSplitsMixin,
    CvSavesPolygonMixin,
    ResultBundleCanMixin,
    unittest.TestCase,
):
    pass

if __name__ == "__main__":
    unittest.main()

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _longest_axis_aligned_edge,
    _EnvVar,
    _rectangle_polygon,
    TestKLayoutCifLoader,
    TestLibOpenCifLoader,
    BoundaryVertexSnapTests,
)
