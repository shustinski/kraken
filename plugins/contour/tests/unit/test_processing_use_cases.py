from __future__ import annotations
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import _core as processing_core
from contour.application.use_cases.processing import build_detection_debug_maps, process_image_path

from test_processing_use_cases_parts.structural_metal_recognition import StructuralMetalRecognitionMixin
from test_processing_use_cases_parts.sem_conductor_recognition import SemConductorRecognitionMixin
from test_processing_use_cases_parts.via_profile_detects import ViaProfileDetectsMixin
from test_processing_use_cases_parts.lowering_white_range import LoweringWhiteRangeMixin
from test_processing_use_cases_parts.via_profile_gradient import ViaProfileGradientMixin
from test_processing_use_cases_parts.via_profile_detects_2 import ViaProfileDetects2Mixin
from test_processing_use_cases_parts.bright_tophat_dog import BrightTophatDogMixin
from test_processing_use_cases_parts.changing_bright_final import ChangingBrightFinalMixin
from test_processing_use_cases_parts.via_profile_spot import ViaProfileSpotMixin
from test_processing_use_cases_parts.build_detection_debug import BuildDetectionDebugMixin
from test_processing_use_cases_parts.process_image_path import ProcessImagePathMixin
from test_processing_use_cases_parts.via_detector_auto import ViaDetectorAutoMixin
from test_processing_use_cases_parts.heuristic_mode_does import HeuristicModeDoesMixin
from test_processing_use_cases_parts.sem_backend_keeps import SemBackendKeepsMixin

class ProcessingUseCasesTests(
    StructuralMetalRecognitionMixin,
    SemConductorRecognitionMixin,
    ViaProfileDetectsMixin,
    LoweringWhiteRangeMixin,
    ViaProfileGradientMixin,
    ViaProfileDetects2Mixin,
    BrightTophatDogMixin,
    ChangingBrightFinalMixin,
    ViaProfileSpotMixin,
    BuildDetectionDebugMixin,
    ProcessImagePathMixin,
    ViaDetectorAutoMixin,
    HeuristicModeDoesMixin,
    SemBackendKeepsMixin,
    unittest.TestCase,
):
    pass

if __name__ == "__main__":
    unittest.main()
