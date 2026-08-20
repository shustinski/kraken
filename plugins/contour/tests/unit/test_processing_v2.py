from __future__ import annotations

from contour.application.processing_v2 import (
    MetalRecoverySettings,
    ProcessingRequestV2,
    ProcessingResultV2,
    ViaDetectionSettings,
)
from contour.domain import PolygonData


def test_v2_request_round_trip_keeps_discriminated_settings() -> None:
    request = ProcessingRequestV2(
        input_id="frame.png",
        input_version="sha256:123",
        recognition=MetalRecoverySettings(
            segmentation_strategy="local_adaptive",
            watershed_core_margin=11.0,
        ),
    )

    restored = ProcessingRequestV2.from_dict(request.to_dict())

    assert restored.to_dict() == request.to_dict()
    assert isinstance(restored.recognition, MetalRecoverySettings)
    assert restored.recognition.watershed_core_margin == 11.0


def test_v2_keeps_seeded_conductor_algorithms() -> None:
    request = ProcessingRequestV2(
        input_id="frame.png",
        input_version="sha256:123",
        recognition=MetalRecoverySettings(segmentation_strategy="random_walker"),
    )

    restored = ProcessingRequestV2.from_dict(request.to_dict())

    assert isinstance(restored.recognition, MetalRecoverySettings)
    assert restored.recognition.segmentation_strategy == "random_walker"
    assert restored.to_legacy_settings().metal_segmentation_strategy == "random_walker"


def test_metal_minimum_contrast_defaults_to_fifty_and_clamps_zero() -> None:
    assert MetalRecoverySettings().min_contrast == 50.0
    assert MetalRecoverySettings(min_contrast=0.0).min_contrast == 1.0


def test_legacy_migrator_reports_unknown_fields() -> None:
    request, report = ProcessingRequestV2.migrate_legacy(
        {
            "recognition_mode": "via",
            "min_via_width": 7,
            "bright_via_diameter_min": 6,
            "bright_via_diameter_max": 12,
            "via_output_diameter": 9,
            "removed_knob": 42,
        }
    )

    assert isinstance(request.recognition, ViaDetectionSettings)
    assert request.recognition.min_width == 7
    assert request.recognition.candidate_diameter_min == 6
    assert request.recognition.candidate_diameter_max == 12
    assert request.recognition.output_diameter == 9
    assert request.to_legacy_settings().via_output_diameter == 9
    assert report.dropped_fields == {"removed_knob": 42}
    assert report.warnings


def test_v2_result_round_trip_preserves_hierarchy_and_provenance() -> None:
    result = ProcessingResultV2(
        polygons=[
            PolygonData(id=1, points=[(0, 0), (10, 0), (10, 10), (0, 10)]),
            PolygonData(id=2, points=[(2, 2), (4, 2), (4, 4), (2, 4)], is_hole=True, parent_id=1),
        ],
        hierarchy={1: [2]},
        warnings=["suspicious repair"],
        stage_timings_ms={"topology": 1.5},
        provenance={"algorithm": "auto"},
    )

    restored = ProcessingResultV2.from_dict(result.to_dict())

    assert restored.hierarchy == {1: [2]}
    assert restored.polygons[1].parent_id == 1
    assert restored.provenance == {"algorithm": "auto"}
