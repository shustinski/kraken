from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QGroupBox

from contour.application.processing import ContourExtractionSettings
from contour.vision.metal_recovery.bsr_owt_ucm import build_bsr_ucm, cut_bsr_hierarchy
from contour.vision.metal_recovery.detector import (
    MetalRecoveryConfig,
    _extract_polygons_per_instance,
)
from contour.vision.metal_recovery.features import (
    build_metal_structural_features,
    clear_metal_feature_cache,
)
from contour.vision.metal_recovery.graph_multi_separator import (
    _assign_peeled_separator_pixels,
    _paired_rim_core_evidence,
    _project_material_separators,
    _recover_paired_rim_ribbons,
    _thin_separator_to_one_pixel,
)
from contour.vision.metal_recovery.material_classifier import classify_partition_material
from contour.vision.metal_recovery.graph_strategies import _lifted_relations
from contour.vision.metal_recovery.owt_ucm import (
    build_ucm_hierarchy,
    oriented_watershed_partition,
)
from contour.vision.metal_recovery.pipeline_stages import (
    build_metal_segmentation_mask_staged,
    clear_metal_segmentation_cache,
)
from contour.vision.metal_recovery.segmentation import (
    MetalSegmentationConfig,
    normalize_metal_segmentation_strategy,
)
from contour.vision.metal_recovery.signed_graph import SignedAffinityBuilder
from contour.vision.metal_recovery.strategy_contracts import StrategyConfigurationError
from contour.vision.metal_recovery.strategy_registry import (
    IMPLEMENTED_NEW_STRATEGIES,
    MetalStrategyConfigs,
    normalize_strategy_parameters,
    strategy_spec,
    visible_strategy_specs,
)

NEW_STRATEGIES = tuple(sorted(IMPLEMENTED_NEW_STRATEGIES))


def _scene(kind: str) -> np.ndarray:
    image = np.full((72, 96), 28, dtype=np.uint8)
    if kind == "two_regions":
        cv2.rectangle(image, (8, 8), (38, 64), 175, -1)
        cv2.rectangle(image, (56, 8), (88, 64), 155, -1)
    elif kind == "parallel":
        cv2.rectangle(image, (16, 5), (39, 67), 175, -1)
        cv2.rectangle(image, (43, 5), (66, 67), 175, -1)
    elif kind == "short_gap":
        cv2.rectangle(image, (10, 10), (86, 62), 165, -1)
        cv2.line(image, (48, 10), (48, 31), 28, 2)
        cv2.line(image, (48, 36), (48, 62), 28, 2)
    elif kind == "wide_dark":
        cv2.rectangle(image, (8, 8), (88, 64), 170, -1)
        cv2.rectangle(image, (11, 11), (85, 61), 72, -1)
    elif kind == "hole":
        cv2.rectangle(image, (8, 6), (88, 66), 170, -1)
        cv2.rectangle(image, (35, 25), (61, 47), 28, -1)
    elif kind == "border_touching":
        cv2.rectangle(image, (0, 9), (43, 63), 170, -1)
    elif kind != "empty":
        raise ValueError(kind)
    return image


@pytest.mark.parametrize("strategy", NEW_STRATEGIES)
def test_new_strategy_registry_has_backend_and_validated_defaults(strategy: str) -> None:
    spec = strategy_spec(strategy)
    assert spec.load_backend() is not None
    defaults = normalize_strategy_parameters(strategy, None)
    assert defaults
    assert set(defaults) == {parameter.key for parameter in spec.parameters}
    assert all(parameter.tooltip for parameter in spec.parameters)


def test_ic_sem_is_an_explicit_non_advertised_extension_point() -> None:
    extension = strategy_spec("ic_sem_expert")
    assert extension.backend_path is None
    assert extension.load_backend() is None
    assert extension.strategy_id not in {spec.strategy_id for spec in visible_strategy_specs()}


@pytest.mark.parametrize("strategy", NEW_STRATEGIES)
def test_new_strategy_is_deterministic_and_returns_canonical_maps(strategy: str) -> None:
    image = _scene("two_regions")
    config = MetalSegmentationConfig(
        segmentation_strategy=strategy,
        min_component_area=2,
        strategy_parameters=MetalStrategyConfigs.from_mapping(None).to_dict(),
    )
    clear_metal_segmentation_cache()
    first = build_metal_segmentation_mask_staged(image, config)
    clear_metal_segmentation_cache()
    second = build_metal_segmentation_mask_staged(image, config)
    np.testing.assert_array_equal(first.mask, second.mask)
    np.testing.assert_array_equal(first.instance_labels, second.instance_labels)
    assert first.mask.shape == image.shape
    assert first.instance_labels is not None
    assert first.instance_labels.dtype == np.int32
    assert first.boundary_map is not None
    assert first.confidence_map is not None
    assert first.strategy == strategy
    assert set(first.timings_ms) >= {
        "feature_build",
        "graph_construction",
        "solver",
        "material_classification",
        "total",
    }


@pytest.mark.parametrize("strategy", NEW_STRATEGIES)
@pytest.mark.parametrize(
    "kind",
    ("two_regions", "parallel", "short_gap", "wide_dark", "hole", "border_touching"),
)
def test_new_strategy_handles_required_synthetic_topologies(strategy: str, kind: str) -> None:
    result = build_metal_segmentation_mask_staged(
        _scene(kind),
        MetalSegmentationConfig(
            segmentation_strategy=strategy,
            min_component_area=2,
            strategy_parameters=MetalStrategyConfigs.from_mapping(None).to_dict(),
        ),
    )
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 255})
    assert result.instance_labels is not None
    assert result.instance_labels.shape == result.mask.shape
    assert np.any(result.mask), f"{strategy} rejected the complete {kind} scene"


@pytest.mark.parametrize("strategy", NEW_STRATEGIES)
def test_new_strategy_rejects_empty_frame(strategy: str) -> None:
    result = build_metal_segmentation_mask_staged(
        _scene("empty"),
        MetalSegmentationConfig(
            segmentation_strategy=strategy,
            min_component_area=2,
            strategy_parameters=MetalStrategyConfigs.from_mapping(None).to_dict(),
        ),
    )
    assert not np.any(result.mask)
    assert result.instance_labels is not None
    assert not np.any(result.instance_labels)


def test_nested_strategy_settings_round_trip_and_old_config_defaults() -> None:
    original = ContourExtractionSettings.from_dict(
        {
            "metal_segmentation_strategy": "owt_ucm",
            "metal_strategy_parameters": {
                "owt_ucm": {"hierarchy_level": 0.73, "orientation_bins": 12},
                "lifted_multicut": {"minimum_lifted_distance": 30, "maximum_lifted_distance": 4},
            },
        }
    )
    restored = ContourExtractionSettings.from_dict(original.to_dict())
    assert restored.metal_strategy_parameters["owt_ucm"]["hierarchy_level"] == pytest.approx(0.73)
    assert restored.metal_strategy_parameters["owt_ucm"]["orientation_bins"] == 8
    assert restored.metal_strategy_parameters["lifted_multicut"]["maximum_lifted_distance"] == 30

    legacy = ContourExtractionSettings.from_dict({"metal_segmentation_strategy": "legacy_otsu"})
    assert legacy.metal_segmentation_strategy == "legacy_otsu"
    assert set(legacy.metal_strategy_parameters) == set(IMPLEMENTED_NEW_STRATEGIES)


def test_recovery_config_snapshot_can_reconstruct_nested_strategy_settings() -> None:
    original = MetalRecoveryConfig(
        segmentation_strategy="owt_ucm",
        strategy_configs=MetalStrategyConfigs.from_mapping({"owt_ucm": {"hierarchy_level": 0.73}}),
    )
    restored = MetalRecoveryConfig(**original.to_snapshot())
    assert isinstance(restored.strategy_configs, MetalStrategyConfigs)
    assert restored.strategy_configs.for_strategy("owt_ucm")["hierarchy_level"] == pytest.approx(0.73)


def test_touching_instance_labels_are_contoured_separately() -> None:
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[5:35, 5:30] = 1
    labels[5:35, 30:55] = 2
    polygons = _extract_polygons_per_instance(
        labels,
        MetalRecoveryConfig(
            min_width_px=0.0,
            min_length_px=0.0,
            min_area=1.0,
            min_component_area=1.0,
            min_perimeter=0.0,
            min_points=3,
        ),
    )
    outer = [polygon for polygon in polygons if not polygon.is_hole]
    assert len(outer) == 2
    assert {polygon.id for polygon in outer} == {1, 2}


def test_strategy_standard_payload_factory_covers_new_strategies() -> None:
    from contour.ui.metal_presets import (
        built_in_metal_presets,
        metal_preset_table,
        strategy_standard_metal_preset_payload,
    )

    assert list(metal_preset_table()) == ["standard"]
    assert list(built_in_metal_presets("ru")) == ["Стандартный"]
    assert list(built_in_metal_presets("en")) == ["Standard"]
    for strategy in NEW_STRATEGIES:
        payload = strategy_standard_metal_preset_payload(strategy)
        assert payload["metal_segmentation_strategy"] == strategy
        assert strategy in payload["metal_strategy_parameters"]


def test_owt_hierarchy_level_is_monotonic() -> None:
    features = build_metal_structural_features(_scene("two_regions"))
    defaults = normalize_strategy_parameters("owt_ucm", None)
    partition = oriented_watershed_partition(features, defaults)
    assert partition.oriented_channels_preview.shape == (*features.gray.shape, 3)
    fine, fine_ucm, _fine_merges = build_ucm_hierarchy(
        partition,
        {**defaults, "hierarchy_level": 0.05},
    )
    coarse, coarse_ucm, _coarse_merges = build_ucm_hierarchy(
        partition,
        {**defaults, "hierarchy_level": 0.8},
    )
    assert int(coarse.max()) <= int(fine.max())
    assert np.any(fine_ucm > 0)
    np.testing.assert_array_equal(fine_ucm, coarse_ucm)


def test_bsr_ucm_uses_dynamic_mean_boundary_hierarchy() -> None:
    labels = np.repeat(np.array([[1, 1, 2, 2, 3, 3]], dtype=np.int32), 8, axis=0)
    channels = np.zeros((8, *labels.shape), dtype=np.float32)
    channels[:, :, 1:3] = 0.1
    channels[:, :, 3:5] = 0.9
    result = build_bsr_ucm(labels, channels)

    assert len(result.hierarchy) == 2
    assert float(result.hierarchy[0]["saliency"]) < float(result.hierarchy[1]["saliency"])
    fine = cut_bsr_hierarchy(labels, result.hierarchy, float(result.hierarchy[0]["saliency"]) - 1e-4)
    middle = cut_bsr_hierarchy(labels, result.hierarchy, float(result.hierarchy[0]["saliency"]) + 1e-4)
    coarse = cut_bsr_hierarchy(labels, result.hierarchy, 1.0)
    assert int(fine.max()) == 3
    assert int(middle.max()) == 2
    assert int(coarse.max()) == 1


@pytest.mark.parametrize("solver", ("greedy_separator_shrinking", "greedy_separator_growing"))
def test_graph_multi_separator_runs_upstream_native_solver(solver: str) -> None:
    result = build_metal_segmentation_mask_staged(
        _scene("two_regions"),
        MetalSegmentationConfig(
            segmentation_strategy="graph_multi_separator",
            min_component_area=2,
            strategy_parameters={"graph_multi_separator": {"solver": solver}},
        ),
    )
    assert result.debug_data["backend"] == "JannikIrmai/multi-separator"
    assert result.debug_data["upstream_commit"] == "437c651ddf1452452cca4cbc3c0eed2065308486"
    assert result.debug_data["solver"] == solver
    assert int(result.debug_data["iterations"]) > 0


def test_graph_multi_separator_tiles_without_resizing_pixels() -> None:
    image = _scene("parallel")
    result = build_metal_segmentation_mask_staged(
        image,
        MetalSegmentationConfig(
            segmentation_strategy="graph_multi_separator",
            min_component_area=2,
            strategy_parameters={
                "graph_multi_separator": {
                    "solver_tile_size": 32,
                    "solver_tile_overlap": 8,
                }
            },
        ),
    )
    assert int(result.debug_data["tile_count"]) == 9
    assert result.mask.shape == image.shape
    assert result.boundary_map is not None
    assert result.boundary_map.shape == image.shape


def test_graph_multi_separator_parallel_tiles_preserve_result() -> None:
    image = _scene("parallel")

    def segment(worker_count: int):
        return build_metal_segmentation_mask_staged(
            image,
            MetalSegmentationConfig(
                segmentation_strategy="graph_multi_separator",
                min_component_area=2,
                strategy_parameters={
                    "graph_multi_separator": {
                        "solver_tile_size": 32,
                        "solver_tile_overlap": 8,
                        "solver_workers": worker_count,
                    }
                },
            ),
        )

    sequential = segment(1)
    parallel = segment(4)

    assert np.array_equal(parallel.mask, sequential.mask)
    assert np.array_equal(parallel.instance_labels, sequential.instance_labels)
    assert int(parallel.debug_data["solver_workers"]) == 4
    assert float(parallel.timings_ms["native_tile_phase"]) >= 0.0


def test_paired_rim_core_distinguishes_bright_trace_from_dark_gap() -> None:
    image = np.full((96, 160), 48, dtype=np.uint8)
    cv2.rectangle(image, (18, 8), (46, 87), 176, -1)
    cv2.rectangle(image, (96, 8), (142, 87), 176, -1)
    cv2.rectangle(image, (105, 8), (133, 87), 48, -1)
    evidence = _paired_rim_core_evidence(build_metal_structural_features(image))

    assert float(np.mean(evidence[10:38, 14:19])) > 0.15
    assert float(np.mean(evidence[10:38, 56:63])) < 0.05
    assert float(np.mean(evidence[10:38, 30:42])) < 0.05


def test_graph_multi_separator_uses_guarded_fallback_before_native_solver() -> None:
    result = build_metal_segmentation_mask_staged(
        _scene("parallel"),
        MetalSegmentationConfig(
            segmentation_strategy="graph_multi_separator",
            min_component_area=2,
            strategy_parameters={
                "graph_multi_separator": {
                    "paired_rim_fallback_min_core_fraction": 0.0,
                    "paired_rim_fallback_fraction": 0.0,
                }
            },
        ),
    )

    assert result.debug_data["backend"] == "opencv-otsu"
    assert result.debug_data["fallback"] == "missing_core_paired_rims"
    assert float(result.timings_ms["solver"]) == 0.0


def test_msp_paired_rim_recovery_accepts_only_narrow_locally_bright_ribbon() -> None:
    image = np.full((72, 96), 30, dtype=np.uint8)
    image[10:62, 42:47] = 90
    image[16:56, 64:84] = 90
    evidence = np.zeros(image.shape, dtype=np.float32)
    evidence[10:62, 42:47] = 0.2
    evidence[16:56, 64:84] = 0.2

    recovered_mask, recovered, count, evidence_limit, contrast_limit = _recover_paired_rim_ribbons(
        image,
        np.zeros(image.shape, dtype=np.uint8),
        evidence,
        normalize_strategy_parameters("graph_multi_separator", None),
    )

    assert count == 1
    assert evidence_limit == pytest.approx(0.05)
    assert contrast_limit >= 24.0
    assert np.all(recovered[10:62, 42:47])
    assert not np.any(recovered[16:56, 64:84])
    assert np.array_equal(recovered_mask > 0, recovered)


def test_thin_separator_reduces_thick_band_to_one_pixel() -> None:
    band = np.zeros((17, 23), dtype=bool)
    band[:, 9:14] = True

    thin = _thin_separator_to_one_pixel(band)

    assert np.any(thin)
    assert int(np.max(thin.sum(axis=1))) == 1
    assert int(np.max(thin.sum(axis=0))) >= 1


def test_peeled_separator_pixels_go_to_both_sides_of_one_pixel_wall() -> None:
    native = np.zeros((11, 15), dtype=bool)
    native[:, 6:10] = True
    thin = _thin_separator_to_one_pixel(native)
    labels = cv2.connectedComponents((~native).astype(np.uint8), connectivity=4)[1].astype(np.int32)
    labels[native] = 0

    assigned = _assign_peeled_separator_pixels(labels, native, thin)

    assert int(np.max(thin.sum(axis=1))) == 1
    assert int(np.max(assigned)) == 2
    left = assigned[5, 2]
    right = assigned[5, 12]
    assert left != 0 and right != 0 and left != right
    assert assigned[5, 6] in {0, left}
    assert assigned[5, 9] in {0, right}


def test_graph_multi_separator_classifies_both_sides_of_thin_separator() -> None:
    image = np.full((16, 24), 40, dtype=np.uint8)
    image[:, 13:] = 200
    separator = np.zeros(image.shape, dtype=bool)
    separator[:, 12] = True
    labels = cv2.connectedComponents((~separator).astype(np.uint8), connectivity=4)[1].astype(np.int32)
    labels[separator] = 0
    core = np.zeros(image.shape, dtype=np.float32)
    core[:, 13:] = 0.9
    substrate = np.zeros(image.shape, dtype=np.float32)
    substrate[:, :12] = 0.9
    features = replace(
        build_metal_structural_features(image),
        core_evidence=core,
        substrate_evidence=substrate,
        local_contrast=np.zeros(image.shape, dtype=np.float32),
    )

    material = classify_partition_material(
        labels,
        features,
        normalize_strategy_parameters("graph_multi_separator", None),
    )

    assert int(np.max(labels)) == 2
    assert material.mask[8, 4] == 0
    assert material.mask[8, 20] == 255
    assert material.mask[8, 12] == 0


def test_graph_multi_separator_projects_bright_internal_separator_and_merges_regions() -> None:
    features = build_metal_structural_features(np.full((9, 13), 160, dtype=np.uint8))
    separator = np.zeros((9, 13), dtype=bool)
    separator[:, 6] = True
    labels = np.ones((9, 13), dtype=np.int32)
    labels[:, 6:] = 2
    labels[separator] = 0
    core = np.zeros(labels.shape, dtype=np.float32)
    core[separator] = 0.8
    substrate = np.zeros(labels.shape, dtype=np.float32)
    material_features = replace(
        features,
        core_evidence=core,
        substrate_evidence=substrate,
    )

    repaired, projected, merged_pairs = _project_material_separators(
        labels,
        separator,
        np.full(labels.shape, 0.5, dtype=np.float32),
        material_features,
        np.asarray([0.0, 0.9, 0.9], dtype=np.float32),
        normalize_strategy_parameters("graph_multi_separator", None),
    )

    assert np.all(projected[separator])
    assert np.all(repaired > 0)
    assert int(repaired.max()) == 1
    assert merged_pairs == 1


def test_graph_multi_separator_preserves_substrate_separator() -> None:
    features = build_metal_structural_features(np.full((9, 13), 160, dtype=np.uint8))
    separator = np.zeros((9, 13), dtype=bool)
    separator[:, 6] = True
    labels = np.ones((9, 13), dtype=np.int32)
    labels[:, 6:] = 2
    labels[separator] = 0
    core = np.zeros(labels.shape, dtype=np.float32)
    substrate = np.zeros(labels.shape, dtype=np.float32)
    substrate[separator] = 0.8
    material_features = replace(
        features,
        core_evidence=core,
        substrate_evidence=substrate,
    )

    repaired, projected, merged_pairs = _project_material_separators(
        labels,
        separator,
        np.full(labels.shape, 0.5, dtype=np.float32),
        material_features,
        np.asarray([0.0, 0.9, 0.9], dtype=np.float32),
        normalize_strategy_parameters("graph_multi_separator", None),
    )

    np.testing.assert_array_equal(repaired, labels)
    assert not np.any(projected)
    assert merged_pairs == 0


def test_graph_multi_separator_projection_can_be_disabled_for_raw_partition_diagnostics() -> None:
    features = build_metal_structural_features(np.full((5, 7), 160, dtype=np.uint8))
    separator = np.zeros((5, 7), dtype=bool)
    separator[:, 3] = True
    labels = np.ones((5, 7), dtype=np.int32)
    labels[:, 3:] = 2
    labels[separator] = 0

    repaired, projected, merged_pairs = _project_material_separators(
        labels,
        separator,
        np.zeros(labels.shape, dtype=np.float32),
        replace(features, core_evidence=np.ones(labels.shape, dtype=np.float32)),
        np.asarray([0.0, 0.9, 0.9], dtype=np.float32),
        {"separator_projection_enabled": False},
    )

    assert repaired is labels
    assert not np.any(projected)
    assert merged_pairs == 0


def test_signed_graph_is_reused_unchanged_between_solvers() -> None:
    features = build_metal_structural_features(_scene("parallel"))
    parameters = normalize_strategy_parameters("gasp", None)
    builder = SignedAffinityBuilder()
    first = builder.build(features, parameters)
    second = builder.build(features, normalize_strategy_parameters("multicut", None))
    assert second.build_time_ms == 0.0
    assert second.pixel_labels is first.pixel_labels
    assert second.edge_u is first.edge_u
    assert second.attraction is first.attraction
    assert second.repulsion is first.repulsion


def test_stage_timings_measure_feature_cache_lookup_at_pipeline_boundary() -> None:
    image = _scene("two_regions")
    clear_metal_feature_cache()
    first = build_metal_segmentation_mask_staged(
        image,
        MetalSegmentationConfig(segmentation_strategy="owt_ucm", min_component_area=2),
    )
    second = build_metal_segmentation_mask_staged(
        image,
        MetalSegmentationConfig(
            segmentation_strategy="graph_multi_separator",
            min_component_area=2,
        ),
    )
    assert first.timings_ms["feature_build"] > second.timings_ms["feature_build"]
    assert first.timings_ms["total"] >= first.timings_ms["feature_build"]


def test_pixel_graph_size_limit_fails_with_controlled_configuration_error() -> None:
    oversized = SimpleNamespace(gray=np.empty((1001, 1000), dtype=np.uint8))
    with pytest.raises(StrategyConfigurationError, match="1,001,000 nodes"):
        SignedAffinityBuilder().build(oversized, {"graph_domain": "pixels"})


def test_signed_graph_marks_diagonal_only_edges_for_mws_offsets() -> None:
    features = build_metal_structural_features(_scene("parallel"))
    parameters = normalize_strategy_parameters(
        "mutex_watershed",
        {
            "connectivity": "8",
            "atomic_segmentation_method": "regular_grid",
            "atomic_region_scale": 12,
        },
    )
    graph = SignedAffinityBuilder().build(features, parameters)
    assert np.any(graph.edge_diagonal_only)
    assert np.any(~graph.edge_diagonal_only)


def test_lifted_relation_toggles_disable_their_respective_costs() -> None:
    features = build_metal_structural_features(_scene("two_regions"))
    base = normalize_strategy_parameters(
        "lifted_multicut",
        {
            "atomic_segmentation_method": "regular_grid",
            "atomic_region_scale": 12,
            "lifted_confidence_threshold": 0.0,
            "maximum_lifted_distance": 12,
        },
    )
    graph = SignedAffinityBuilder().build(features, base)
    _u, _v, attraction, _repulsion, _debug = _lifted_relations(
        graph,
        features,
        {**base, "same_trace_lifted_attraction": False},
    )
    assert attraction.size
    assert not np.any(attraction)

    _u, _v, _attraction, repulsion, _debug = _lifted_relations(
        graph,
        features,
        {**base, "cross_boundary_lifted_repulsion": False},
    )
    assert repulsion.size
    assert not np.any(repulsion)


@pytest.mark.parametrize(
    ("strategy", "overrides", "debug_key", "expected"),
    (
        (
            "mutex_watershed",
            {"edge_ordering": "signed_margin"},
            "edge_ordering",
            "signed_margin",
        ),
        (
            "multicut",
            {"initialization": "positive_components"},
            "initialization",
            "positive_components",
        ),
    ),
)
def test_solver_variants_are_executed_and_reported(
    strategy: str,
    overrides: dict[str, object],
    debug_key: str,
    expected: str,
) -> None:
    result = build_metal_segmentation_mask_staged(
        _scene("parallel"),
        MetalSegmentationConfig(
            segmentation_strategy=strategy,
            min_component_area=2,
            strategy_parameters={strategy: overrides},
        ),
    )
    assert result.debug_data[debug_key] == expected


@pytest.mark.parametrize("strategy", NEW_STRATEGIES)
def test_new_strategy_normalization_is_explicit(strategy: str) -> None:
    assert normalize_metal_segmentation_strategy(strategy) == strategy


@pytest.mark.gui
def test_new_strategy_ui_switches_panels_and_round_trips_values(qtbot) -> None:
    from contour.widget import PolygonExtractionWidget

    widget = PolygonExtractionWidget()
    qtbot.addWidget(widget)
    selector_values = {
        str(widget.metal_segmentation_strategy_combo.itemData(index))
        for index in range(widget.metal_segmentation_strategy_combo.count())
    }
    assert set(NEW_STRATEGIES) <= selector_values

    owt_index = widget.metal_segmentation_strategy_combo.findData("owt_ucm")
    widget.metal_segmentation_strategy_combo.setCurrentIndex(owt_index)
    assert widget.metal_advanced_group.isChecked()
    assert not widget.metal_strategy_parameters_group.isHidden()
    assert widget.metal_adaptive_group.isHidden()
    assert widget.metal_strategy_parameter_stack.currentIndex() == widget.metal_strategy_parameter_pages["owt_ucm"]

    hierarchy = widget.metal_strategy_parameter_widgets["owt_ucm"]["hierarchy_level"]
    orientation_bins = widget.metal_strategy_parameter_widgets["owt_ucm"]["orientation_bins"]
    hierarchy.setValue(0.73)
    orientation_bins.setValue(100)
    assert orientation_bins.value() == 8
    hierarchy.setValue(-1.0)
    assert hierarchy.value() == 0.0
    hierarchy.setValue(0.73)
    settings = widget._current_contour_settings()
    assert settings.metal_segmentation_strategy == "owt_ucm"
    assert settings.metal_strategy_parameters["owt_ucm"]["hierarchy_level"] == pytest.approx(0.73)

    restored = ContourExtractionSettings.from_dict(settings.to_dict())
    hierarchy.setValue(0.1)
    widget._set_extraction_settings(restored)
    assert hierarchy.value() == pytest.approx(0.73)

    gasp_index = widget.metal_segmentation_strategy_combo.findData("gasp")
    widget.metal_segmentation_strategy_combo.setCurrentIndex(gasp_index)
    assert widget.metal_strategy_parameter_stack.currentIndex() == widget.metal_strategy_parameter_pages["gasp"]
    assert "linkage_criterion" in widget.metal_strategy_parameter_widgets["gasp"]


@pytest.mark.gui
def test_recognition_strategy_parameters_are_fully_retranslated_to_russian(qtbot) -> None:
    from contour.widget import PolygonExtractionWidget

    widget = PolygonExtractionWidget()
    qtbot.addWidget(widget)
    widget.set_ui_language("ru")
    strategy_index = widget.metal_segmentation_strategy_combo.findData("graph_multi_separator")
    widget.metal_segmentation_strategy_combo.setCurrentIndex(strategy_index)

    sensitivity = widget.metal_strategy_parameter_widgets["graph_multi_separator"]["gradient_field_sensitivity"]
    sensitivity_form = sensitivity.parentWidget().layout()
    sensitivity_label = sensitivity_form.labelForField(sensitivity)
    solver = widget.metal_strategy_parameter_widgets["graph_multi_separator"]["solver"]

    assert widget.metal_segmentation_strategy_combo.itemText(strategy_index) == ("Графовый мульти-разделитель")
    assert widget.metal_strategy_parameters_group.title() == ("Параметры: Графовый мульти-разделитель")
    assert sensitivity_label.text() == "Чувствительность градиентного поля"
    assert sensitivity.toolTip().startswith("Параметр выбранного алгоритма:")
    assert [solver.itemText(index) for index in range(solver.count())] == [
        "Жадное сокращение разделителя",
        "Жадное наращивание разделителя",
    ]
    assert {
        group.title() for group in widget.metal_strategy_parameter_stack.currentWidget().findChildren(QGroupBox)
    } == {"Основные", "Дополнительные"}


@pytest.mark.gui
def test_recognition_ui_shows_only_parameters_used_by_selected_pipeline(qtbot) -> None:
    from contour.widget import PolygonExtractionWidget

    widget = PolygonExtractionWidget()
    qtbot.addWidget(widget)

    def select(strategy: str) -> None:
        index = widget.metal_segmentation_strategy_combo.findData(strategy)
        widget.metal_segmentation_strategy_combo.setCurrentIndex(index)

    select("auto")
    assert not widget.metal_min_contrast_widget.isHidden()
    assert not widget.metal_gap_bridge_spin.isHidden()
    assert not widget.metal_speckle_removal_spin.isHidden()
    assert not widget.metal_auto_contrast_step_spin.isHidden()
    assert not widget.metal_adaptive_group.isHidden()
    assert not widget.metal_watershed_group.isHidden()
    assert widget.metal_strategy_parameters_group.isHidden()

    select("gradient_watershed")
    assert widget.metal_min_contrast_widget.isHidden()
    assert widget.metal_gap_bridge_spin.isHidden()
    assert not widget.metal_speckle_removal_spin.isHidden()
    assert widget.metal_auto_contrast_step_spin.isHidden()
    assert not widget.metal_watershed_group.isHidden()

    select("local_adaptive")
    assert not widget.metal_min_contrast_widget.isHidden()
    assert not widget.metal_gap_bridge_spin.isHidden()
    assert not widget.metal_speckle_removal_spin.isHidden()
    assert widget.metal_auto_contrast_step_spin.isHidden()
    assert not widget.metal_adaptive_group.isHidden()
    assert widget.metal_watershed_group.isHidden()

    select("graph_multi_separator")
    assert widget.metal_min_contrast_widget.isHidden()
    assert widget.metal_gap_bridge_spin.isHidden()
    assert widget.metal_speckle_removal_spin.isHidden()
    assert widget.metal_auto_contrast_step_spin.isHidden()
    assert widget.metal_adaptive_group.isHidden()
    assert widget.metal_watershed_group.isHidden()
    assert not widget.metal_strategy_parameters_group.isHidden()
