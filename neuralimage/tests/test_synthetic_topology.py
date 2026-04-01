import numpy as np
import cv2
import random

from augmentations import SyntheticTopologyGenerator, SyntheticTopologyParameters
from augmentations.pcb_defects import PCBDefectAugmentor
from lib.data_interfaces import PCBDefectParameters


def _count_corner_pixels(mask: np.ndarray) -> int:
    count = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        up = y > 0 and mask[y - 1, x]
        down = y < mask.shape[0] - 1 and mask[y + 1, x]
        left = x > 0 and mask[y, x - 1]
        right = x < mask.shape[1] - 1 and mask[y, x + 1]
        if (up or down) and (left or right):
            count += 1
    return count


def _count_internal_endpoints(mask: np.ndarray) -> int:
    height, width = mask.shape
    count = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if not mask[y, x]:
                continue
            neighbors = int(mask[y - 1, x]) + int(mask[y + 1, x]) + int(mask[y, x - 1]) + int(mask[y, x + 1])
            if neighbors == 1:
                count += 1
    return count


def test_synthetic_topology_generator_is_deterministic_for_same_seed():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(trace_count=4, segment_count=3, trace_half_width=2)
    )

    image_a, label_a = generator.generate(size_hw=(96, 128), channels=1, seed=17)
    image_b, label_b = generator.generate(size_hw=(96, 128), channels=1, seed=17)

    assert np.array_equal(image_a, image_b)
    assert np.array_equal(label_a, label_b)


def test_ic_topology_varies_between_seeds():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=9,
            segment_count_range=(2, 4),
            trace_half_width_range=(1, 2),
            topology_domain='ic',
            topology_family='ic_mixed',
        )
    )

    _image_a, label_a = generator.generate(size_hw=(160, 160), channels=1, seed=101)
    _image_b, label_b = generator.generate(size_hw=(160, 160), channels=1, seed=102)

    assert not np.array_equal(label_a, label_b)


def test_synthetic_topology_generator_returns_consistent_image_and_mask():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(trace_count=5, segment_count=4, trace_half_width=2)
    )

    image, label = generator.generate(size_hw=(96, 128), channels=1, seed=29)
    mask = label[0] > 0.5

    assert image.shape == (1, 96, 128)
    assert label.shape == (1, 96, 128)
    assert int(np.count_nonzero(mask)) > 0
    assert float(image[0][mask].mean()) > float(image[0][~mask].mean())


def test_synthetic_topology_generator_keeps_traces_disconnected():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(trace_count=4, segment_count=4, trace_half_width=2, trace_clearance=3)
    )

    _image, label = generator.generate(size_hw=(192, 192), channels=1, seed=41)
    mask = (label[0] > 0.5).astype(np.uint8)
    component_count, _labels = cv2.connectedComponents(mask, connectivity=8)

    assert component_count - 1 >= 4


def test_synthetic_topology_polyline_has_no_self_intersections():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(trace_count=1, segment_count=8, trace_half_width=1)
    )
    rng = random.Random(123)

    polyline = generator._build_trace_polyline(160, 160, rng, segment_count=8)
    segments = list(zip(polyline[:-1], polyline[1:]))

    assert len(segments) >= 1
    for index, segment in enumerate(segments):
        for other_index, other_segment in enumerate(segments):
            if other_index >= index:
                continue
            if abs(index - other_index) <= 1:
                continue
            assert generator._segments_intersect(segment[0], segment[1], other_segment[0], other_segment[1]) is False


def test_synthetic_topology_supports_single_segment_range():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=3,
            segment_count_range=(1, 1),
            trace_half_width_range=(1, 2),
        )
    )

    image, label = generator.generate(size_hw=(96, 128), channels=1, seed=77)

    assert image.shape == (1, 96, 128)
    assert label.shape == (1, 96, 128)
    assert int(np.count_nonzero(label)) > 0


def test_ic_router_generation_handles_small_canvas():
    for family in ('ic_channels', 'ic_cell_array', 'ic_tree'):
        generator = SyntheticTopologyGenerator(
            SyntheticTopologyParameters(
                trace_count=6,
                segment_count_range=(1, 3),
                trace_half_width_range=(1, 2),
                topology_domain='ic',
                topology_family=family,
            )
        )

        image, label = generator.generate(size_hw=(32, 32), channels=1, seed=11)

        assert image.shape == (1, 32, 32)
        assert label.shape == (1, 32, 32)
        assert int(np.count_nonzero(label)) > 0


def test_ic_route_half_width_is_sampled_from_range():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_half_width_range=(1, 4),
            topology_domain='ic',
            topology_family='ic_channels',
        )
    )
    rng = random.Random(7)

    sampled = {generator._sample_ic_route_half_width(rng) for _ in range(24)}

    assert sampled >= {1, 2, 3, 4}


def test_ic_mixed_includes_boundary_spanning_and_local_topology():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=12,
            segment_count_range=(2, 4),
            trace_half_width_range=(1, 3),
            topology_domain='ic',
            topology_family='ic_mixed',
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=17)
    mask = (label[0] > 0.5).astype(np.uint8)
    touches_boundary = bool(mask[:, 0].any() or mask[:, -1].any() or mask[0, :].any() or mask[-1, :].any())
    dense_rows = int(np.count_nonzero(mask.sum(axis=1) > 20))
    dense_cols = int(np.count_nonzero(mask.sum(axis=0) > 20))

    assert touches_boundary is True
    assert dense_rows >= 35
    assert dense_cols >= 35


def test_ic_channels_fill_frame_under_high_trace_count():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=100,
            segment_count_range=(2, 4),
            trace_half_width_range=(1, 3),
            topology_domain='ic',
            topology_family='ic_channels',
        )
    )

    _image, label = generator.generate(size_hw=(192, 192), channels=1, seed=21)
    mask = (label[0] > 0.5).astype(np.uint8)

    assert float(mask.mean()) >= 0.25
    assert int(np.count_nonzero(mask.sum(axis=1) > 40)) >= 100
    assert int(np.count_nonzero(mask.sum(axis=0) > 40)) >= 100


def test_parallel_synthetic_topology_supports_short_defect():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(1, 2),
            trace_half_width_range=(2, 3),
            topology_family='parallel',
            via_count_range=(0, 0),
        )
    )
    image, label = generator.generate(size_hw=(160, 160), channels=1, seed=91)
    names = ('break', 'short', 'missing_copper', 'excess_copper', 'pinhole', 'spurious_copper', 'via', 'misalignment')
    augmentor = PCBDefectAugmentor(
        PCBDefectParameters(
            enabled=True,
            defect_probability=1.0,
            min_defects=1,
            max_defects=1,
            defect_probabilities={name: (1.0 if name == 'short' else 0.0) for name in names},
            defect_severities={name: (0.8 if name == 'short' else 0.5) for name in names},
        )
    )

    _augmented_image, defect_mask, _augmented_mask = augmentor(image, label, seed=13, return_augmented_mask=True)

    assert int(np.count_nonzero(defect_mask)) > 0


def test_pcb_parallel_topology_contains_full_span_traces():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=10,
            segment_count_range=(1, 2),
            trace_half_width_range=(2, 3),
            topology_domain='pcb',
            topology_family='pcb_parallel',
        )
    )

    _image, label = generator.generate(size_hw=(192, 192), channels=1, seed=9)
    mask = (label[0] > 0.5).astype(np.uint8)
    touches_left = bool(np.any(mask[:, 0] > 0))
    touches_right = bool(np.any(mask[:, -1] > 0))
    touches_top = bool(np.any(mask[0, :] > 0))
    touches_bottom = bool(np.any(mask[-1, :] > 0))

    assert (touches_left and touches_right) or (touches_top and touches_bottom)


def test_pcb_topology_can_render_rgb_image():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(1, 3),
            trace_half_width_range=(2, 3),
            topology_domain='pcb',
            topology_family='pcb_parallel',
        )
    )

    image, label = generator.generate(size_hw=(128, 128), channels=3, seed=23)

    assert image.shape == (3, 128, 128)
    assert label.shape == (1, 128, 128)
    assert not np.array_equal(image[0], image[1])


def test_cell_array_synthetic_topology_supports_via_defect():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(2, 4),
            trace_half_width_range=(2, 3),
            topology_family='cell_array',
            via_count_range=(2, 3),
        )
    )
    image, label = generator.generate(size_hw=(160, 160), channels=1, seed=113)
    names = ('break', 'short', 'missing_copper', 'excess_copper', 'pinhole', 'spurious_copper', 'via', 'misalignment')
    augmentor = PCBDefectAugmentor(
        PCBDefectParameters(
            enabled=True,
            defect_probability=1.0,
            min_defects=1,
            max_defects=1,
            defect_probabilities={name: (1.0 if name == 'via' else 0.0) for name in names},
            defect_severities={name: (0.8 if name == 'via' else 0.5) for name in names},
        )
    )

    _augmented_image, defect_mask, _augmented_mask = augmentor(image, label, seed=19, return_augmented_mask=True)

    assert int(np.count_nonzero(defect_mask)) > 0


def test_ic_family_mapping_generates_repetitive_topology():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=9,
            segment_count_range=(2, 3),
            trace_half_width_range=(1, 2),
            topology_domain='ic',
            topology_family='ic_cell_array',
            via_count_range=(1, 2),
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=37)
    mask = (label[0] > 0.5).astype(np.uint8)
    dense_cols = np.count_nonzero(mask.sum(axis=0) > 20)
    component_count, _labels = cv2.connectedComponents(mask, connectivity=8)
    ys, xs = np.nonzero(mask)

    assert dense_cols >= 12
    assert component_count - 1 >= 3
    assert int(xs.max()) - int(xs.min()) >= 80
    assert int(ys.max()) - int(ys.min()) >= 24


def test_ic_channel_topology_spans_broad_vertical_extent():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=9,
            segment_count_range=(2, 4),
            trace_half_width_range=(1, 2),
            topology_domain='ic',
            topology_family='ic_channels',
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=17)
    mask = (label[0] > 0.5).astype(np.uint8)
    ys, _xs = np.nonzero(mask)

    assert int(ys.min()) <= 30
    assert int(ys.max()) >= 130


def test_ic_channels_generate_many_turns():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=9,
            segment_count_range=(2, 4),
            trace_half_width_range=(1, 2),
            topology_domain='ic',
            topology_family='ic_channels',
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=17)
    mask = (label[0] > 0.5)

    assert _count_corner_pixels(mask) >= 1000


def test_ic_channels_have_internal_trace_endpoints():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=9,
            segment_count_range=(2, 4),
            trace_half_width_range=(1, 2),
            topology_domain='ic',
            topology_family='ic_channels',
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=17)
    mask = (label[0] > 0.5)

    assert _count_internal_endpoints(mask) >= 4


def test_ic_tree_backbone_contains_multiple_turn_levels():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(1, 3),
            trace_half_width_range=(2, 3),
            topology_domain='ic',
            topology_family='ic_tree',
        )
    )

    points = generator._build_ic_tree_backbone_points(160, 80, 20, 140, random.Random(11))

    assert len({y for _x, y in points}) >= 4


def test_ic_tree_branch_polyline_forms_l_shape():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(1, 3),
            trace_half_width_range=(2, 3),
            topology_domain='ic',
            topology_family='ic_tree',
        )
    )

    points = generator._build_ic_tree_branch_polyline((80, 80), 160, 20, 140, random.Random(13), direction=1, max_length=40)

    assert len(points) >= 3
    assert len({point[0] for point in points}) >= 2
    assert len({point[1] for point in points}) >= 2


def test_ic_tree_spans_multiple_vertical_bands():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(1, 3),
            trace_half_width_range=(2, 3),
            topology_domain='ic',
            topology_family='ic_tree',
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=17)
    mask = (label[0] > 0.5).astype(np.uint8)
    ys, _xs = np.nonzero(mask)

    assert int(ys.min()) <= 24
    assert int(ys.max()) >= 130


def test_ic_tree_contains_long_local_trunk():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(1, 3),
            trace_half_width_range=(2, 3),
            topology_domain='ic',
            topology_family='ic_tree',
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=55)
    mask = (label[0] > 0.5).astype(np.uint8)
    longest_row = int(mask.sum(axis=1).max())
    longest_col = int(mask.sum(axis=0).max())

    assert max(longest_row, longest_col) >= 70


def test_ic_tree_contains_multiple_branches():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(1, 3),
            trace_half_width_range=(2, 3),
            topology_domain='ic',
            topology_family='ic_tree',
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=56)
    mask = (label[0] > 0.5).astype(np.uint8)
    row_strength = mask.sum(axis=1)
    col_strength = mask.sum(axis=0)
    branch_lines = max(
        int(np.count_nonzero(col_strength > 18)),
        int(np.count_nonzero(row_strength > 18)),
    )

    assert branch_lines >= 6


def test_ic_tree_contains_multiple_independent_conductors():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=8,
            segment_count_range=(1, 3),
            trace_half_width_range=(2, 3),
            topology_domain='ic',
            topology_family='ic_tree',
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=57)
    component_count, _labels = cv2.connectedComponents((label[0] > 0.5).astype(np.uint8), connectivity=8)

    assert component_count - 1 >= 2


def test_ic_topology_does_not_generate_pcb_via_holes():
    generator = SyntheticTopologyGenerator(
        SyntheticTopologyParameters(
            trace_count=10,
            segment_count_range=(2, 3),
            trace_half_width_range=(1, 2),
            topology_domain='ic',
            topology_family='ic_cell_array',
            via_count_range=(2, 4),
        )
    )

    _image, label = generator.generate(size_hw=(160, 160), channels=1, seed=71)
    augmentor = PCBDefectAugmentor(PCBDefectParameters(enabled=True))
    via_candidates = augmentor._detect_via_holes((label[0] > 0.5).astype(np.uint8) * 255)

    assert via_candidates == []
