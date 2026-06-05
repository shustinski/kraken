from __future__ import annotations

import unittest

from contour.application.frame_layers import (
    build_additional_layer_frame_map,
    build_base_frame_number_map,
    build_base_frame_records,
    extract_frame_number,
    sort_base_frame_records,
)


class FrameLayerMappingTests(unittest.TestCase):
    def test_extract_frame_number(self) -> None:
        self.assertEqual(extract_frame_number("xxxx_1.png"), 1)
        self.assertEqual(extract_frame_number("xxxx_1237.tif"), 1237)
        self.assertEqual(extract_frame_number("sample_name_00042.jpg"), 42)
        self.assertIsNone(extract_frame_number("sample_name.jpg"))

    def test_sort_uses_numeric_suffix(self) -> None:
        ordered = sort_base_frame_records(
            [
                "demo_10.png",
                "demo_2.png",
                "demo_1237.png",
                "demo_1.png",
            ]
        )
        self.assertEqual([record.frame_number for record in ordered], [1, 2, 10, 1237])

    def test_build_base_map_ignores_duplicate_numbers_deterministically(self) -> None:
        records, warnings = build_base_frame_records(["frame_1.png", "frame_001.png", "frame_2.png"])
        frame_map = build_base_frame_number_map(records)
        self.assertIn(1, frame_map.values())
        self.assertTrue(any(path in frame_map for path in ("frame_1.png", "frame_001.png")))
        self.assertEqual(frame_map["frame_2.png"], 2)
        self.assertTrue(any("Duplicate base frame number 1" in message for message in warnings))

    def test_additional_layer_map_ignores_frames_outside_base(self) -> None:
        frame_map, warnings = build_additional_layer_frame_map(
            ["layer_1.png", "layer_5.png"],
            base_frame_numbers={1, 2},
        )
        self.assertEqual(frame_map, {1: "layer_1.png"})
        self.assertTrue(any("not present in base layer" in message for message in warnings))

    def test_additional_layer_missing_frame_keeps_other_frames(self) -> None:
        frame_map, warnings = build_additional_layer_frame_map(
            ["layer_2.png"],
            base_frame_numbers={1, 2, 3},
        )
        self.assertEqual(frame_map, {2: "layer_2.png"})
        self.assertEqual(warnings, [])

